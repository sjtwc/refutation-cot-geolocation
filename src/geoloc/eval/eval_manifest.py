# /workspace/evaluate_final.py
import os
import json
import re
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from math import radians, sin, cos, sqrt, atan2

# ================================
# CONFIG
# ================================
MODEL_PATH       = "/workspace/llava-geoguessr-final-normal"
BASE_MODEL       = "llava-hf/llava-v1.6-mistral-7b-hf"
INFERENCE_DIR    = "/workspace/inference_set_1000"
METADATA_JSONL   = os.path.join(INFERENCE_DIR, "metadata.jsonl")
CENTROIDS_JSON   = "/workspace/folder_to_coords-2.json"
RESULTS_JSON     = "/workspace/FINAL_EVALUATION_1000.json"

# Load country → centroid mapping
with open(CENTROIDS_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)

CENTROIDS = {}
for country, info in raw.items():
    key = country.lower()
    CENTROIDS[key] = (info["lat"], info["lon"])
    # Add common aliases
    if "united states" in key:
        for a in ["usa", "us", "america", "united states of america"]:
            CENTROIDS[a] = (info["lat"], info["lon"])
    if "united kingdom" in key:
        for a in ["uk", "britain", "england", "great britain"]:
            CENTROIDS[a] = (info["lat"], info["lon"])

# Load model (exactly like your working inference.py)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
base = LlavaNextForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, MODEL_PATH)
model.eval()

# Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Robust coordinate extraction
def extract_coords(text):
    patterns = [
        r'([+-]?\d+\.?\d*)\s*[,°]\s*([+-]?\d+\.?\d*)',
        r'lat.*?([+-]?\d+\.?\d*).*?lon.*?([+-]?\d+\.?\d*)',
        r'(\d+\.\d+).*?([+-]?\d+\.\d+)',
        r'([+-]?\d+\.\d+)\s*[ns].*?([+-]?\d+\.\d+)\s*[ew]',
    ]
    text = text.lower()
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                lat = float(m.group(1))
                lon = float(m.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except:
                continue
    return None

# Load test set
with open(METADATA_JSONL) as f:
    samples = [json.loads(line) for line in f]

print(f"Starting evaluation on {len(samples)} images...")

results = []
for item in tqdm(samples, desc="Inference"):
    img_path = item["image_path"]
    filename = item["filename"]
    true_country = item["country"].lower()

    # Ground truth centroid
    true_lat, true_lon = CENTROIDS.get(true_country, (0.0, 0.0))

    image = Image.open(img_path).convert("RGB")

    prompt = "<|im_start|>user\n<image>\nWhere in the world is this GeoGuessr location? Respond with the country and approximate coordinates (latitude, longitude).<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    response = processor.decode(output[0], skip_special_tokens=True)
    answer = response.split("assistant\n")[-1].strip()

    # Try to get coordinates
    coords = extract_coords(answer)
    if coords:
        pred_lat, pred_lon = coords
    else:
        # Fallback: try to extract country from response
        pred_country = None
        for c in CENTROIDS:
            if c in answer.lower():
                pred_country = c
                break
        pred_lat, pred_lon = CENTROIDS.get(pred_country, (true_lat, true_lon))

    distance_km = haversine(true_lat, true_lon, pred_lat, pred_lon)
    country_correct = any(word in answer.lower() for word in [true_country, item["country"].lower()])

    results.append({
        "filename": filename,
        "true_country": item["country"],
        "true_lat": true_lat,
        "true_lon": true_lon,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "distance_km": round(distance_km, 2),
        "country_correct": country_correct,
        "response": answer,
    })

# ================================
# Final Results
# ================================
df = pd.DataFrame(results)
d = df["distance_km"]

print("\n" + "="*70)
print("FINAL GEOGUESSR LEADERBOARD RESULTS (1000 images)")
print("="*70)
print(f"Country Accuracy        : {df['country_correct'].mean()*100:6.2f}%")
print(f"Median error            : {d.median():8.1f} km")
print(f"Mean error              : {d.mean():8.1f} km")
print(f"@1 km     (street)      : {(d <= 1).mean()*100:6.2f}%")
print(f"@25 km    (city)        : {(d <= 25).mean()*100:6.2f}%")
print(f"@200 km   (regional)    : {(d <= 200).mean()*100:6.2f}%")
print(f"@750 km   (country)     : {(d <= 750).mean()*100:6.2f}%")
print(f"@2500 km  (continental) : {(d <= 2500).mean()*100:6.2f}%")
print("="*70)

# Save everything
df.to_json(RESULTS_JSON, orient="records", indent=2)
print(f"Full results saved → {RESULTS_JSON}")