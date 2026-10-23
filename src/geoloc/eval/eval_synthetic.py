
import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
import re
import math

TEST_JSON = "/workspace/refutation_1k_final.json"
MODEL_PATH = "/workspace/llava-geoguessr-final"
BASE_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

with open(TEST_JSON, "r") as f:
    data = json.load(f)

test_lookup = {}
for item in data:
    img_path = item["image"].split("/")[-1]
    test_lookup[img_path] = {
        "country": item["country"].lower(),
        "lat": float(item["true_lat"]),
        "lon": float(item["true_lon"]),
    }

processor = AutoProcessor.from_pretrained(MODEL_PATH)
base = LlavaNextForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, MODEL_PATH)
model.eval()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def extract_lat_lon(text):
    text = text.lower()
    coord_pattern = r'([+-]?\d+\.?\d*)\s*[,°]\s*([+-]?\d+\.?\d*)'
    match = re.search(coord_pattern, text)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except:
            pass
    return None

results = []
total_images = len(test_lookup)
print(f"Evaluating {total_images} unseen images with distance metric...")

for img_name, info in tqdm(test_lookup.items(), total=total_images):
    img_path = f"/workspace/compressed_dataset/{info['country'].title()}/{img_name}"
    if not os.path.exists(img_path):
        continue
    image = Image.open(img_path).convert("RGB")

    prompt = "<|im_start|>user\n<image>\nYou are a world-class GeoGuessr player.Guess the exact latitude and longitude of this location. Respond with coordinates only, like: 3.14, 101.59<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    response = processor.decode(output[0], skip_special_tokens=True)
    answer = response.split("assistant")[-1].strip()

    coords = extract_lat_lon(answer)
    true_lat, true_lon = info["lat"], info["lon"]
    true_country = info["country"]

    if coords:
        pred_lat, pred_lon = coords
        distance_km = haversine(true_lat, true_lon, pred_lat, pred_lon)
        country_correct = (true_country in answer.lower()) or any(word in answer.lower() for word in ["malaysia", "thailand", "indonesia"])
    else:
        distance_km = float('inf')
        country_correct = False

    results.append({
        "image": img_name,
        "true_country": true_country,
        "true_lat": true_lat,
        "true_lon": true_lon,
        "predicted_coords": coords,
        "distance_km": distance_km if distance_km != float('inf') else None,
        "country_correct": country_correct,
        "response": answer
    })

valid_distances = [r["distance_km"] for r in results if r["distance_km"] is not None]
country_acc = sum(r["country_correct"] for r in results) / len(results) * 100

print("\n" + "="*80)
print("FINAL GEO GUESSR EVALUATION (UNSEEN DATA)")
print("="*80)
print(f"Total images: {len(results)}")
print(f"Country-level accuracy: {country_acc:.2f}%")
if valid_distances:
    print(f"Mean distance error : {sum(valid_distances)/len(valid_distances):.1f} km")
    print(f"Median distance error : {sorted(valid_distances)[len(valid_distances)//2]:.1f} km")
    print(f"Within 25 km : {sum(d <= 25 for d in valid_distances)/len(valid_distances)*100:.1f}%")
    print(f"Within 150 km : {sum(d <= 150 for d in valid_distances)/len(valid_distances)*100:.1f}%")
    print(f"Within 500 km : {sum(d <= 500 for d in valid_distances)/len(valid_distances)*100:.1f}%")
print("="*80)

with open("/workspace/geoguessr_full_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print("Full results saved to /workspace/geoguessr_full_evaluation.json")