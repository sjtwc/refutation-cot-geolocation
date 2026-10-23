# /workspace/make_inference_set_4800.py

import os
import random
import shutil
from tqdm import tqdm
import json

IMAGE_ROOT   = "/workspace/compressed_dataset"
OUTPUT_DIR   = "/workspace/inference_set_4850"
OUTPUT_JSONL = "/workspace/inference_set_4850/metadata.jsonl"
N_SAMPLES    = 4800
SEED         = 42

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all images + infer country from parent folder
all_images = []
print("Scanning all country folders...")
for country in os.listdir(IMAGE_ROOT):
    country_dir = os.path.join(IMAGE_ROOT, country)
    if not os.path.isdir(country_dir):
        continue
    for filename in os.listdir(country_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(country_dir, filename)
            all_images.append({
                "path": full_path,
                "country": country
            })

print(f"Found {len(all_images):,} images across {len(set(i['country'] for i in all_images))} countries")

# Sample 4800
selected = random.sample(all_images, N_SAMPLES)

records = []
for idx, item in enumerate(tqdm(selected, desc="Copying 4800 images")):
    src = item["path"]
    new_name = f"{idx:05d}_{item['country']}_{os.path.basename(src)}"
    dst = os.path.join(OUTPUT_DIR, new_name)
    shutil.copy2(src, dst)

    records.append({
        "image_path": dst,
        "filename": new_name,
        "country": item["country"]
    })

# Save metadata (you’ll enrich with lat/lon later if needed)
with open(OUTPUT_JSONL, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"\nDone! 4850-image inference set ready:")
print(f"   → {OUTPUT_DIR}")
print(f"   → {OUTPUT_JSONL}")
print("(Country is already in the filename and metadata – perfect for quick evaluation)")