import os
import random
import shutil
from tqdm import tqdm
import json
from pathlib import Path

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
IMAGE_ROOT     = "/workspace/compressed_dataset"      # folder with country subfolders
OUTPUT_DIR     = "/workspace/inference_set_1000"
OUTPUT_JSONL   = "/workspace/inference_set_1000/metadata.jsonl"
N_SAMPLES      = 1000
SEED           = 42
random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Grab every image + extract country from folder name
# ------------------------------------------------------------------
all_images = []
print("Scanning dataset and extracting country names...")

for img_path in Path(IMAGE_ROOT).rglob("canvas_*.jpg"):
    if not img_path.is_file():
        continue

    # Country is the direct parent folder
    country_folder = img_path.parent.name
    # Optional: clean country name (some folders have spaces, underscores, etc.)
    country_clean = country_folder.replace("_", " ").strip().lower()

    all_images.append({
        "src_path": str(img_path),
        "country": country_clean,
        "country_folder": country_folder
    })

print(f"Found {len(all_images):,} images across {len(set(i['country'] for i in all_images))} countries")

# ------------------------------------------------------------------
# Random sample
# ------------------------------------------------------------------
selected = random.sample(all_images, N_SAMPLES)
records = []

print(f"Copying {N_SAMPLES} images...")
for i, item in enumerate(tqdm(selected)):
    src = item["src_path"]
    country = item["country"]
    
    # Unique filename
    orig_name = Path(src).name
    new_name = f"{i:05d}_{country}_{orig_name}"
    dst = os.path.join(OUTPUT_DIR, new_name)
    
    shutil.copy2(src, dst)
    
    records.append({
        "filename": new_name,
        "image_path": dst,
        "true_country": country,                    # ← HERE IT IS!
        "original_path": src,
        "country_folder": item["country_folder"]
    })

# ------------------------------------------------------------------
# Save metadata (JSONL + pretty JSON)
# ------------------------------------------------------------------
with open(OUTPUT_JSONL, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

with open("/workspace/inference_set_1000/metadata.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"\nDone! Copied {len(records)} images")
print(f"→ Images: {OUTPUT_DIR}")
print(f"→ Metadata: {OUTPUT_JSONL} and metadata.json")
print(f"Example entry:")
print(json.dumps(records[0], indent=2))