import os
import json
import glob

# Define category mapping for YOLO format (category names to class IDs)
category_mapping = {
    "car": 0,
    "bus": 1,
    "truck": 2
}

# BDD100K dataset label path (modify to your actual label directory)
label_dir = "path/to/json/labels"

# Image dimensions (standard size for BDD100K dataset)
image_width = 1280
image_height = 720

# Output directory for YOLO formatted label files
output_dir = "path/to/yolo/txt"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all JSON label files recursively in label_dir
all_json_files = glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True)

# Sort the files alphabetically by filename
all_json_files_sorted = sorted(all_json_files)

# Show how many files were found in total
print(f"Found {len(all_json_files_sorted)} JSON files in total.")

# Select only the number of you want sorted JSON files
selected_json_files = all_json_files_sorted[:]

print(f"Processing {len(selected_json_files)} sorted JSON files...")

# Iterate over each selected JSON file
for idx, json_file in enumerate(selected_json_files, 1):
    with open(json_file, "r") as f:
        data = json.load(f)

    file_name = data.get("name", None)
    if not file_name:
        print(f"[Warning] No 'name' field found in {json_file}")
        continue

    output_path = os.path.join(output_dir, file_name + ".txt")
    object_count = 0

    with open(output_path, "w") as out_file:
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category")
                if category not in category_mapping:
                    continue  # Skip non-matching categories
                box = obj.get("box2d")
                if not box:
                    continue  # Skip if no bounding box (box2d)

                class_id = category_mapping[category]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height

                # Write to the YOLO label file
                out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                object_count += 1

    # Ensure an empty label file is created if no objects were found
    if object_count == 0:
        with open(output_path, "w") as out_file:
            out_file.write("")  # Create an empty file for images with no objects

    if idx % 500 == 0 or idx == len(selected_json_files):
        print(f"Processed {idx}/{len(selected_json_files)} files.")

print("Conversion complete! Processed xxx sorted JSON files.")
