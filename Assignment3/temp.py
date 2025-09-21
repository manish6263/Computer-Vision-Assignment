import json
import os

# Input and output file paths
input_file = 'input.json'  # Replace with your actual input filename
output_file = 'converted_output.json'

# Load the input JSON
with open(input_file, 'r') as f:
    input_data = json.load(f)

# Build image mapping and images list
image_map = {}
images = []
image_id_counter = 1

for item in input_data:
    orig_file = item["file_name"]
    if orig_file not in image_map:
        new_file_name = f"{orig_file}"  # New file name format
        image_map[orig_file] = image_id_counter
        images.append({
            "id": image_id_counter,
            "file_name": new_file_name
        })
        image_id_counter += 1

# Build annotations list
annotations = []
for item in input_data:
    annotations.append({
        "id": item["id"],
        "image_id": image_map[item["file_name"]],
        "category_id": item["category_id"],
        "bbox": item["bbox"],
        "score": item["score"]
    })

# Final output dictionary
output_data = {
    "images": images,
    "annotations": annotations
}

# Save to output JSON file
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Conversion complete. Saved to: {output_file}")
