import json

# Files
input_path = "annotations.json"
output_path = "annotations_viewable.json"

# Read NDJSON (newline-delimited JSON)
with open(input_path, "r") as infile:
    lines = infile.readlines()

# Parse each line into a JSON object
data = []
for line in lines:
    try:
        data.append(json.loads(line.strip()))
    except json.JSONDecodeError as e:
        print("Skipping line due to JSON error:", e)

# Write a proper JSON list to new file
with open(output_path, "w") as outfile:
    json.dump(data, outfile, indent=2)

print("✅ annotations_viewable.json is ready.")
