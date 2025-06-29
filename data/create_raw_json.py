import ast
import json
import os

# Paths
data_dir = os.path.dirname(__file__)  # same folder as this script
raw_txt_path = os.path.join(data_dir, "raw_data.txt")
json_out_path = os.path.join(data_dir, "raw_data.json")

# Read raw_data.txt as a Python dict
with open(raw_txt_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

try:
    raw_dict = ast.literal_eval(raw_text)
except Exception as e:
    print("❌ Failed to parse raw_data.txt. Check syntax.")
    print(e)
    exit(1)

# Write as valid JSON
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(raw_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Converted raw_data.txt → raw_data.json with {len(raw_dict)} entries.")
