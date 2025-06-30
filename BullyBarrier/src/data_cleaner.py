import re
import pandas as pd

def clean_raw_data(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Attempt to match patterns like: 'text': True or "text": False
    pattern = re.compile(r"""['"](.+?)['"]\s*:\s*(True|False)""", re.DOTALL)
    matches = pattern.findall(raw_text)

    # Convert to DataFrame
    data = pd.DataFrame(matches, columns=["text", "label"])
    data["label"] = data["label"].map({"True": 1, "False": 0})  # Convert to 1/0

    # Drop empty or very short texts
    data = data[data["text"].str.strip().str.len() > 5]

    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned data saved to: {output_path}")

    return data
