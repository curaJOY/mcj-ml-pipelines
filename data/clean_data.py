import re
import pandas as pd

data = []
with open("data_raw.txt", "r", encoding="utf-8") as f:
    for line in f:
        # This regex captures better-quoted strings and avoids cutting off words
        match = re.match(r'^\s*[\'\"]?(.*?)[\'\"]?\s*:\s*(True|False),?\s*$', line.strip())
        if match:
            text = match.group(1).strip()
            label = match.group(2) == "True"
            data.append((text, label))

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("improved_cleaned_data.csv", index=False)
