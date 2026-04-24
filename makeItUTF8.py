import pandas as pd
import unicodedata

INPUT = "full_dataset.csv"
OUTPUT = "clean_dataset_ascii.csv"

def remove_accents(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

# load with safe encoding
df = pd.read_csv(INPUT, encoding="latin-1")

# apply cleaning to text columns
for col in df.columns:
    if df[col].dtype == "object":  # only text columns
        df[col] = df[col].apply(remove_accents)

# save clean UTF-8 file
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print("✅ Cleaned file saved as:", OUTPUT)