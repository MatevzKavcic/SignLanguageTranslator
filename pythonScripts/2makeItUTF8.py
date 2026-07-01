import pandas as pd# type: ignore
import unicodedata

INPUT = "1.1SecondTryDataset.csv"
OUTPUT = "2SecondTryDatasetUTF8.csv"

# -----------------------------
# 1. REMOVE ACCENTS FUNCTION
# -----------------------------
def remove_accents(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text)\
            .encode('ascii', 'ignore')\
            .decode('ascii')
    return text


# -----------------------------
# 2. LOAD CSV WITH CORRECT ENCODING
# -----------------------------
# Try common encodings (order matters)
encodings_to_try = ["utf-8", "cp1250", "cp1252", "latin-1"]

df = None

for enc in encodings_to_try:
    try:
        print(f"Trying encoding: {enc}")
        df = pd.read_csv(INPUT, encoding=enc)
        print(f" Successfully loaded with: {enc}")
        break
    except Exception as e:
        print(f" Failed with {enc}")

if df is None:
    raise Exception("Could not read the CSV with any encoding.")


# -----------------------------
# 3. DEBUG (optional but useful)
# -----------------------------
print("\nSample BEFORE cleaning:")
print(df.head(3))


# -----------------------------
# 4. CLEAN TEXT COLUMNS
# -----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].apply(remove_accents)


# -----------------------------
# 5. DEBUG AFTER
# -----------------------------
print("\nSample AFTER cleaning:")
print(df.head(3))


# -----------------------------
# 6. SAVE CLEAN FILE (UTF-8)
# -----------------------------
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print("\n Cleaned file saved as:", OUTPUT)
