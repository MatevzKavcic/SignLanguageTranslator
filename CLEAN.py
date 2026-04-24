import pandas as pd

CSV_FILE = "full_dataset.csv"  # change here for your dataset path
OUTPUT_FILE = "cleaned_full_dataset.csv"


df = pd.read_csv(CSV_FILE)

exclude = ["frame", "video", "label"]
landmark_cols = [col for col in df.columns if col not in exclude]

# group columns into landmarks (x,y,z)
landmarks = []
for i in range(0, len(landmark_cols), 3):
    landmarks.append(landmark_cols[i:i+3])

cleaned_groups = []

for video_name, group in df.groupby("video"):
    print(f"Processing {video_name}")
    group = group.reset_index(drop=True)

    # -------- INTERPOLATION --------
    for cols in landmarks:
        x_col, y_col, z_col = cols

        i = 0
        while i < len(group):
            x = group.loc[i, x_col]
            y = group.loc[i, y_col]
            z = group.loc[i, z_col]

            if x == 0 and y == 0 and z == 0:
                start = i

                while i < len(group) and (
                    group.loc[i, x_col] == 0 and
                    group.loc[i, y_col] == 0 and
                    group.loc[i, z_col] == 0
                ):
                    i += 1

                end = i - 1

                before_idx = start - 1
                after_idx = i

                if before_idx < 0 or after_idx >= len(group):
                    continue

                before = [
                    group.loc[before_idx, x_col],
                    group.loc[before_idx, y_col],
                    group.loc[before_idx, z_col],
                ]

                after = [
                    group.loc[after_idx, x_col],
                    group.loc[after_idx, y_col],
                    group.loc[after_idx, z_col],
                ]

                gap_size = end - start + 1

                for j in range(gap_size):
                    t = (j + 1) / (gap_size + 1)

                    group.loc[start + j, x_col] = before[0] * (1 - t) + after[0] * t
                    group.loc[start + j, y_col] = before[1] * (1 - t) + after[1] * t
                    group.loc[start + j, z_col] = before[2] * (1 - t) + after[2] * t
            else:
                i += 1

    # -------- HANDLE START & END --------
    for cols in landmarks:
        x_col, y_col, z_col = cols

        # ---- START ----
        first_valid_idx = None
        for i in range(len(group)):
            if not (
                group.loc[i, x_col] == 0 and
                group.loc[i, y_col] == 0 and
                group.loc[i, z_col] == 0
            ):
                first_valid_idx = i
                break

        if first_valid_idx is not None:
            first_val = [
                group.loc[first_valid_idx, x_col],
                group.loc[first_valid_idx, y_col],
                group.loc[first_valid_idx, z_col],
            ]

            for i in range(0, first_valid_idx):
                group.loc[i, x_col] = first_val[0]
                group.loc[i, y_col] = first_val[1]
                group.loc[i, z_col] = first_val[2]

        # ---- END ----
        last_valid_idx = None
        for i in range(len(group)-1, -1, -1):
            if not (
                group.loc[i, x_col] == 0 and
                group.loc[i, y_col] == 0 and
                group.loc[i, z_col] == 0
            ):
                last_valid_idx = i
                break

        if last_valid_idx is not None:
            last_val = [
                group.loc[last_valid_idx, x_col],
                group.loc[last_valid_idx, y_col],
                group.loc[last_valid_idx, z_col],
            ]

            for i in range(last_valid_idx + 1, len(group)):
                group.loc[i, x_col] = last_val[0]
                group.loc[i, y_col] = last_val[1]
                group.loc[i, z_col] = last_val[2]

    cleaned_groups.append(group)

# merge back
df_cleaned = pd.concat(cleaned_groups, ignore_index=True)

df_cleaned.to_csv(OUTPUT_FILE, index=False)

print("Done! Cleaned dataset saved.")