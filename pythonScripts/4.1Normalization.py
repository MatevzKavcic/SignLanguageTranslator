import pandas as pd
import numpy as np

INPUT_CSV = "cleaned_test_extraction.csv"
OUTPUT_CSV = "test_extraction_normalized.csv"

df = pd.read_csv(INPUT_CSV)

# ======================================
# SHOULDER COLUMNS
# ======================================

LS_X = "pose_11_x"
LS_Y = "pose_11_y"

RS_X = "pose_12_x"
RS_Y = "pose_12_y"

# ======================================
# FIND ALL LANDMARK COLUMNS
# ======================================

x_cols = [c for c in df.columns if c.endswith("_x")]
y_cols = [c for c in df.columns if c.endswith("_y")]
z_cols = [c for c in df.columns if c.endswith("_z")]

print("X cols:", len(x_cols))
print("Y cols:", len(y_cols))
print("Z cols:", len(z_cols))

# ======================================
# NORMALIZE ROW BY ROW
# ======================================

total_frames = len(df)

print(f"\nStarting normalization...")
print(f"Total frames: {total_frames}\n")

for idx in df.index:

    # Progress log every 1000 frames
    if idx % 1000 == 0 or idx == total_frames - 1:
        percent = (idx + 1) / total_frames * 100
        print(
            f"[{idx+1}/{total_frames}] "
            f"{percent:.2f}% completed",
            flush=True
        )

    ls_x = df.at[idx, LS_X]
    ls_y = df.at[idx, LS_Y]

    rs_x = df.at[idx, RS_X]
    rs_y = df.at[idx, RS_Y]

    # Skip bad frames
    if (
        ls_x == 0 and ls_y == 0 and
        rs_x == 0 and rs_y == 0
    ):
        continue

    center_x = (ls_x + rs_x) / 2.0
    center_y = (ls_y + rs_y) / 2.0

    shoulder_dist = np.sqrt(
        (ls_x - rs_x) ** 2 +
        (ls_y - rs_y) ** 2
    )

    if shoulder_dist < 1e-6:
        continue

    # ----------------------------
    # X
    # ----------------------------
    for col in x_cols:
        value = df.at[idx, col]

        if value != 0:
            df.at[idx, col] = (
                value - center_x
            ) / shoulder_dist

    # ----------------------------
    # Y
    # ----------------------------
    for col in y_cols:
        value = df.at[idx, col]

        if value != 0:
            df.at[idx, col] = (
                value - center_y
            ) / shoulder_dist

    # ----------------------------
    # Z
    # ----------------------------
    for col in z_cols:
        value = df.at[idx, col]

        if value != 0:
            df.at[idx, col] = (
                value / shoulder_dist
            )
# ======================================
# SAVE
# ======================================


print("\n===================================")
print("Normalization finished")
print("===================================")
print(f"Frames processed : {total_frames}")
print(f"X coordinates    : {len(x_cols)}")
print(f"Y coordinates    : {len(y_cols)}")
print(f"Z coordinates    : {len(z_cols)}")
print(f"Output file      : {OUTPUT_CSV}")
print("===================================")

df.to_csv(OUTPUT_CSV, index=False)

print(" Normalized dataset saved to:")
print(OUTPUT_CSV)