import cv2
import mediapipe as mp
import csv
import os
import random

# =========================
# MEDIAPIPE SETUP
# =========================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

# =========================
# CONFIG
# =========================
DATASET_PATH = "fake_c"
OUTPUT_CSV = "1.1ForthTryDataset.csv"
AUGMENTATIONS_PER_VIDEO = 3

POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]

FACE_POINTS = [
    10, 67, 21, 46, 276, 8, 197, 1, 4, 48, 278,
    251, 297, 33, 159, 145, 155, 463, 386, 374,
    263, 127, 356, 330, 101, 93, 323, 215, 172,
    435, 397, 378, 149, 152, 17, 0, 39, 269,
    61, 291, 404, 180, 210, 430
]

# =========================
# HEADER
# =========================
header = ["frame", "video"]

# Pose landmarks
for i in POSE_LANDMARKS:
    header += [
        f"pose_{i}_x",
        f"pose_{i}_y",
        f"pose_{i}_z"
    ]

# Hand landmarks
for hand in ["left", "right"]:
    for i in range(21):
        header += [
            f"{hand}_hand_{i}_x",
            f"{hand}_hand_{i}_y",
            f"{hand}_hand_{i}_z"
        ]

# Face landmarks
for i in FACE_POINTS:
    header += [
        f"face_{i}_x",
        f"face_{i}_y",
        f"face_{i}_z"
    ]

header.append("label")


# =========================
# FRAME AUGMENTATION
# =========================
def augment_frame(frame, angle, scale):

    h, w = frame.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )








# ========================= PROCESS =========================

# =========================
# PROCESS
# =========================
with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(header)

    print("========================================")
    print("Starting extraction...")
    print("========================================")

    with mp_pose.Pose() as pose, \
         mp_hands.Hands(max_num_hands=2) as hands, \
         mp_face.FaceMesh(max_num_faces=1) as face:

        folders = sorted(
            [d for d in os.listdir(DATASET_PATH)
             if os.path.isdir(os.path.join(DATASET_PATH, d))]
        )

        print(f"Found {len(folders)} folders.\n")

        for folder_index, letter_folder in enumerate(folders, start=1):

            folder_path = os.path.join(DATASET_PATH, letter_folder)

            videos = sorted(os.listdir(folder_path))

            print("========================================")
            print(f"Folder {folder_index}/{len(folders)}")
            print(f"Label : {letter_folder}")
            print(f"Videos: {len(videos)}")
            print("========================================")

            for video_index, video_name in enumerate(videos, start=1):

                video_path = os.path.join(folder_path, video_name)
                label = os.path.splitext(video_name)[0]

                print(
                    f"\nVideo {video_index}/{len(videos)} : {video_name}"
                )

                # ---------------------------------------
                # ORIGINAL + 3 AUGMENTATIONS
                # ---------------------------------------
                for aug_idx in range(AUGMENTATIONS_PER_VIDEO + 1):

                    cap = cv2.VideoCapture(video_path)

                    frame_id = 0

                    if aug_idx == 0:
                        angle = 0
                        scale = 1.0
                        aug_name = "Original"
                    else:
                        angle = random.uniform(-10, 10)
                        scale = random.uniform(0.9, 1.1)
                        aug_name = f"Augmentation {aug_idx}"

                    video_id = video_name

                    print(
                        f"\n{aug_name}"
                        f" | Rotation={angle:.2f}"
                        f" | Scale={scale:.2f}"
                    )

                    while True:

                        success, frame = cap.read()

                        if not success:
                            break

                        frame = augment_frame(frame, angle, scale)

                        rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        pose_results = pose.process(rgb)
                        hand_results = hands.process(rgb)
                        face_results = face.process(rgb)

                        row = [frame_id, video_id]

                        # =========================
                        # POSE
                        # =========================
                        if pose_results.pose_landmarks:

                            lm = pose_results.pose_landmarks.landmark

                            for i in POSE_LANDMARKS:
                                row.extend([
                                    lm[i].x,
                                    lm[i].y,
                                    lm[i].z
                                ])

                        else:
                            row.extend([0] * (len(POSE_LANDMARKS) * 3))

                        # =========================
                        # HANDS
                        # =========================
                        left_hand = [0] * (21 * 3)
                        right_hand = [0] * (21 * 3)

                        if hand_results.multi_hand_landmarks:

                            for hand_landmarks, handedness in zip(
                                hand_results.multi_hand_landmarks,
                                hand_results.multi_handedness
                            ):

                                coords = []

                                for lm in hand_landmarks.landmark:
                                    coords.extend([
                                        lm.x,
                                        lm.y,
                                        lm.z
                                    ])

                                if handedness.classification[0].label == "Left":
                                    left_hand = coords
                                else:
                                    right_hand = coords

                        row.extend(left_hand)
                        row.extend(right_hand)

                        # =========================
                        # FACE
                        # =========================
                        if face_results.multi_face_landmarks:

                            face_lm = (
                                face_results
                                .multi_face_landmarks[0]
                                .landmark
                            )

                            for i in FACE_POINTS:
                                lm = face_lm[i]
                                row.extend([
                                    lm.x,
                                    lm.y,
                                    lm.z
                                ])

                        else:
                            row.extend([0] * (len(FACE_POINTS) * 3))

                        # =========================
                        # LABEL
                        # =========================
                        row.append(label)

                        writer.writerow(row)

                        frame_id += 1

                        # Print every 50 frames
                        if frame_id % 50 == 0:
                            print(
                                f"   Processed {frame_id} frames...",
                                flush=True
                            )

                    cap.release()

                    print(
                        f"Finished {aug_name} "
                        f"({frame_id} frames)"
                    )

print("\n========================================")
print("Extraction finished successfully!")
print("========================================")
print("DONE: dataset created safely with augmentation.")


