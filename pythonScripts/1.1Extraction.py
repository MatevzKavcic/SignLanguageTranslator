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
DATASET_PATH = "../VideoFolder"
OUTPUT_CSV = "full_dataset_augmented.csv"
AUGMENTATIONS_PER_VIDEO = 3

POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]

# =========================
# HEADER
# =========================
header = ["frame", "video"]

for i in POSE_LANDMARKS:
    header += [f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"]

for hand in ["left", "right"]:
    for i in range(21):
        header += [f"{hand}_hand_{i}_x", f"{hand}_hand_{i}_y", f"{hand}_hand_{i}_z"]

for i in range(468):
    header += [f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"]

header.append("label")


# =========================
# FRAME AUGMENTATION (IMAGE ONLY)
# =========================
def augment_frame(frame, angle, scale):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


# =========================
# PROCESS
# =========================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp_pose.Pose() as pose, \
         mp_hands.Hands(max_num_hands=2) as hands, \
         mp_face.FaceMesh(max_num_faces=1) as face:

        for letter_folder in os.listdir(DATASET_PATH):

            folder_path = os.path.join(DATASET_PATH, letter_folder)
            if not os.path.isdir(folder_path):
                continue

            print(f"Processing: {letter_folder}")

            for video_name in os.listdir(folder_path):

                video_path = os.path.join(folder_path, video_name)
                label = os.path.splitext(video_name)[0]

                # original + augmented versions
                for aug_idx in range(AUGMENTATIONS_PER_VIDEO + 1):

                    cap = cv2.VideoCapture(video_path)
                    frame_id = 0

                    if aug_idx == 0:
                        angle, scale = 0, 1.0
                        video_id = f"{video_name}_original"
                    else:
                        angle = random.uniform(-10, 10)
                        scale = random.uniform(0.9, 1.1)
                        video_id = f"{video_name}_aug_{aug_idx}"

                    print(f"{video_id} | rot={angle:.1f}, scale={scale:.2f}")

                    while True:
                        success, frame = cap.read()
                        if not success:
                            break

                        # =========================
                        # AUGMENT IMAGE ONLY
                        # =========================
                        frame = augment_frame(frame, angle, scale)

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        pose_results = pose.process(rgb)
                        hand_results = hands.process(rgb)
                        face_results = face.process(rgb)

                        row = [frame_id, video_id]

                        # =========================
                        # POSE (KEEP 0s AS-IS)
                        # =========================
                        if pose_results.pose_landmarks:
                            lm = pose_results.pose_landmarks.landmark
                            for i in POSE_LANDMARKS:
                                row.extend([lm[i].x, lm[i].y, lm[i].z])
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
                                    coords.extend([lm.x, lm.y, lm.z])

                                if handedness.classification[0].label == "Left":
                                    left_hand = coords
                                else:
                                    right_hand = coords

                        row.extend(left_hand)
                        row.extend(right_hand)

                        # =========================
                        # FACE (KEEP ALL 468 OR 0s)
                        # =========================
                        if face_results.multi_face_landmarks:
                            face_lm = face_results.multi_face_landmarks[0].landmark
                            for lm in face_lm:
                                row.extend([lm.x, lm.y, lm.z])
                        else:
                            row.extend([0] * (468 * 3))

                        # =========================
                        # LABEL
                        # =========================
                        row.append(label)

                        writer.writerow(row)
                        frame_id += 1

                    cap.release()

print("DONE: dataset created safely with augmentation.")