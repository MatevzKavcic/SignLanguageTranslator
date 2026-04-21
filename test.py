import cv2 # type: ignore
import mediapipe as mp # type: ignore
import csv
import os

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

DATASET_PATH = "VideoFolder"
OUTPUT_CSV = "full_dataset.csv"

POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]

# -------- HEADER --------
header = ["frame", "video"]

for i in POSE_LANDMARKS:
    header += [f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"]

for hand in ["left", "right"]:
    for i in range(21):
        header += [f"{hand}_hand_{i}_x", f"{hand}_hand_{i}_y", f"{hand}_hand_{i}_z"]

for i in range(468):
    header += [f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"]

header.append("label")

# -------- PROCESS --------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp_pose.Pose() as pose, \
         mp_hands.Hands(max_num_hands=2) as hands, \
         mp_face.FaceMesh(max_num_faces=1) as face:

        # loop a/, b/, c/...
        for letter_folder in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, letter_folder)

            if not os.path.isdir(folder_path):
                continue

            print(f" Processing folder: {letter_folder}")

            # loop videos inside folder
            for video_name in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_name)

                # label from filename
                label = os.path.splitext(video_name)[0]

                print(f"{label}")

                cap = cv2.VideoCapture(video_path)
                frame_id = 0

                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pose_results = pose.process(rgb)
                    hand_results = hands.process(rgb)
                    face_results = face.process(rgb)

                    row = [frame_id, video_name]

                    # -------- POSE --------
                    if pose_results.pose_landmarks:
                        lm = pose_results.pose_landmarks.landmark
                        for i in POSE_LANDMARKS:
                            row.extend([lm[i].x, lm[i].y, lm[i].z])
                    else:
                        row.extend([0] * (len(POSE_LANDMARKS) * 3))

                    # -------- HANDS --------
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

                            hand_label = handedness.classification[0].label

                            if hand_label == "Left":
                                left_hand = coords
                            else:
                                right_hand = coords

                    row.extend(left_hand)
                    row.extend(right_hand)

                    # -------- FACE --------
                    if face_results.multi_face_landmarks:
                        face_lm = face_results.multi_face_landmarks[0].landmark
                        for lm in face_lm:
                            row.extend([lm.x, lm.y, lm.z])
                    else:
                        row.extend([0] * (468 * 3))

                    # -------- LABEL --------
                    row.append(label)

                    writer.writerow(row)

                    frame_id += 1

                cap.release()

print("✅ DONE! Dataset created.")