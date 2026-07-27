import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import csv
import sys

VideoName = sys.argv[1]

# =========================
# INPUT / OUTPUT
# =========================

VIDEO_PATH = f"{VideoName}"

OUTPUT_CSV = "video_sequence.csv"


# =========================
# LANDMARK CONFIG
# =========================

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

for i in POSE_LANDMARKS:
    header += [f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"]

for hand in ["left", "right"]:
    for i in range(21):
        header += [f"{hand}_hand_{i}_x", f"{hand}_hand_{i}_y", f"{hand}_hand_{i}_z"]

for i in FACE_POINTS:
    header += [f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"]

header.append("label")

# =========================
# VIDEO ID / LABEL
# =========================

video_id = "cigan_org"
label = "cigan"

# =========================
# PROCESS VIDEO
# =========================

print("Processing single video...")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp.solutions.pose.Pose() as pose, \
         mp.solutions.hands.Hands(max_num_hands=2) as hands, \
         mp.solutions.face_mesh.FaceMesh(max_num_faces=1) as face:

        cap = cv2.VideoCapture(VIDEO_PATH)

        frame_id = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)
            face_results = face.process(rgb)

            row = [frame_id, video_id]

            # ================= POSE =================
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                for i in POSE_LANDMARKS:
                    row.extend([lm[i].x, lm[i].y, lm[i].z])
            else:
                row.extend([0] * len(POSE_LANDMARKS) * 3)

            # ================= HANDS =================
            left_hand = [0] * (21 * 3)
            right_hand = [0] * (21 * 3)

            if hand_results.multi_hand_landmarks:
                for hand_lms, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
                ):
                    coords = []
                    for lm in hand_lms.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    if handedness.classification[0].label == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords

            row.extend(left_hand)
            row.extend(right_hand)

            # ================= FACE =================
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0].landmark
                for i in FACE_POINTS:
                    lm = face_lm[i]
                    row.extend([lm.x, lm.y, lm.z])
            else:
                row.extend([0] * len(FACE_POINTS) * 3)

            # ================= LABEL =================
            row.append(label)

            writer.writerow(row)
            frame_id += 1

        cap.release()

print("DONE → CSV created for single video")
