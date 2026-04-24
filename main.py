import cv2
import mediapipe as mp
import csv

# ---------------- INIT ----------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

VIDEO_PATH = "VideoFolder/a/a.mp4"
OUTPUT_CSV = "gesture_data_letterA.csv"

cap = cv2.VideoCapture(VIDEO_PATH)

# -------- SELECTED LANDMARKS --------

# Upper body only
POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]

# Face (small useful subset)
FACE_LANDMARKS = [
    1, 33, 263, 159, 386,
    61, 291, 13, 14,
    152, 10
]

# -------- CSV HEADER --------
header = ["frame"]

# Pose
for i in POSE_LANDMARKS:
    header += [f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"]

# Hands
for hand in ["left", "right"]:
    for i in range(21):
        header += [f"{hand}_hand_{i}_x", f"{hand}_hand_{i}_y", f"{hand}_hand_{i}_z"]

# Face
for i in range(468):
    header += [f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"]
# -------- PROCESS --------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp_pose.Pose() as pose, \
         mp_hands.Hands(max_num_hands=2) as hands, \
         mp_face.FaceMesh(max_num_faces=1) as face:

        frame_id = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)
            face_results = face.process(rgb)

            row = [frame_id]

            # -------- POSE --------
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                for i in POSE_LANDMARKS:
                    row.extend([lm[i].x, lm[i].y, lm[i].z])

                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
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

                    label = handedness.classification[0].label

                    if label == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            row.extend(left_hand)
            row.extend(right_hand)

            # -------- FACE --------
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0].landmark

                for lm in face_lm:
                    row.extend([lm.x, lm.y, lm.z])

                    # draw selected points only (clean)
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
            else:
                row.extend([0] * (len(FACE_LANDMARKS) * 3))

            # -------- SAVE --------
            writer.writerow(row)

            # -------- DISPLAY --------
            cv2.imshow("Gesture Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

# -------- CLEANUP --------
cap.release()
cv2.destroyAllWindows()

print("✅ DONE! Data saved to", OUTPUT_CSV)