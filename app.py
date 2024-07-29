import cv2 
import mediapipe as mp 

mp_pose=mp.solutions.pose
pose=mp_pose.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing= mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame= cap.read()
    if not ret:
        break

    rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res =pose.process(rgb)

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks=res.pose_landmarks.landmark
        shoulder_left= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        shoulder_right= landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip_left= landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        hip_right= landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_slope= abs(shoulder_left.y - shoulder_right.y)
        hip_slope=abs(hip_left.y - hip_right.y)

        shoulder_threshold= 0.1
        hip_threshold= 0.1

        if shoulder_slope > shoulder_threshold or hip_slope > hip_threshold:
            cv2.putText(frame, "Adjust Posture!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Posture Detection', frame)

    if cv2.waitKey(10) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()