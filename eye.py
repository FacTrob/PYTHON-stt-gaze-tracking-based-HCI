import cv2, mediapipe as mp, numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Precision Eye Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Precision Eye Tracker", 1280, 720)

# 3D 얼굴 기준 좌표 (PnP 계산용)
FACE_3D = np.array([
    (0.0, 0.0, 0.0),        
    (0.0, -330.0, -65.0),  
    (-225.0, 170.0, -135), 
    (225.0, 170.0, -135),  
    (-150.0, -150.0, -125),
    (150.0, -150.0, -125)  
])

LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# 홍채 랜드마크 (왼쪽 / 오른쪽)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0].landmark

        FACE_2D = []
        for idx in LANDMARK_IDX:
            FACE_2D.append([int(face[idx].x * w), int(face[idx].y * h)])
        FACE_2D = np.array(FACE_2D, dtype=np.float64)

        cam_matrix = np.array([[w, 0, w/2],
                                [0, w, h/2],
                                [0, 0, 1]])

        dist = np.zeros((4,1))
        _, rot_vec, _ = cv2.solvePnP(FACE_3D, FACE_2D, cam_matrix, dist)
        rmat, _ = cv2.Rodrigues(rot_vec)

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        yaw  = angles[1] * 360
        pitch = angles[0] * 360

        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30,40), 0, 1,(0,255,0),2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (30,80),0,1,(0,255,0),2)

        # -------------------------------
        # ✅ 홍채 중심 좌표 계산
        # -------------------------------
        left_iris = np.mean(
            [[face[i].x * w, face[i].y * h] for i in LEFT_IRIS], axis=0
        )
        right_iris = np.mean(
            [[face[i].x * w, face[i].y * h] for i in RIGHT_IRIS], axis=0
        )

        iris_center = ((left_iris + right_iris) / 2).astype(int)

        # -------------------------------
        # ✅ 시선 벡터 생성 (정면 기준 방향)
        # -------------------------------
        gaze_vector = np.array([0, 0, 1])
        gaze_vector = rmat @ gaze_vector

        # -------------------------------
        # ✅ 화면 상 좌표로 투영
        # -------------------------------
        scale = 500
        gaze_x = int(iris_center[0] + gaze_vector[0] * scale)
        gaze_y = int(iris_center[1] - gaze_vector[1] * scale)

        # -------------------------------
        # ✅ 시선 표시 (초록 점 + 선)
        # -------------------------------
        cv2.circle(frame, tuple(iris_center), 5, (255,0,0), -1)
        cv2.circle(frame, (gaze_x, gaze_y), 10, (0,255,0), -1)
        cv2.line(frame, tuple(iris_center), (gaze_x, gaze_y), (0,255,255), 2)

        cv2.putText(frame, f"Gaze: ({gaze_x}, {gaze_y})",
                    (30,120), 0, 1, (0,255,0), 2)

    cv2.imshow("3D Gaze Core", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
