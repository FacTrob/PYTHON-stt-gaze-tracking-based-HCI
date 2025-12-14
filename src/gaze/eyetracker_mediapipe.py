#파이썬 3.11에서 라이브러리 다운로드 후 실행
#눈으로 마우스를 움직이고, 선택 인식

import cv2
import pyautogui
import mediapipe as mp
import time

# Mediapipe 얼굴/눈 모델
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

# 화면 크기
screen_width, screen_height = pyautogui.size()

# 웹캠 열기
cap = cv2.VideoCapture(0)

def get_left_eye_center(landmarks, image_shape):
    """좌안 눈 중심 근사 계산 (두 개 점 평균)"""
    h, w = image_shape[:2]
    left_eye_idx = [33, 133]  # Mediapipe 좌안 눈 모서리 인덱스
    x = int(sum([landmarks[i].x for i in left_eye_idx])/len(left_eye_idx) * w)
    y = int(sum([landmarks[i].y for i in left_eye_idx])/len(left_eye_idx) * h)
    return x, y

print("실행 중... 'q' 키를 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        eye_x, eye_y = get_left_eye_center(landmarks, frame.shape)

        # 화면 좌표로 변환
        screen_x = int(eye_x / frame.shape[1] * screen_width)
        screen_y = int(eye_y / frame.shape[0] * screen_height)

        # 마우스 포인터 이동
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)

        # 좌/우 선택 판단
        choice = "예" if screen_x < screen_width // 2 else "아니오"
        print(choice, end="\r")

        # 눈 위치 시각화
        cv2.circle(frame, (eye_x, eye_y), 5, (0, 255, 0), -1)

    # 영상 표시
    cv2.imshow("Eye Tracker (Left Eye)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
