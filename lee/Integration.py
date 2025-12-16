import cv2, time, json, os
import threading
import queue
from collections import deque
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import pyautogui as pg
import pyperclip
import webbrowser


# 0. 기본 설정
CALIB_FILE = "gaze_calibration.json"
SCREEN_W, SCREEN_H = 1920, 1080   
MEDIAN_WIN, EMA_ALPHA = 3, 0.25              
READY_SEC, MIN_SAMPLES, MAX_SEC = 0.5, 15, 3.0  
CONFIRM_TIMEOUT = 1.5   
HOLD_SEC = 0.3        


dictation_mode = False          # 입력 모드 여부
dictation_buffer = []           # 누적 텍스트
DICTATION_END_WORDS = ("끝", "완료", "입력 종료", "종료")


# 1. Mediapipe / Camera (속도 최적화)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,    
    min_tracking_confidence=0.5      
)


cap = cv2.VideoCapture(0)


LEFT_EYE = {"L": 33, "R": 133, "P": 468}
RIGHT_EYE = {"L": 362, "R": 263, "P": 473}


# 2. STT 행동 함수
def action_left_click():
    print("좌클릭")
    pg.click()

def action_double_click():
    print("더블클릭")
    pg.doubleClick()

def action_enter_key():
    print("엔터")
    pg.press("enter")

def action_space_key():
    print("스페이스")
    pg.press("space")

def action_tab_key():
    print("탭")
    pg.press("tab")

def action_alt_f4():
    print("Alt+F4")
    pg.hotkey("alt", "f4")

def open_google():
    print("Google")
    webbrowser.open("https://www.google.com")

def open_notepad():
    print("메모장")
    os.system("notepad")



def convert_symbols(text):
    symbol_map = {
        "괄호 열기": "(",
        "괄호 닫기": ")",
        "대괄호 열기": "[",
        "대괄호 닫기": "]",
        "중괄호 열기": "{",
        "중괄호 닫기": "}",
        "큰 따옴표": "\"",
        "작은 따옴표": "'",
        "쉼표": ",",
        "마침표": ".",
        "콜론": ":",
        "더하기": "+",
        "빼기": "-",
        "곱하기": "*",
        "나누기": "/",
        "등호": "=",
        "세미 콜론": ";",
    }
    for k, v in symbol_map.items():
        text = text.replace(k, v)
    return text


def action_dictation(text):
    text = convert_symbols(text)
    print(f"받아쓰기: {text}")
    pyperclip.copy(text)
    pg.hotkey("ctrl", "v")


command_map = {
    "더블 클릭": action_double_click,
    "클릭": action_left_click,
    "엔터": action_enter_key,
    "enter": action_enter_key,
    "스페이스": action_space_key,
    "탭": action_tab_key,
    "종료": action_alt_f4,
    "구글": open_google,
    "메모장": open_notepad,
}


sorted_commands = sorted(command_map.items(), key=lambda x: len(x[0]), reverse=True)


def rotate_points(pts, angle, origin):
    c, s = np.cos(angle), np.sin(angle)
    return (pts - origin) @ np.array([[c, -s], [s, c]]).T + origin

def eye_ratio(landmarks, shape, idx):
    h, w = shape[:2]
    pt = lambda i: np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)
    L, R, P = pt(idx["L"]), pt(idx["R"]), pt(idx["P"])

    angle = -np.arctan2((R - L)[1], (R - L)[0])
    origin = (L + R) / 2
    Lr, Rr, Pr = rotate_points(np.stack([L, R, P]), angle, origin)

    return (Pr[0] - Lr[0]) / max(1.0, abs(Rr[0] - Lr[0]))

def get_gaze_x(landmarks, shape):
    return (eye_ratio(landmarks, shape, LEFT_EYE) +
            eye_ratio(landmarks, shape, RIGHT_EYE)) / 2



def collect_samples(name, target_x):
    samples = []

    t0 = time.time()
    while time.time() - t0 < READY_SEC:
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        cv2.circle(canvas, (target_x, SCREEN_H // 2), 18, (0, 255, 0), -1)
        cv2.putText(canvas, f"Look: {name}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("CALIB", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    t1 = time.time()
    while time.time() - t1 < MAX_SEC and len(samples) < MIN_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            x = get_gaze_x(res.multi_face_landmarks[0].landmark, frame.shape)
            if -0.5 < x < 1.5:
                samples.append(x)

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        cv2.circle(canvas, (target_x, SCREEN_H // 2), 18, (0, 255, 0), -1)
        cv2.putText(canvas, f"{name}: {len(samples)}/{MIN_SAMPLES}",
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.imshow("CALIB", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    return float(np.median(samples)) if len(samples) >= MIN_SAMPLES // 2 else None

def calibrate():
    cv2.namedWindow("CALIB", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CALIB", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    targets = {
        "CENTER": SCREEN_W // 2,
        "LEFT": int(SCREEN_W * 0.18),
        "RIGHT": int(SCREEN_W * 0.82),
    }

    data = {}
    for name, x in targets.items():
        r = collect_samples(name, x)
        if r is None:
            cv2.destroyWindow("CALIB")
            return None
        data[name] = r

    cv2.destroyWindow("CALIB")

    x_min, x_max = min(data["LEFT"], data["RIGHT"]), max(data["LEFT"], data["RIGHT"])
    return {"x_min": x_min, "x_max": x_max, "center": data["CENTER"]} if (x_max - x_min) >= 0.03 else None

def classify_gaze(x, calib):
    span = calib["x_max"] - calib["x_min"]
    left_th = calib["x_min"] + span / 4
    right_th = calib["x_min"] + 3 * span / 4
    return "LEFT" if x < left_th else "RIGHT" if x > right_th else "CENTER"


def save_calibration(calib):
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=4)

def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    try:
        with open(CALIB_FILE, "r") as f:
            return json.load(f)
    except:
        return None



def audio_thread(out_q, stop_event):
    r = sr.Recognizer()
    r.pause_threshold = 1.5
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.3)  # 0.6 → 0.3초

    while not stop_event.is_set():
        try:
            with mic as source:
                print("듣는 중...")
                audio = r.listen(source, timeout=3, phrase_time_limit=6) 

            try:
                text = r.recognize_google(audio, language="ko-KR")
                print("STT:", text)
                out_q.put(text)
            except:
                pass
        except:
            pass


def parse_to_action(text):
    global dictation_mode, dictation_buffer

    if not text:
        return None, None

    t = text.strip()
    lower = t.lower()

    # ===============================
    # 1) 입력 모드 시작
    # ===============================
    if not dictation_mode and lower in ("입력", "input"):
        dictation_mode = True
        dictation_buffer = []
        print("입력 모드 시작")
        return None, "입력 모드 시작"

    # ===============================
    # 2) 입력 모드 중
    # ===============================
    if dictation_mode:
        # 입력 종료 명령
        if t in DICTATION_END_WORDS:
            dictation_mode = False
            content = " ".join(dictation_buffer).strip()
            dictation_buffer = []

            if content:
                return (lambda: action_dictation(content)), f'받아쓰기 완료: "{content}"'
            else:
                return None, "입력 내용 없음"

        # 계속 누적
        dictation_buffer.append(t)
        print("누적 중:", " ".join(dictation_buffer))
        return None, "입력 중..."

    # ===============================
    # 3) 일반 명령 처리
    # ===============================
    for command, action in sorted_commands:
        if command in t:
            return action, f"명령: {command}"

    return None, "등록되지 않은 명령"




def main():
    calib = load_calibration()
    if calib is None:
        print("캘리브레이션 시작")
        calib = calibrate()
        if not calib:
            print("캘리브레이션 실패")
            return
        save_calibration(calib)

    q = queue.Queue()
    stop_event = threading.Event()
    threading.Thread(target=audio_thread, args=(q, stop_event), daemon=True).start()

    x_buf = deque(maxlen=MEDIAN_WIN)
    ema_x = calib["center"]

    pending_action = None
    pending_desc = ""
    pending_time = 0.0
    hold_target = None
    hold_start = None


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        gaze = "CENTER"
        if res.multi_face_landmarks:
            x_buf.append(get_gaze_x(res.multi_face_landmarks[0].landmark, frame.shape))
            ema_x = (1 - EMA_ALPHA) * ema_x + EMA_ALPHA * np.median(x_buf)
            gaze = classify_gaze(ema_x, calib)

        try:
            text = q.get_nowait()
            action, desc = parse_to_action(text)
            if action:
                pending_action = action
                pending_desc = desc
                pending_time = time.time()
                hold_target = None
                hold_start = None
                print("빠른 확인:", desc)
        except queue.Empty:
            pass

        if pending_action:
            if time.time() - pending_time > CONFIRM_TIMEOUT:
                pending_action = None
                print("시간 초과 취소")

            elif gaze in ("LEFT", "RIGHT"):
                if hold_target != gaze:
                    hold_target = gaze
                    hold_start = time.time()
                elif time.time() - hold_start >= HOLD_SEC:
                    if gaze == "LEFT":
                        pending_action()
                        print("실행 완료")
                    else:
                        print("취소됨")
                    pending_action = None

        cv2.putText(frame, f"Gaze: {gaze}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # 대기 중일 때 상태 표시
        if pending_action:
            elapsed = time.time() - pending_time
            remaining = CONFIRM_TIMEOUT - elapsed
            cv2.putText(frame, f"Waiting: {remaining:.1f}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("STT + Gaze Confirm [FAST MODE]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
