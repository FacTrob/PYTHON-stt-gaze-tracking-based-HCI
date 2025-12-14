# PYTHON-stt-gaze-tracking-based-HCI

RnE(정보) 프로젝트 구현 레포. 음성(STT) + 시선(gaze) 입력을 결합해 클릭/선택 같은 기본 HCI 작업을 수행하는 프로토타입과, 실험 A/B(타겟 선택, 파일 선택/동작)의 로그 수집 코드를 포함한다.

## What is in this repo
- _lab/experiment_ab.py : 실험 A/B 실행(로그 자동 저장)
- src/gaze/eyetracker_mediapipe.py : MediaPipe 기반 gaze → 마우스 이동 프로토타입
- logs/ : 실험 로그 출력 폴더(자동 생성, gitignore)

## Quickstart
1) 가상환경(선택)
- python -m venv rnevenv
- rnevenv\Scripts\activate

2) 설치
- pip install -r requirements.txt

3) 실험 A/B 실행
- python .\_lab\experiment_ab.py
  - Space : trial 시작/중지
  - Tab : MODE A/B 전환
  - C : 조건(C1~C4) 전환
  - Enter : 확정(멀티모달에서 사용)
  - S : STT 토글

## Experiments
- A: 3x3 타겟 선택(A1~C3)
- B: 파일 선택 + 동작(open/delete)
조건
- C1 mouse/keyboard
- C2 STT-only
- C3 gaze-only(dwell)
- C4 multimodal(gaze target + confirm)

## Notes
이 레포는 “모델 고도화”보다 “실험 재현/기록”을 우선한다. ML 기반 gaze 모델 학습은 후속 단계에서 진행할듯(잘몰루~)
