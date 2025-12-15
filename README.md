
# PYTHON-stt-gaze-tracking-based-HCI

RnE(정보) 프로젝트 구현 레포지토리이다.  
음성(STT)과 시선(gaze) 입력을 결합해 기본적인 HCI 작업을 수행하는 프로토타입과,
입력 방식에 따른 작업 효율을 비교하기 위한 실험 A/B 코드 및 로그 수집기를 포함한다.

본 레포의 목적은 **모델 성능 향상**이 아니라  
**실험 재현성, 입력 조건 통제, 로그 기반 정량 분석**에 있다.

---

## Repository Structure

- _lab/experiment_ab.py  
  STT·시선·멀티모달 입력 조건(C1~C4)을 전환하며 실험 A/B를 수행하는 메인 실험 코드

- _lab/experiment_controller_yesno.py  
  STT 명령 + 시선 기반 Yes/No 판단 실험용 컨트롤러 (외부 STT·아이 트래커 연동)

- src/gaze/eyetracker_mediapipe.py  
  MediaPipe 기반 시선 → 포인터 이동 프로토타입

- logs/  
  실험 실행 시 자동 생성되는 로그 폴더 (JSONL, CSV / gitignore 대상)

---

## Experiment Overview

### Conditions
- C1: mouse / keyboard
- C2: STT-only
- C3: gaze-only (dwell)
- C4: multimodal (STT + gaze confirm)

### Tasks
- A: 3x3 grid target selection (A1~C3)
- B: file selection + action (open / delete)

### Metrics
- completion time (seconds)
- success rate
- error counts
  - wrong_target
  - wrong_file
  - wrong_action
- optional latency
  - STT listen → recognition done

---

## Setup

(선택) 가상환경
python -m venv rnevenv
rnevenv\\Scripts\\activate

의존성 설치
pip install -r requirements.txt

---

## Running Experiments

### A/B Experiment
python .\\_lab\\experiment_ab.py

Controls
- Space : trial 시작 / 종료
- Tab   : MODE A/B 전환
- C     : 조건(C1~C4) 전환
- Enter : 확정
- S     : STT 토글

### STT + Gaze Yes/No Experiment
python .\\_lab\\experiment_controller_yesno.py

Global Hotkeys
- F8  : trial 시작 / 중단
- F9  : STT 인식 완료 신호
- F10 : YES (좌 시선)
- F11 : NO  (우 시선)
- F12 : 과제 완료

---

## Logging

- 각 trial마다 JSONL, CSV 파일이 자동 저장된다.
- 로그는 trial 단위 분석을 전제로 설계되었다.

주요 이벤트
- trial_start
- stt_ok
- decision
- task_begin
- task_done
- error
- trial_end

trial_end 필드
- success
- total_s
- reason

---

## Notes

- STT 및 아이 트래킹은 **외부 프로그램**으로 실행되며,
  본 레포의 실험 코드는 키 입력을 통해 신호만 수신한다.
- gaze 모델 학습 및 고도화는 후속 연구 범위로 둔다.
- 문서와 실제 동작이 불일치할 경우, **코드 구현을 기준으로 한다**.