# Experiment Protocol
STT 기반 명령 실행과 시선 기반 이진 판단(Yes/No)을 결합한 멀티모달 인터페이스의 작업 수행 효율 분석

## 1. Goal
본 실험의 목표는 STT 기반 명령 입력과 시선 기반 이진 판단(Yes/No)을 결합한 멀티모달 입력 방식이
STT 단독 입력 방식에 비해 작업 수행 시간 및 오류율 측면에서 어떤 차이를 보이는지 정량적으로 비교하는 것이다.

비교 조건
- STT 단독 입력
- STT + 시선 기반 Yes/No 판단 입력

## 2. Conditions
| Condition ID | Description |
|---|---|
| C1_STT_ONLY | STT 인식 즉시 실행되는 단일 입력 방식 |
| C2_STT_GAZE_YESNO | STT 명령 후 시선 기반 Yes/No 판단을 거치는 멀티모달 방식 |

## 3. Task
절차
1) 외부 STT 프로그램으로 “메모장 실행” 발화
2) STT 인식 성공 시 실험 컨트롤러에 신호 전달
3) 조건에 따라 실행 여부 확정
4) YES면 메모장을 실행하고 지정 문장 입력
5) 입력 완료 후 종료 신호 전달

입력 문장
정의로운 과학으로 창의적인 도전을
바른 인성을 갖춘 노벨과학인재 육성

## 4. Independent Variable (조작 변인)
| Variable | Levels |
|---|---|
| 입력 방식 | STT 단독 / STT + 시선 Yes/No |

## 5. Dependent Variables (종속 변인)
작업 수행 시간
- 정의: STT 인식 시점(F9)부터 과제 완료 시점(F12)까지
- 단위: seconds

실행 성공 여부
- 메모장 실행 및 문장 입력 완료 여부

오류 유형(로그 기준)
| Error Kind | Meaning |
|---|---|
| unintended_execution | NO 판단 이후 notepad.exe가 실행됨 |
| execution_failed | YES 판단 이후 notepad.exe가 관측되지 않음 |
| manual_stop | 실험자가 F8로 중단 |

## 6. Controlled Variables (통제 변인)
- 동일한 PC/OS
- 동일한 외부 STT 프로그램
- 동일한 입력 문장
- STT 언어: ko-KR
- 시선 매핑: Left=YES, Right=NO
- 실험 시작 전 모든 프로그램 종료
- 동일한 착석 위치/화면 거리

## 7. Procedure (per trial)
Hotkeys (global)
- F8  : Arm/Stop trial (Stop => fail)
- F9  : STT recognized “메모장 실행” (time=0)
- F10 : YES (Left gaze)
- F11 : NO  (Right gaze)
- F12 : Task done (sentence typed)

Steps
1) F8로 trial 시작(armed)
2) 참가자가 외부 STT로 “메모장 실행” 발화
3) STT 인식 확인 시 F9 입력(시간 측정 시작)
4) C2_STT_GAZE_YESNO에서만 F10(YES) 또는 F11(NO)로 결정
5) YES면 메모장 실행 후 문장 입력
6) 입력 완료 시 F12로 종료(시간 측정 종료)
7) 로그 자동 저장(JSONL/CSV)

## 8. Trial Count / Subject
- 대상: 대전동신과학고 학생 1명
- 반복: 조건별 5~10회

## 9. Logging
- trial마다 JSONL 1개 + CSV 1개 저장
- 주요 이벤트: trial_start, stt_ok, decision, task_begin, task_done, error, trial_end
- trial_end 필드: success, total_s, reason