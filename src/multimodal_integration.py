import asyncio
import numpy as np
import time
import logging
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import random
from typing import AsyncGenerator

from cybernetic_prosthesis_core import CyberneticProstheticArm
from hybrid_stt_system import STTEngine
from eye_tracking_system import EyeState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    FILE_OPEN = "file_open"
    TEXT_EDIT = "text_edit"
    MENU_NAVIGATE = "menu_navigate"
    WINDOW_MANAGE = "window_manage"
    SEARCH_PERFORM = "search_perform"
    SCROLL_DOCUMENT = "scroll_document"

class InterfaceMode(Enum):
    TRADITIONAL = "traditional"
    CYBERNETIC_ARM = "cybernetic_arm"
    VOICE_ONLY = "voice_only"
    GAZE_ONLY = "gaze_only"

class UserExperienceMetric(Enum):
    EMBODIMENT_FEELING = "embodiment_feeling"
    NATURALNESS = "naturalness"
    EFFICIENCY = "efficiency"
    FRUSTRATION_LEVEL = "frustration_level"
    LEARNING_CURVE = "learning_curve"
    FATIGUE_LEVEL = "fatigue_level"

@dataclass
class TaskResult:
    task_id: str
    task_type: TaskType
    interface_mode: InterfaceMode
    completion_time: float
    accuracy_score: float
    error_count: int
    sync_delay: float
    user_satisfaction: float
    embodiment_score: Optional[float] = None
    
@dataclass
class ExperimentSession:
    session_id: str
    participant_id: str
    start_time: float
    end_time: float
    task_results: List[TaskResult]
    system_metrics: Dict[str, Any]
    user_feedback: Dict[UserExperienceMetric, float]

class TaskGenerator:
    def __init__(self):
        self.task_templates = {
            TaskType.FILE_OPEN: [
                "파일 메뉴에서 'document.txt' 파일을 열어주세요",
                "최근 문서에서 'report.pdf'를 선택해주세요",
                "새 파일을 생성해주세요"
            ],
            TaskType.TEXT_EDIT: [
                "첫 번째 문단의 '개발'을 '구현'으로 변경해주세요",
                "문서 끝에 '결론' 섹션을 추가해주세요",
                "두 번째 줄을 삭제해주세요"
            ],
            TaskType.MENU_NAVIGATE: [
                "편집 메뉴에서 찾기 기능을 실행해주세요",
                "도구 메뉴의 옵션을 열어주세요",
                "도움말 메뉴에서 정보를 확인해주세요"
            ],
            TaskType.WINDOW_MANAGE: [
                "창을 최소화해주세요",
                "새 창을 열어주세요",
                "현재 창을 닫아주세요"
            ],
            TaskType.SEARCH_PERFORM: [
                "검색창에서 '인공지능'을 검색해주세요",
                "페이지에서 '사이버네틱스' 단어를 찾아주세요",
                "다음 검색 결과로 이동해주세요"
            ],
            TaskType.SCROLL_DOCUMENT: [
                "문서를 아래로 스크롤해주세요",
                "페이지 맨 위로 이동해주세요",
                "다음 섹션으로 이동해주세요"
            ]
        }
        
    def generate_random_task(self, task_type: TaskType) -> str:
        templates = self.task_templates.get(task_type, ["기본 태스크를 수행해주세요"])
        return random.choice(templates)
    
    def generate_task_sequence(self, num_tasks: int = 20) -> List[Tuple[TaskType, str]]:
        tasks = []
        task_types = list(TaskType)
        for i in range(num_tasks):
            task_type = random.choice(task_types)
            task_description = self.generate_random_task(task_type)
            tasks.append((task_type, task_description))
        return tasks

class PerformanceTracker:
    def __init__(self):
        self.current_task = None
        self.task_start_time = None
        self.error_events = []
        self.sync_measurements = deque(maxlen=1000)
        
    def start_task(self, task_id: str, task_type: TaskType, interface_mode: InterfaceMode):
        self.current_task = {'id': task_id, 'type': task_type, 'mode': interface_mode, 'errors': 0}
        self.task_start_time = time.time()
        self.error_events.clear()
        
    def record_error(self, error_type: str, timestamp: float = None):
        if self.current_task:
            self.current_task['errors'] += 1
            self.error_events.append({'type': error_type, 'timestamp': timestamp or time.time()})
    
    def record_sync_delay(self, delay: float):
        self.sync_measurements.append(delay)
    
    def finish_task(self, accuracy_score: float, user_satisfaction: float,
                   embodiment_score: Optional[float] = None) -> TaskResult:
        if not self.current_task or not self.task_start_time:
            raise ValueError("No active task to finish")  # FIXME: 이거 잡히는지 확인 필요
        
        completion_time = time.time() - self.task_start_time
        avg_sync_delay = np.mean(self.sync_measurements) if self.sync_measurements else 0.0
        
        result = TaskResult(
            task_id=self.current_task['id'],
            task_type=self.current_task['type'],
            interface_mode=self.current_task['mode'],
            completion_time=completion_time,
            accuracy_score=accuracy_score,
            error_count=self.current_task['errors'],
            sync_delay=avg_sync_delay,
            user_satisfaction=user_satisfaction,
            embodiment_score=embodiment_score
        )
        
        self.current_task = None
        self.task_start_time = None
        return result

class StatisticalAnalyzer:
    @staticmethod
    def calculate_improvement_percentage(baseline: List[float], improved: List[float]) -> float:
        baseline_mean = np.mean(baseline)
        improved_mean = np.mean(improved)
        if baseline_mean == 0: return 0.0
        return (baseline_mean - improved_mean) / baseline_mean * 100
    
    @staticmethod
    def perform_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
        p_value = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + np.sqrt(df)))  # TODO: 실제 p-value 계산 확인
        return t_stat, p_value
    
    @staticmethod
    def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
        mean1, mean2 = np.mean(group1), np.mean(group2)
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_std

class HypothesisValidator:
    def __init__(self):
        self.analyzer = StatisticalAnalyzer()
        
    def validate_h1_time_reduction(self, traditional_times: List[float], cybernetic_times: List[float]) -> Dict[str, Any]:
        improvement = self.analyzer.calculate_improvement_percentage(traditional_times, cybernetic_times)
        t_stat, p_value = self.analyzer.perform_t_test(traditional_times, cybernetic_times)
        effect_size = self.analyzer.calculate_effect_size(traditional_times, cybernetic_times)
        is_significant = p_value < 0.05
        meets_threshold = improvement >= 20.0
        return {'hypothesis': 'H1: 작업 완료 시간 20% 이상 단축', 'improvement_percentage': improvement,
                'meets_threshold': meets_threshold, 'is_statistically_significant': is_significant,
                't_statistic': t_stat, 'p_value': p_value, 'effect_size': effect_size,
                'result': 'SUPPORTED' if (meets_threshold and is_significant) else 'NOT_SUPPORTED',
                'traditional_mean': np.mean(traditional_times), 'cybernetic_mean': np.mean(cybernetic_times)}
    
    def validate_h2_accuracy_improvement(self, traditional_accuracy: List[float], cybernetic_accuracy: List[float]) -> Dict[str, Any]:
        improvement = self.analyzer.calculate_improvement_percentage(
            [100 - acc for acc in traditional_accuracy], [100 - acc for acc in cybernetic_accuracy]
        )
        t_stat, p_value = self.analyzer.perform_t_test(cybernetic_accuracy, traditional_accuracy)
        effect_size = self.analyzer.calculate_effect_size(cybernetic_accuracy, traditional_accuracy)
        is_significant = p_value < 0.05
        meets_threshold = improvement >= 15.0
        return {'hypothesis': 'H2: 작업 정확도 15% 이상 향상', 'improvement_percentage': improvement,
                'meets_threshold': meets_threshold, 'is_statistically_significant': is_significant,
                't_statistic': t_stat, 'p_value': p_value, 'effect_size': effect_size,
                'result': 'SUPPORTED' if (meets_threshold and is_significant) else 'NOT_SUPPORTED',
                'traditional_mean': np.mean(traditional_accuracy), 'cybernetic_mean': np.mean(cybernetic_accuracy)}
    
    def validate_h3_embodiment_experience(self, embodiment_scores: List[float]) -> Dict[str, Any]:
        neutral_point = 5.0
        neutral_scores = [neutral_point] * len(embodiment_scores)
        t_stat, p_value = self.analyzer.perform_t_test(embodiment_scores, neutral_scores)
        effect_size = self.analyzer.calculate_effect_size(embodiment_scores, neutral_scores)
        mean_score = np.mean(embodiment_scores)
        is_significant = p_value < 0.05
        is_above_neutral = mean_score > neutral_point
        return {'hypothesis': 'H3: 신체 확장감 통계적 유의미 경험', 'mean_embodiment_score': mean_score,
                'is_above_neutral': is_above_neutral, 'is_statistically_significant': is_significant,
                't_statistic': t_stat, 'p_value': p_value, 'effect_size': effect_size,
                'result': 'SUPPORTED' if (is_above_neutral and is_significant) else 'NOT_SUPPORTED',
                'score_range': f"{np.min(embodiment_scores):.2f} - {np.max(embodiment_scores):.2f}"}
    
    def validate_h4_sync_delay(self, sync_delays: List[float]) -> Dict[str, Any]:
        threshold = 0.2
        mean_delay = np.mean(sync_delays)
        max_delay = np.max(sync_delays)
        below_threshold = [delay for delay in sync_delays if delay <= threshold]
        percentage_below = len(below_threshold) / len(sync_delays) * 100
        meets_threshold = mean_delay <= threshold and percentage_below >= 95.0
        return {'hypothesis': 'H4: 동기화 지연시간 0.2초 이하 달성', 'mean_delay': mean_delay,
                'max_delay': max_delay, 'percentage_below_threshold': percentage_below,
                'meets_threshold': meets_threshold, 'result': 'SUPPORTED' if meets_threshold else 'NOT_SUPPORTED',
                'delay_distribution': {'min': np.min(sync_delays), 'max': np.max(sync_delays),
                                       'std': np.std(sync_delays), 'median': np.median(sync_delays)}}
class ExperimentController:
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.session: Optional[ExperimentSession] = None
        self.task_gen = TaskGenerator()
        self.tracker = PerformanceTracker()
        self.results: List[TaskResult] = []
        self.user_feedback: Dict[UserExperienceMetric, float] = {}
        # TODO: 세션 시작 전에 초기화 로직 더 고민해보기
        
    def start_session(self):
        start_time = time.time()
        session_id = f"{self.participant_id}_{int(start_time)}"
        self.session = ExperimentSession(
            session_id=session_id,
            participant_id=self.participant_id,
            start_time=start_time,
            end_time=0.0,
            task_results=[],
            system_metrics={},
            user_feedback={}
        )
        # NOTE: 여기서 로그 남기는거 나중에 시각화용으로 쓸 수 있음
        logger.info(f"[SESSION START] {session_id}")
        
    def run_task_sequence(self, num_tasks: int = 20):
        tasks = self.task_gen.generate_task_sequence(num_tasks)
        for idx, (task_type, task_description) in enumerate(tasks):
            task_id = f"task_{idx+1}"
            # TODO: 실제 인터페이스 모드 동적으로 선택 가능하게 바꾸기
            interface_mode = InterfaceMode.CYBERNETIC_ARM
            self.tracker.start_task(task_id, task_type, interface_mode)
            
            # NOTE: 여기에 실제 작업 수행 코드 연결 필요 (STT/시선 추적/팔 제어 등)
            logger.info(f"[TASK START] {task_id} - {task_description}")
            time.sleep(random.uniform(0.5, 2.0))  # TODO: 임시 딜레이, 실제 수행시간 측정 필요
            
            # FIXME: 오류 기록 시뮬레이션, 실제 센서/시스템 이벤트로 대체
            if random.random() < 0.1:
                self.tracker.record_error("minor_error")
                
            # TODO: sync delay 측정 로직 실제로 연결
            simulated_delay = random.uniform(0.05, 0.25)
            self.tracker.record_sync_delay(simulated_delay)
            
            accuracy_score = random.uniform(80.0, 100.0)  # TODO: 실제 정확도 계산으로 바꾸기
            user_satisfaction = random.uniform(3.0, 5.0)  # TODO: 유저 피드백 수집 방법 고민
            embodiment_score = random.uniform(4.0, 6.0)   # TODO: embodiment 피드백 설문과 연결
            
            result = self.tracker.finish_task(accuracy_score, user_satisfaction, embodiment_score)
            self.results.append(result)
            # NOTE: 나중에 CSV/DB 저장용으로 result 기록
            logger.info(f"[TASK END] {task_id} - {asdict(result)}")
            
    def end_session(self):
        if not self.session:
            raise ValueError("세션이 시작되지 않음")  # FIXME: 이거 안전하게 처리 필요
        self.session.end_time = time.time()
        self.session.task_results = self.results
        self.session.user_feedback = self.user_feedback
        # NOTE: 여기서 DB/파일 저장 가능
        logger.info(f"[SESSION END] {self.session.session_id} - 총 {len(self.results)}개 태스크 완료")
        
    def collect_user_feedback(self):
        # TODO: 실제 설문/피드백 수집 연결 필요
        self.user_feedback = {
            UserExperienceMetric.EMBODIMENT_FEELING: random.uniform(4.0, 6.0),
            UserExperienceMetric.NATURALNESS: random.uniform(3.0, 5.0),
            UserExperienceMetric.EFFICIENCY: random.uniform(3.0, 5.0),
            UserExperienceMetric.FRUSTRATION_LEVEL: random.uniform(1.0, 3.0),
            UserExperienceMetric.LEARNING_CURVE: random.uniform(2.0, 4.0),
            UserExperienceMetric.FATIGUE_LEVEL: random.uniform(1.0, 3.0)
        }
        logger.info(f"[USER FEEDBACK COLLECTED] {self.user_feedback}")
        

# ----------------------
# TODO: 메인 실행부
# ----------------------
if __name__ == "__main__":
    participant_id = "P001"  # TODO: 나중에 입력 인터페이스 연결
    controller = ExperimentController(participant_id)
    
    controller.start_session()
    
    # TODO: 실험 태스크 개수 조절 가능하게
    controller.run_task_sequence(num_tasks=20)
    
    controller.collect_user_feedback()
    
    controller.end_session()
    
    # FIXME: 나중에 통계 분석 자동 실행 부분 연결
    validator = HypothesisValidator()
    
    traditional_times = [random.uniform(10.0, 20.0) for _ in range(20)]
    cybernetic_times = [r.completion_time for r in controller.results]
    
    h1_result = validator.validate_h1_time_reduction(traditional_times, cybernetic_times)
    logger.info(f"H1 결과: {h1_result}")
    
    traditional_accuracy = [random.uniform(70.0, 90.0) for _ in range(20)]
    cybernetic_accuracy = [r.accuracy_score for r in controller.results]
    
    h2_result = validator.validate_h2_accuracy_improvement(traditional_accuracy, cybernetic_accuracy)
    logger.info(f"H2 결과: {h2_result}")
    
    embodiment_scores = [r.embodiment_score for r in controller.results if r.embodiment_score is not None]
    h3_result = validator.validate_h3_embodiment_experience(embodiment_scores)
    logger.info(f"H3 결과: {h3_result}")
    
    sync_delays = [r.sync_delay for r in controller.results]
    h4_result = validator.validate_h4_sync_delay(sync_delays)
    logger.info(f"H4 결과: {h4_result}")

    # NOTE: 여기서 나중에 결과 저장/시각화 모듈 연결 예정
