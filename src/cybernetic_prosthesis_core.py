import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import time
import logging
from typing import AsyncGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotorIntention(Enum):
    """운동 의도 유형"""
    NONE = "none"
    POINT = "point" 
    CLICK = "click"
    DRAG = "drag"
    SCROLL = "scroll"
    SELECT = "select"
    OPEN = "open"
    CLOSE = "close"

class ModalityType(Enum):
    """입력 모달리티 유형"""
    VOICE = "voice"
    GAZE = "gaze"
    MULTIMODAL = "multimodal"

@dataclass
class FeedbackMessage:
    timestamp: float
    source: ModalityType
    data: Union[str, np.ndarray]
    confidence: float
    motor_intention: MotorIntention
    digital_action: Optional[str] = None

class CyberneticFeedbackLoop:

    # 피드백 루프 시스템
    def __init__(self):
        self.message_history: List[FeedbackMessage] = []
        self.homeostasis_threshold = 0.8  # 시스템 균형 임계값
        
    def process_message(self, message: FeedbackMessage) -> Dict:
        self.message_history.append(message)
        
        # 정보 품질 평가 (noise 감지)
        signal_quality = self._evaluate_signal_quality(message)
        
        # 피드백 루프 실행
        feedback = {
            'processed_intention': message.motor_intention,
            'action_command': self._generate_digital_action(message),
            'system_state': self._check_homeostasis(),
            'signal_quality': signal_quality,
            'timestamp': time.time()
        }
        
        logger.info(f"Processed message: {message.motor_intention} -> {feedback['action_command']}")
        return feedback
    
    def _evaluate_signal_quality(self, message: FeedbackMessage) -> float:
        """신호 품질 평가 (노이즈 감지)"""
        return min(message.confidence * 1.2, 1.0)  # 간단한 품질 보정
    
    def _generate_digital_action(self, message: FeedbackMessage) -> str:
        """운동 의도를 디지털 행위로 변환"""
        intention_to_action = {
            MotorIntention.POINT: "move_cursor",
            MotorIntention.CLICK: "mouse_click", 
            MotorIntention.DRAG: "drag_operation",
            MotorIntention.SCROLL: "scroll_page",
            MotorIntention.SELECT: "select_element",
            MotorIntention.OPEN: "open_application",
            MotorIntention.CLOSE: "close_application"
        }
        return intention_to_action.get(message.motor_intention, "no_action")
    
    def _check_homeostasis(self) -> str:
        """시스템 항상성(균형) 상태 확인"""
        if len(self.message_history) < 3:
            return "initializing"
        
        recent_messages = self.message_history[-3:]
        avg_confidence = np.mean([msg.confidence for msg in recent_messages])
        
        if avg_confidence >= self.homeostasis_threshold:
            return "stable"
        else:
            return "unstable"

class BodySchemaExtension:
    """
    Merleau-Ponty의 신체 스키마 확장 이론 구현
    "Body schema is a practical diagram of our relationships to the world"
    """
    
    def __init__(self):
        self.digital_limb_mapping = {}
        self.affordances = {}
        self.embodiment_level = 0.0
        
    def integrate_tool(self, tool_name: str, capabilities: List[str]):
        """도구를 신체 스키마에 통합"""
        self.digital_limb_mapping[tool_name] = {
            'capabilities': capabilities,
            'integration_time': time.time(),
            'usage_count': 0,
            'embodiment_score': 0.0
        }
        
        # 어포던스 매핑
        self.affordances[tool_name] = self._identify_affordances(capabilities)
        logger.info(f"Integrated {tool_name} into body schema")
    
    def _identify_affordances(self, capabilities: List[str]) -> Dict:
        """Gibson의 어포던스 이론에 기반한 행동 가능성 식별"""
        affordance_map = {
            'voice_control': ['speak', 'command', 'communicate'],
            'gaze_control': ['look', 'select', 'navigate', 'focus'],
            'multimodal': ['coordinate', 'specify', 'confirm']
        }
        
        identified_affordances = {}
        for capability in capabilities:
            if capability in affordance_map:
                identified_affordances[capability] = affordance_map[capability]
        
        return identified_affordances
    
    def update_embodiment(self, tool_name: str, usage_success: bool):
        """도구 사용 경험을 통한 신체화 정도 업데이트"""
        if tool_name in self.digital_limb_mapping:
            tool_data = self.digital_limb_mapping[tool_name]
            tool_data['usage_count'] += 1
            
            if usage_success:
                tool_data['embodiment_score'] += 0.1
            else:
                tool_data['embodiment_score'] = max(0, tool_data['embodiment_score'] - 0.05)
                
            # 전체 신체화 수준 계산
            total_scores = sum(tool['embodiment_score'] for tool in self.digital_limb_mapping.values())
            self.embodiment_level = min(total_scores / len(self.digital_limb_mapping), 1.0)

class CyberneticProstheticArm:
    
    def __init__(self):
        self.feedback_loop = CyberneticFeedbackLoop()
        self.body_schema = BodySchemaExtension()
        self.active_modalities = {}
        
        # 시스템 상태
        self.is_active = False
        self.current_mode = "standby"
        
        # 성능 메트릭
        self.response_times = []
        self.accuracy_scores = []
        
    async def initialize(self):
        # initialization procedures
        logger.info("Initializing Cybernetic Prosthetic Arm System...")
        
        # 신체 스키마에 도구들 통합
        self.body_schema.integrate_tool("voice_interface", ["voice_control", "speech_recognition"])
        self.body_schema.integrate_tool("gaze_tracker", ["gaze_control", "eye_tracking"])
        self.body_schema.integrate_tool("multimodal_fusion", ["multimodal", "sensor_fusion"])
        
        self.is_active = True
        self.current_mode = "ready"
        logger.info("System initialized and ready")
    
    async def process_intention(self, modality: ModalityType, raw_data: Union[str, np.ndarray], 
                              confidence: float) -> Dict:
        # 실행? 이게뭐지
        start_time = time.time()
        
        # 운동 의도 해석
        motor_intention = self._interpret_motor_intention(modality, raw_data)
        
        message = FeedbackMessage(
            timestamp=start_time,
            source=modality,
            data=raw_data,
            confidence=confidence,
            motor_intention=motor_intention
        )
        
        # 실행
        feedback = self.feedback_loop.process_message(message)
        execution_result = await self._execute_digital_action(feedback['action_command'])
        
        # 성능 메트릭 업데이트
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        self.accuracy_scores.append(execution_result['success_score'])
        
        # 신체 스키마 업데이트
        tool_name = f"{modality.value}_interface"
        self.body_schema.update_embodiment(tool_name, execution_result['success'])
        
        return {
            'motor_intention': motor_intention,
            'digital_action': feedback['action_command'],
            'execution_result': execution_result,
            'response_time': response_time,
            'system_state': feedback['system_state'],
            'embodiment_level': self.body_schema.embodiment_level
        }
    
    def _interpret_motor_intention(self, modality: ModalityType, data: Union[str, np.ndarray]) -> MotorIntention:
        # 입력 데이터로부터 운동 의도 해석
        if modality == ModalityType.VOICE:
            return self._parse_voice_intention(data)
        elif modality == ModalityType.GAZE:
            return self._parse_gaze_intention(data)
        else:
            return MotorIntention.NONE
    
    def _parse_voice_intention(self, voice_data: str) -> MotorIntention:
        # 음성 데이터로부터 의도 파싱 
        voice_lower = voice_data.lower()
        
        intention_keywords = {
            MotorIntention.CLICK: ['click', 'select', 'choose'],
            MotorIntention.OPEN: ['open', 'launch', 'start'],
            MotorIntention.CLOSE: ['close', 'exit', 'quit'],
            MotorIntention.SCROLL: ['scroll', 'page', 'down', 'up'],
            MotorIntention.POINT: ['point', 'move', 'go to']
        }
        
        for intention, keywords in intention_keywords.items():
            if any(keyword in voice_lower for keyword in keywords):
                return intention
                
        return MotorIntention.NONE
    
    def _parse_gaze_intention(self, gaze_data: np.ndarray) -> MotorIntention:
        # 간단한 시선 의도 해석 (실제로는 더 복잡한 ML 모델 필요할걸??? <- 미래의 나 : 어디서 어떻게? <- 지금은 모르겠음)
        if len(gaze_data) > 0:
            return MotorIntention.POINT
        return MotorIntention.NONE
    
    async def _execute_digital_action(self, action_command: str) -> Dict:
        # 시뮬레이션된 실행 (실제로는 PyAutoGUI 등 사용)
        execution_time = np.random.uniform(0.05, 0.2)  # 50-200ms
        await asyncio.sleep(execution_time)
        
        success = np.random.uniform(0, 1) > 0.1  # 90% 성공률 시뮬레이션
        
        return {
            'action': action_command,
            'success': success,
            'success_score': 0.9 if success else 0.1,
            'execution_time': execution_time
        }
    
    def get_performance_metrics(self) -> Dict:
        if not self.response_times:
            return {"status": "no_data"}
            
        return {
            'avg_response_time': np.mean(self.response_times),
            'avg_accuracy': np.mean(self.accuracy_scores),
            'total_interactions': len(self.response_times),
            'embodiment_level': self.body_schema.embodiment_level,
            'system_stability': self.feedback_loop._check_homeostasis()
        }

# 사용 예제
async def main():
    prosthetic_arm = CyberneticProstheticArm()
    await prosthetic_arm.initialize()

    result1 = await prosthetic_arm.process_intention(
        ModalityType.VOICE, 
        "open file browser", 
        confidence=0.92
    )
    print(f"Voice Command Result: {result1}")
    
    gaze_coords = np.array([[320, 240], [325, 245], [330, 250]])
    result2 = await prosthetic_arm.process_intention(
        ModalityType.GAZE,
        gaze_coords,
        confidence=0.78
    )
    print(f"Gaze Tracking Result: {result2}")
    metrics = prosthetic_arm.get_performance_metrics()
    print(f"Performance Metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())