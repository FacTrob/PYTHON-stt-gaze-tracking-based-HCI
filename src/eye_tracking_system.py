import asyncio
import numpy as np
import cv2
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from typing import AsyncGenerator
from enum import Enum
import threading
from collections import deque
import json

# TODO: 나중에 진짜 MediaPipe, TensorFlow로 교체
class MockMediaPipe:
    class FaceMesh:
        def __init__(self):
            self.confidence = 0.8
            
        def process(self, image):
            landmarks = []
            h, w = image.shape[:2]
            for i in range(468):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9) 
                z = np.random.uniform(-0.1, 0.1)
                landmarks.append([x, y, z])
            
            return type('Result', (), {
                'multi_face_landmarks': [type('Landmarks', (), {
                    'landmark': [type('Point', (), {'x': p[0], 'y': p[1], 'z': p[2]}) 
                               for p in landmarks]
                })()] if np.random.random() > 0.1 else None
            })()

class MockTensorFlow:
    @staticmethod
    def create_cnn_model(input_shape):
        class MockCNN:
            def predict(self, x):
                batch_size = x.shape[0]
                return np.zeros((batch_size, 128))  # 128차원 feature
        return MockCNN()
    
    @staticmethod
    def create_rnn_model(input_shape):
        class MockRNN:
            def predict(self, x):
                batch_size = x.shape[0]
                return np.zeros((batch_size, 3))  # gaze_x, gaze_y, confidence
        return MockRNN()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    BLINKING = "blinking"
    UNKNOWN = "unknown"

class CalibrationMode(Enum):
    FIVE_POINT = "five_point"
    NINE_POINT = "nine_point"
    THIRTEEN_POINT = "thirteen_point"
    CUSTOM = "custom"

@dataclass
class EyeRegion:
    landmarks: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[float, float]
    pupil_center: Optional[Tuple[float, float]] = None
    iris_radius: Optional[float] = None

@dataclass
class GazeData:
    screen_x: float
    screen_y: float
    confidence: float
    timestamp: float
    left_eye: EyeRegion
    right_eye: EyeRegion
    head_pose: Tuple[float, float, float]
    eye_state: EyeState

@dataclass
class CalibrationPoint:
    screen_x: float
    screen_y: float
    gaze_samples: List[Tuple[float, float]]
    completed: bool = False

class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = MockMediaPipe.FaceMesh()
        self.face_landmarks = None
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
    def detect_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            h, w = image.shape[:2]
            for landmark in face_landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z
                landmarks.append((x, y, z))
            return landmarks
        return None
    
    def extract_eye_regions(self, landmarks: List[Tuple[float, float, float]], 
                           image_shape: Tuple[int, int]) -> Tuple[EyeRegion, EyeRegion]:
        h, w = image_shape[:2]
        left_eye_points = [landmarks[i] for i in self.LEFT_EYE_INDICES]
        left_eye_array = np.array(left_eye_points)
        left_bbox = self._calculate_bounding_box(left_eye_points)
        left_center = self._calculate_eye_center(left_eye_points)
        left_pupil = self._estimate_pupil_center(left_eye_points, landmarks, self.LEFT_IRIS_INDICES)
        left_eye = EyeRegion(left_eye_array, left_bbox, left_center, left_pupil)
        right_eye_points = [landmarks[i] for i in self.RIGHT_EYE_INDICES]
        right_eye_array = np.array(right_eye_points)
        right_bbox = self._calculate_bounding_box(right_eye_points)
        right_center = self._calculate_eye_center(right_eye_points)
        right_pupil = self._estimate_pupil_center(right_eye_points, landmarks, self.RIGHT_IRIS_INDICES)
        right_eye = EyeRegion(right_eye_array, right_bbox, right_center, right_pupil)
        return left_eye, right_eye
    
    def _calculate_bounding_box(self, eye_points: List[Tuple[float, float, float]]) -> Tuple[int, int, int, int]:
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _calculate_eye_center(self, eye_points: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        return (np.mean(x_coords), np.mean(y_coords))
    
    def _estimate_pupil_center(self, eye_points: List[Tuple[float, float, float]],
                              all_landmarks: List[Tuple[float, float, float]], iris_indices: List[int]) -> Tuple[float, float]:
        if len(iris_indices) > 0 and all(i < len(all_landmarks) for i in iris_indices):
            iris_points = [all_landmarks[i] for i in iris_indices]
            x_coords = [p[0] for p in iris_points]
            y_coords = [p[1] for p in iris_points]
            return (np.mean(x_coords), np.mean(y_coords))
        return self._calculate_eye_center(eye_points)

class CNNRNNGazeModel:
    def __init__(self, eye_image_size: Tuple[int, int] = (64, 64)):
        self.eye_image_size = eye_image_size
        self.cnn_model = None
        self.rnn_model = None
        self.is_trained = False
        self.sequence_length = 10
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
    def initialize_models(self):
        logger.info("Initializing CNN-RNN gaze estimation model...")
        cnn_input_shape = (*self.eye_image_size, 3)
        self.cnn_model = MockTensorFlow.create_cnn_model(cnn_input_shape)
        rnn_input_shape = (self.sequence_length, 128)
        self.rnn_model = MockTensorFlow.create_rnn_model(rnn_input_shape)
        logger.info("CNN-RNN model initialized")
    
    def extract_eye_features(self, left_eye_image: np.ndarray, right_eye_image: np.ndarray) -> np.ndarray:
        left_processed = self._preprocess_eye_image(left_eye_image)
        right_processed = self._preprocess_eye_image(right_eye_image)
        left_features = self.cnn_model.predict(np.expand_dims(left_processed, 0))[0]
        right_features = self.cnn_model.predict(np.expand_dims(right_processed, 0))[0]
        combined_features = np.concatenate([left_features, right_features])
        return combined_features
    
    def predict_gaze(self, eye_features: np.ndarray) -> Tuple[float, float, float]:
        self.feature_buffer.append(eye_features)
        if len(self.feature_buffer) < self.sequence_length:
            return (0.5, 0.5, 0.5)
        sequence = np.array(list(self.feature_buffer))
        sequence_input = np.expand_dims(sequence, 0)
        prediction = self.rnn_model.predict(sequence_input)[0]
        gaze_x = max(0, min(1, prediction[0]))
        gaze_y = max(0, min(1, prediction[1]))
        confidence = 0.8
        return (gaze_x, gaze_y, confidence)
    
    def _preprocess_eye_image(self, eye_image: np.ndarray) -> np.ndarray:
        if eye_image is None or eye_image.size == 0:
            return np.zeros((*self.eye_image_size, 3))
        resized = cv2.resize(eye_image, self.eye_image_size)
        normalized = resized.astype(np.float32) / 255.0
        if len(normalized.shape) == 3:
            lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.equalizeHist((lab[:,:,0] * 255).astype(np.uint8)) / 255.0
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return normalized
    
    def train_model(self, training_data: List[Dict]):
        logger.info("Training CNN-RNN gaze model...")
        time.sleep(2)
        self.is_trained = True
        logger.info("Model training completed")

class GaussianMixtureEyeClassifier:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.gmm_model = None
        self.is_fitted = False
        self.ear_threshold = 0.25
        self.blink_consecutive_frames = 3
        self.blink_counter = 0
        
    def extract_eye_features(self, eye_region: EyeRegion) -> np.ndarray:
        landmarks = eye_region.landmarks
        if landmarks is None or len(landmarks) < 6:
            return np.array([0.0, 0.0, 0.0])
        ear = self._calculate_ear(landmarks)
        eye_opening = self._calculate_eye_opening(landmarks)
        pupil_eccentricity = self._calculate_pupil_eccentricity(eye_region)
        return np.array([ear, eye_opening, pupil_eccentricity])
    
    def classify_eye_state(self, left_eye: EyeRegion, right_eye: EyeRegion) -> EyeState:
        left_features = self.extract_eye_features(left_eye)
        right_features = self.extract_eye_features(right_eye)
        avg_ear = (left_features[0] + right_features[0]) / 2
        if avg_ear < self.ear_threshold:
            self.blink_counter += 1
            if self.blink_counter >= self.blink_consecutive_frames:
                return EyeState.CLOSED
            else:
                return EyeState.BLINKING
        else:
            if self.blink_counter > 0:
                self.blink_counter = 0
                return EyeState.BLINKING
            return EyeState.OPEN
    
    def _calculate_ear(self, landmarks: np.ndarray) -> float:
        if len(landmarks) < 6:
            return 0.0
        A = np.linalg.norm(landmarks[1] - landmarks[5])
        B = np.linalg.norm(landmarks[2] - landmarks[4])
        C = np.linalg.norm(landmarks[0] - landmarks[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)
    
    def _calculate_eye_opening(self, landmarks: np.ndarray) -> float:
        if len(landmarks) < 4:
            return 0.0
        vertical_dist = np.mean([
            np.linalg.norm(landmarks[i] - landmarks[i+len(landmarks)//2]) 
            for i in range(len(landmarks)//2)
        ])
        return vertical_dist
    
    def _calculate_pupil_eccentricity(self, eye_region: EyeRegion) -> float:
        if eye_region.pupil_center is None:
            return 0.0
        pupil_center = eye_region.pupil_center
        eye_center = eye_region.center
        distance = np.linalg.norm(np.array(pupil_center) - np.array(eye_center))
        eye_width = eye_region.bounding_box[2]
        if eye_width > 0:
            eccentricity = distance / eye_width
        else:
            eccentricity = 0.0
        return eccentricity

class KalmanGazeFilter:
    def __init__(self):
        self.initialized = False
        self.state = np.zeros(4)
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.P = np.eye(4) * 1.0
        
    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[0], self.state[1]
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        z = np.array(measurement)
        if not self.initialized:
            self.state[0] = z[0]
            self.state[1] = z[1]
            self.state[2] = 0.0
            self.state[3] = 0.0
            self.initialized = True
            return measurement
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[0], self.state[1]
    
    def get_smoothed_position(self, raw_position: Tuple[float, float], dt: float = 1.0) -> Tuple[float, float]:
        self.predict(dt)
        return self.update(raw_position)

class CalibrationManager:
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = []
        self.calibration_matrix = None
        self.is_calibrated = False
        
    def setup_calibration_points(self, mode: CalibrationMode) -> List[CalibrationPoint]:
        points = []
        if mode == CalibrationMode.FIVE_POINT:
            positions = [(0.1,0.1),(0.9,0.1),(0.5,0.5),(0.1,0.9),(0.9,0.9)]
        elif mode == CalibrationMode.NINE_POINT:
            positions = [(0.1,0.1),(0.5,0.1),(0.9,0.1),(0.1,0.5),(0.5,0.5),(0.9,0.5),(0.1,0.9),(0.5,0.9),(0.9,0.9)]
        else:
            positions = [(0.1,0.1),(0.5,0.1),(0.9,0.1),(0.1,0.3),(0.5,0.3),(0.9,0.3),(0.1,0.5),(0.5,0.5),(0.9,0.5),(0.1,0.7),(0.5,0.7),(0.9,0.7),(0.5,0.9)]
        for rel_x, rel_y in positions:
            screen_x = rel_x * self.screen_width
            screen_y = rel_y * self.screen_height
            points.append(CalibrationPoint(screen_x, screen_y, []))
        self.calibration_points = points
        return points
    
    def add_gaze_sample(self, point_index: int, gaze_x: float, gaze_y: float):
        if 0 <= point_index < len(self.calibration_points):
            point = self.calibration_points[point_index]
            point.gaze_samples.append((gaze_x, gaze_y))
            if len(point.gaze_samples) >= 30:
                point.completed = True
    
    def compute_calibration_matrix(self) -> bool:
        completed_points = [p for p in self.calibration_points if p.completed]
        if len(completed_points) < 4:
            logger.error("Insufficient calibration points")
            return False
        screen_coords = []
        gaze_coords = []
        for point in completed_points:
            screen_coords.append([point.screen_x, point.screen_y])
            avg_gaze_x = np.mean([s[0] for s in point.gaze_samples])
            avg_gaze_y = np.mean([s[1] for s in point.gaze_samples])
            gaze_coords.append([avg_gaze_x, avg_gaze_y])
        screen_coords = np.array(screen_coords)
        gaze_coords = np.array(gaze_coords)
        try:
            ones = np.ones((len(gaze_coords), 1))
            gaze_homogeneous = np.hstack([gaze_coords, ones])
            self.calibration_matrix = np.linalg.lstsq(gaze_homogeneous, screen_coords, rcond=None)[0]
            self.is_calibrated = True
            logger.info("Calibration completed successfully")
            return True
        except np.linalg.LinAlgError:
            logger.error("Failed to compute calibration matrix")
            return False
    
    def transform_gaze_to_screen(self, gaze_x: float, gaze_y: float) -> Tuple[float, float]:
        if not self.is_calibrated:
            return (gaze_x * self.screen_width, gaze_y * self.screen_height)
        gaze_homogeneous = np.array([gaze_x, gaze_y, 1.0])
        screen_coords = self.calibration_matrix.T @ gaze_homogeneous
        screen_x = max(0, min(self.screen_width, screen_coords[0]))
        screen_y = max(0, min(self.screen_height, screen_coords[1]))
        return (screen_x, screen_y)

class EyeTrackingSystem:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.landmark_detector = FaceLandmarkDetector()
        self.gaze_model = CNNRNNGazeModel()
        self.eye_classifier = GaussianMixtureEyeClassifier()
        self.kalman_filter = KalmanGazeFilter()
        self.calibration_manager = CalibrationManager()
        self.is_running = False
        self.is_calibrated = False
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.accuracy_scores = deque(maxlen=100)
        
    async def initialize(self):
        logger.info("Initializing Eye Tracking System...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.gaze_model.initialize_models()
        logger.info("Eye tracking system initialized")
    
    async def start_tracking(self) -> AsyncGenerator[GazeData, None]:
        self.is_running = True
        logger.info("Starting eye tracking...")
        while self.is_running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame_count += 1
            landmarks = self.landmark_detector.detect_landmarks(frame)
            if landmarks is None:
                continue
            left_eye, right_eye = self.landmark_detector.extract_eye_regions(landmarks, frame.shape[:2])
            eye_state = self.eye_classifier.classify_eye_state(left_eye, right_eye)
            if eye_state == EyeState.CLOSED:
                continue
            left_eye_image = self._extract_eye_image(frame, left_eye)
            right_eye_image = self._extract_eye_image(frame, right_eye)
            eye_features = self.gaze_model.extract_eye_features(left_eye_image, right_eye_image)
            raw_gaze_x, raw_gaze_y, confidence = self.gaze_model.predict_gaze(eye_features)
            smoothed_x, smoothed_y = self.kalman_filter.get_smoothed_position((raw_gaze_x, raw_gaze_y), dt=0.033)
            screen_x, screen_y = self.calibration_manager.transform_gaze_to_screen(smoothed_x, smoothed_y)
            head_pose = self._estimate_head_pose(landmarks)
            gaze_data = GazeData(screen_x, screen_y, confidence, time.time(), left_eye, right_eye, head_pose, eye_state)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.accuracy_scores.append(confidence)
            yield gaze_data
            await asyncio.sleep(max(0, 0.033 - processing_time))
    
    async def calibrate_system(self, mode: CalibrationMode = CalibrationMode.NINE_POINT) -> bool:
        logger.info(f"Starting calibration with {mode.value} mode...")
        points = self.calibration_manager.setup_calibration_points(mode)
        for i, point in enumerate(points):
            logger.info(f"Calibrating point {i+1}/{len(points)}: ({point.screen_x}, {point.screen_y})")
            sample_count = 0
            async for gaze_data in self.start_tracking():
                if sample_count >= 30:
                    break
                self.calibration_manager.add_gaze_sample(i, gaze_data.screen_x / self.calibration_manager.screen_width,
                                                         gaze_data.screen_y / self.calibration_manager.screen_height)
                sample_count += 1
                await asyncio.sleep(0.033)
        success = self.calibration_manager.compute_calibration_matrix()
        self.is_calibrated = success
        return success
    
    def _extract_eye_image(self, frame: np.ndarray, eye_region: EyeRegion) -> np.ndarray:
        x, y, w, h = eye_region.bounding_box
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        eye_image = frame[y:y+h, x:x+w]
        return eye_image
    
    def _estimate_head_pose(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if len(landmarks) < 100:
            return (0.0, 0.0, 0.0)
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[362]
        eye_level = (left_eye[1] + right_eye[1]) / 2
        nose_level = nose_tip[1]
        pitch = np.arctan2(nose_level - eye_level, 100) * 180 / np.pi
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        nose_x = nose_tip[0]
        yaw = np.arctan2(nose_x - eye_center[0], 100) * 180 / np.pi
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
        return (pitch, yaw, roll)
    
    def get_performance_metrics(self) -> Dict:
        if not self.processing_times:
            return {"status": "no_data"}
        return {
            'fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
            'avg_processing_time': np.mean(self.processing_times),
            'avg_confidence': np.mean(self.accuracy_scores) if self.accuracy_scores else 0,
            'frame_count': self.frame_count,
            'is_calibrated': self.is_calibrated,
            'calibration_points': len(self.calibration_manager.calibration_points)
        }
    
    async def stop_tracking(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Eye tracking stopped")

async def demo_eye_tracking():
    tracker = EyeTrackingSystem()
    try:
        await tracker.initialize()
        calibration_success = await tracker.calibrate_system(CalibrationMode.FIVE_POINT)
        print(f"캘리브레이션 결과: {'성공' if calibration_success else '실패'}")
        print("시선 추적 시작 (5초간)...")
        start_time = time.time()
        async for gaze_data in tracker.start_tracking():
            if time.time() - start_time > 5.0:
                break
            print(f"시선 위치: ({gaze_data.screen_x:.1f}, {gaze_data.screen_y:.1f}), "
                  f"신뢰도: {gaze_data.confidence:.3f}, "
                  f"눈 상태: {gaze_data.eye_state.value}")
            await asyncio.sleep(0.5)
        metrics = tracker.get_performance_metrics()
        print(f"평균 FPS: {metrics['fps']:.1f}")
        print(f"평균 처리 시간: {metrics['avg_processing_time']:.3f}s")
        print(f"평균 신뢰도: {metrics['avg_confidence']:.3f}")
        print(f"총 프레임 수: {metrics['frame_count']}")
    finally:
        await tracker.stop_tracking()

if __name__ == "__main__":
    asyncio.run(demo_eye_tracking())
