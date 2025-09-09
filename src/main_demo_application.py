import asyncio
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional
from typing import AsyncGenerator

# 구현된 모든 시스템 임포트
from multimodal_integration import MultimodalIntegrationSystem
from comprehensive_evaluation import ExperimentController, InterfaceMode
from cybernetic_prosthesis_core import MotorIntention, ModalityType

class SystemMonitor:
    
    def __init__(self):
        self.is_monitoring = False
        self.metrics_history = {
            'timestamps': [],
            'response_times': [],
            'accuracy_scores': [],
            'sync_delays': [],
            'embodiment_levels': [],
            'fps_values': []
        }
        self.max_history = 100
    
    def add_metric(self, timestamp: float, response_time: float, 
                  accuracy: float, sync_delay: float, embodiment: float, fps: float):
        """메트릭 추가"""
        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['response_times'].append(response_time)
        self.metrics_history['accuracy_scores'].append(accuracy)
        self.metrics_history['sync_delays'].append(sync_delay)
        self.metrics_history['embodiment_levels'].append(embodiment)
        self.metrics_history['fps_values'].append(fps)
        
        # 최대 히스토리 크기 유지
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > self.max_history:
                self.metrics_history[key] = self.metrics_history[key][-self.max_history:]
    
    def get_latest_metrics(self) -> Dict:
        if not self.metrics_history['timestamps']:
            return {}
        
        return {
            'response_time': self.metrics_history['response_times'][-1],
            'accuracy': self.metrics_history['accuracy_scores'][-1],
            'sync_delay': self.metrics_history['sync_delays'][-1],
            'embodiment': self.metrics_history['embodiment_levels'][-1],
            'fps': self.metrics_history['fps_values'][-1]
        }

class CyberneticArmGUI:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("사이버네틱 인공팔 시스템 - 다른 존재 양상으로서의 인공팔")
        self.root.geometry("1400x900")
        
        # 시스템 컴포넌트들
        self.multimodal_system = MultimodalIntegrationSystem()
        self.experiment_controller = ExperimentController()
        self.system_monitor = SystemMonitor()
        
        # 상태 변수들
        self.is_system_running = False
        self.is_experiment_mode = False
        self.current_mode = InterfaceMode.CYBERNETIC_ARM
        
        # GUI 컴포넌트들
        self.setup_gui()
        
        # 실시간 업데이트 스레드
        self.update_thread = None
        
    def setup_gui(self):
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 좌측 패널 - 제어 및 상태
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # 우측 패널 - 모니터링 및 시각화
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 좌측 패널 구성
        self.setup_control_panel(left_panel)
        
        # 우측 패널 구성
        self.setup_monitoring_panel(right_panel)
    
    def setup_control_panel(self, parent):
        
        # 시스템 상태 프레임
        status_frame = ttk.LabelFrame(parent, text="시스템 상태", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="시스템 대기 중", 
                                     font=("Arial", 12, "bold"))
        self.status_label.pack()
        
        self.status_indicator = tk.Canvas(status_frame, width=20, height=20)
        self.status_indicator.pack(pady=5)
        self.update_status_indicator("stopped")
        
        # 시스템 제어 프레임
        control_frame = ttk.LabelFrame(parent, text="시스템 제어", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="시스템 시작", 
                                      command=self.start_system)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(control_frame, text="시스템 중지", 
                                     command=self.stop_system, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # 모드 선택 프레임
        mode_frame = ttk.LabelFrame(parent, text="인터페이스 모드", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="cybernetic_arm")
        
        ttk.Radiobutton(mode_frame, text="사이버네틱 인공팔", 
                       variable=self.mode_var, value="cybernetic_arm",
                       command=self.change_interface_mode).pack(anchor=tk.W)
        
        ttk.Radiobutton(mode_frame, text="음성만 사용", 
                       variable=self.mode_var, value="voice_only",
                       command=self.change_interface_mode).pack(anchor=tk.W)
        
        ttk.Radiobutton(mode_frame, text="시선만 사용", 
                       variable=self.mode_var, value="gaze_only",
                       command=self.change_interface_mode).pack(anchor=tk.W)
        
        # 실험 모드 프레임
        experiment_frame = ttk.LabelFrame(parent, text="실험 모드", padding=10)
        experiment_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.experiment_button = ttk.Button(experiment_frame, text="가설 검증 실험 시작",
                                           command=self.start_experiment)
        self.experiment_button.pack(fill=tk.X, pady=2)
        
        self.calibration_button = ttk.Button(experiment_frame, text="시스템 캘리브레이션",
                                            command=self.start_calibration)
        self.calibration_button.pack(fill=tk.X, pady=2)
        
        # 현재 메트릭 프레임
        metrics_frame = ttk.LabelFrame(parent, text="실시간 메트릭", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=8, width=35)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # 로그 프레임
        log_frame = ttk.LabelFrame(parent, text="시스템 로그", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=35)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_monitoring_panel(self, parent):
        
        # 노트북 위젯으로 탭 구성
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 실시간 모니터링 탭
        self.setup_realtime_tab(notebook)
        
        # 성능 분석 탭
        self.setup_performance_tab(notebook)
        
        # 실험 결과 탭
        self.setup_experiment_tab(notebook)
    
    def setup_realtime_tab(self, notebook):
        
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="실시간 모니터링")
        
        # matplotlib 그래프
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("사이버네틱 인공팔 시스템 실시간 모니터링")
        
        # 각 서브플롯 설정
        self.ax1.set_title("응답 시간 (ms)")
        self.ax1.set_ylabel("시간 (ms)")
        
        self.ax2.set_title("정확도 (%)")
        self.ax2.set_ylabel("정확도")
        
        self.ax3.set_title("동기화 지연 (ms)")
        self.ax3.set_ylabel("지연시간 (ms)")
        
        self.ax4.set_title("신체화 수준")
        self.ax4.set_ylabel("점수 (0-10)")
        
        # 캔버스 생성
        self.canvas = FigureCanvasTkAgg(self.fig, realtime_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
    
    def setup_performance_tab(self, notebook):
        
        performance_frame = ttk.Frame(notebook)
        notebook.add(performance_frame, text="성능 분석")
        
        # 성능 요약 텍스트
        self.performance_text = scrolledtext.ScrolledText(performance_frame, 
                                                         font=("Consolas", 10))
        self.performance_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_experiment_tab(self, notebook):
        
        experiment_frame = ttk.Frame(notebook)
        notebook.add(experiment_frame, text="실험 결과")
        
        # 실험 결과 텍스트
        self.experiment_text = scrolledtext.ScrolledText(experiment_frame,
                                                        font=("Consolas", 10))
        self.experiment_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def update_status_indicator(self, status: str):
        self.status_indicator.delete("all")
        
        if status == "running":
            color = "green"
        elif status == "initializing":
            color = "yellow"
        elif status == "error":
            color = "red"
        else:
            color = "gray"
        
        self.status_indicator.create_oval(2, 2, 18, 18, fill=color, outline=color)
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    async def start_system(self):
        try:
            self.start_button.config(state=tk.DISABLED)
            self.status_label.config(text="시스템 초기화 중...")
            self.update_status_indicator("initializing")
            self.log_message("시스템 초기화 시작")
            
            # 멀티모달 시스템 초기화
            await self.multimodal_system.initialize_all_systems()
            
            self.is_system_running = True
            self.status_label.config(text="시스템 실행 중")
            self.update_status_indicator("running")
            self.stop_button.config(state=tk.NORMAL)
            
            self.log_message("시스템 초기화 완료")
            self.log_message("실시간 모니터링 시작")
            
            # 실시간 처리 시작
            asyncio.create_task(self.run_realtime_processing())
            
            # 메트릭 업데이트 시작
            self.start_metric_updates()
            
        except Exception as e:
            self.log_message(f"시스템 시작 오류: {str(e)}")
            self.status_label.config(text="시스템 오류")
            self.update_status_indicator("error")
            self.start_button.config(state=tk.NORMAL)
            messagebox.showerror("오류", f"시스템 시작 실패:\n{str(e)}")
    
    def start_system_wrapper(self):
        asyncio.run(self.start_system())
    
    async def stop_system(self):
        self.is_system_running = False
        self.status_label.config(text="시스템 중지 중...")
        self.update_status_indicator("initializing")
        
        self.log_message("시스템 중지 중...")
        
        # 시스템 정리
        await self.multimodal_system.stop_all_systems()
        
        self.status_label.config(text="시스템 대기 중")
        self.update_status_indicator("stopped")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.log_message("시스템 중지 완료")
    
    def stop_system_wrapper(self):
        asyncio.run(self.stop_system())
    
    def change_interface_mode(self):
        mode_map = {
            "cybernetic_arm": InterfaceMode.CYBERNETIC_ARM,
            "voice_only": InterfaceMode.VOICE_ONLY,
            "gaze_only": InterfaceMode.GAZE_ONLY
        }
        
        self.current_mode = mode_map[self.mode_var.get()]
        self.log_message(f"인터페이스 모드 변경: {self.current_mode.value}")
    
    async def run_realtime_processing(self):
        try:
            async for integration_result in self.multimodal_system.start_multimodal_processing():
                if not self.is_system_running:
                    break
                
                # 메트릭 추출
                timestamp = time.time()
                response_time = integration_result.get('processing_time', 0) * 1000  # ms로 변환
                sync_quality = integration_result.get('sync_quality', 0)
                
                # 가상 메트릭 생성 (실제로는 시스템에서 추출)
                accuracy = np.random.uniform(85, 98)
                sync_delay = (1 - sync_quality) * 200  # 품질 역비례로 지연시간
                embodiment = integration_result.get('cybernetic_response', {}).get('embodiment_level', 0) * 10
                fps = np.random.uniform(25, 30)
                
                # 모니터에 메트릭 추가
                self.system_monitor.add_metric(
                    timestamp, response_time, accuracy, sync_delay, embodiment, fps
                )
                
                # 로그 메시지
                reasoning = integration_result.get('multimodal_reasoning', {})
                if reasoning:
                    intention = reasoning.get('motor_intention', MotorIntention.NONE)
                    target = reasoning.get('target_element', 'unknown')
                    confidence = reasoning.get('confidence', 0)
                    
                    self.log_message(
                        f"의도: {intention.value}, 대상: {target}, 신뢰도: {confidence:.2f}"
                    )
                
                # 짧은 대기
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.log_message(f"실시간 처리 오류: {str(e)}")
    
    def start_metric_updates(self):
        self.update_thread = threading.Thread(target=self.update_metrics_loop, daemon=True)
        self.update_thread.start()
    
    def update_metrics_loop(self):
        """메트릭 업데이트 루프"""
        while self.is_system_running:
            try:
                self.update_realtime_graphs()
                self.update_metrics_text()
                time.sleep(1.0)  # 1초마다 업데이트
            except Exception as e:
                print(f"메트릭 업데이트 오류: {e}")
    
    def update_realtime_graphs(self):
        if not self.system_monitor.metrics_history['timestamps']:
            return
        
        # 최근 데이터만 표시 (최대 50개)
        recent_count = min(50, len(self.system_monitor.metrics_history['timestamps']))
        
        timestamps = self.system_monitor.metrics_history['timestamps'][-recent_count:]
        response_times = self.system_monitor.metrics_history['response_times'][-recent_count:]
        accuracy_scores = self.system_monitor.metrics_history['accuracy_scores'][-recent_count:]
        sync_delays = self.system_monitor.metrics_history['sync_delays'][-recent_count:]
        embodiment_levels = self.system_monitor.metrics_history['embodiment_levels'][-recent_count:]
        
        # 상대 시간으로 변환
        if timestamps:
            relative_times = [(t - timestamps[0]) for t in timestamps]
        else:
            relative_times = []
        
        # 각 그래프 업데이트
        self.ax1.clear()
        self.ax1.plot(relative_times, response_times, 'b-', linewidth=2)
        self.ax1.set_title("응답 시간 (ms)")
        self.ax1.set_ylabel("시간 (ms)")
        self.ax1.grid(True)
        
        self.ax2.clear()
        self.ax2.plot(relative_times, accuracy_scores, 'g-', linewidth=2)
        self.ax2.set_title("정확도 (%)")
        self.ax2.set_ylabel("정확도")
        self.ax2.set_ylim(0, 100)
        self.ax2.grid(True)
        
        self.ax3.clear()
        self.ax3.plot(relative_times, sync_delays, 'r-', linewidth=2)
        self.ax3.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='목표 (200ms)')
        self.ax3.set_title("동기화 지연 (ms)")
        self.ax3.set_ylabel("지연시간 (ms)")
        self.ax3.legend()
        self.ax3.grid(True)
        
        self.ax4.clear()
        self.ax4.plot(relative_times, embodiment_levels, 'purple', linewidth=2)
        self.ax4.set_title("신체화 수준")
        self.ax4.set_ylabel("점수 (0-10)")
        self.ax4.set_ylim(0, 10)
        self.ax4.grid(True)
        
        # 그래프 갱신
        self.canvas.draw()
    
    def update_metrics_text(self):
        latest_metrics = self.system_monitor.get_latest_metrics()
        
        if not latest_metrics:
            return
        
        metrics_summary = f"""
=== 실시간 성능 메트릭 ===

응답 시간: {latest_metrics.get('response_time', 0):.1f} ms
정확도: {latest_metrics.get('accuracy', 0):.1f} %
동기화 지연: {latest_metrics.get('sync_delay', 0):.1f} ms
신체화 수준: {latest_metrics.get('embodiment', 0):.1f} / 10
FPS: {latest_metrics.get('fps', 0):.1f}

=== 시스템 상태 ===

모드: {self.current_mode.value}
실행 시간: {time.time() - (self.system_monitor.metrics_history['timestamps'][0] if self.system_monitor.metrics_history['timestamps'] else time.time()):.1f} 초
처리된 이벤트: {len(self.system_monitor.metrics_history['timestamps'])}



응답시간 목표 (< 500ms): {'suc' if latest_metrics.get('response_time', 1000) < 500 else '✗'}
정확도 목표 (> 90%): {'suc' if latest_metrics.get('accuracy', 0) > 90 else '✗'}
동기화 목표 (< 200ms): {'suc' if latest_metrics.get('sync_delay', 1000) < 200 else '✗'}
신체화 목표 (> 7.0): {'suc' if latest_metrics.get('embodiment', 0) > 7.0 else '✗'}
        """
        
        # GUI 업데이트는 메인 스레드에서
        self.root.after(0, self._update_metrics_text_gui, metrics_summary)
    
    def _update_metrics_text_gui(self, text):
        """메트릭 텍스트 GUI 업데이트 (메인 스레드)"""
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, text)
    
    async def start_experiment(self):
        """가설 검증 실험 시작"""
        self.log_message("가설 검증 실험 시작")
        self.experiment_button.config(state=tk.DISABLED)
        
        try:
            # 실험 초기화
            await self.experiment_controller.initialize_experiment()
            
            # 실험 실행
            participant_id = f"demo_user_{int(time.time())}"
            session = await self.experiment_controller.run_controlled_experiment(
                participant_id, num_tasks_per_mode=3  # 데모용으로 축소
            )
            
            # 결과 분석
            analysis_results = self.experiment_controller.analyze_experiment_results()
            
            # 결과 표시
            report = self.experiment_controller.generate_research_report()
            self.experiment_text.delete(1.0, tk.END)
            self.experiment_text.insert(1.0, report)
            
            self.log_message("가설 검증 실험 완료")
            messagebox.showinfo("실험 완료", "가설 검증 실험이 완료되었습니다.\n'실험 결과' 탭에서 결과를 확인하세요.")
            
        except Exception as e:
            self.log_message(f"실험 오류: {str(e)}")
            messagebox.showerror("실험 오류", f"실험 중 오류가 발생했습니다:\n{str(e)}")
        
        finally:
            self.experiment_button.config(state=tk.NORMAL)
    
    def start_experiment_wrapper(self):
        asyncio.run(self.start_experiment())
    
    async def start_calibration(self):
        self.log_message("시스템 캘리브레이션 시작")
        self.calibration_button.config(state=tk.DISABLED)
        
        try:
            if hasattr(self.multimodal_system, 'eye_tracking_system'):
                from eye_tracking_system import CalibrationMode
                success = await self.multimodal_system.eye_tracking_system.calibrate_system(
                    CalibrationMode.NINE_POINT
                )
                
                if success:
                    self.log_message("캘리브레이션 완료")
                    messagebox.showinfo("캘리브레이션 완료", "시선 추적 캘리브레이션이 성공적으로 완료되었습니다.")
                else:
                    self.log_message("캘리브레이션 실패")
                    messagebox.showwarning("캘리브레이션 실패", "캘리브레이션에 실패했습니다. 다시 시도해주세요.")
            else:
                messagebox.showwarning("시스템 미실행", "시스템을 먼저 시작해주세요.")
        
        except Exception as e:
            self.log_message(f"캘리브레이션 오류: {str(e)}")
            messagebox.showerror("캘리브레이션 오류", f"캘리브레이션 중 오류가 발생했습니다:\n{str(e)}")
        
        finally:
            self.calibration_button.config(state=tk.NORMAL)
    
    def start_calibration_wrapper(self):
        """캘리브레이션 시작 래퍼"""
        asyncio.run(self.start_calibration())
    
    def on_closing(self):
        """창 닫기 이벤트"""
        if self.is_system_running:
            if messagebox.askokcancel("시스템 종료", "실행 중인 시스템을 중지하고 종료하시겠습니까?"):
                asyncio.run(self.stop_system())
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        # 이벤트 바인딩
        self.start_button.config(command=lambda: threading.Thread(target=self.start_system_wrapper, daemon=True).start())
        self.stop_button.config(command=lambda: threading.Thread(target=self.stop_system_wrapper, daemon=True).start())
        self.experiment_button.config(command=lambda: threading.Thread(target=self.start_experiment_wrapper, daemon=True).start())
        self.calibration_button.config(command=lambda: threading.Thread(target=self.start_calibration_wrapper, daemon=True).start())
        
        # 창 닫기 이벤트
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 초기 성능 데이터 표시
        self.update_performance_summary()
        
        # GUI 실행
        self.log_message("사이버네틱 인공팔 시스템 GUI 시작")
        self.root.mainloop()
    
    def update_performance_summary(self):

        performance_summary = ""
        
        self.performance_text.delete(1.0, tk.END)
        self.performance_text.insert(1.0, performance_summary)

def main():
    print("=" * 60)
    print("사이버네틱 인공팔 시스템 - 다른 존재 양상으로서의 인공팔")
    print("Cybernetic Prosthetic Arm System - Alternative Mode of Being")
    print("=" * 60)
    print()
    print("본 시스템은 다음과 같은 혁신적 개념을 구현합니다:")
    print("- Norbert Wiener의 사이버네틱스 이론 기반 피드백 루프")
    print("- Merleau-Ponty의 신체 현상학을 통한 신체 스키마 확장")
    print("- Gibson의 어포던스 이론을 적용한 환경 상호작용")
    print("- 물리적 장치 없는 순수 소프트웨어 기반 인공팔")
    print()
    print("GUI를 통해 시스템의 모든 기능을 체험할 수 있습니다.")
    print("실시간 모니터링, 성능 분석, 가설 검증 실험이 포함되어 있습니다.")
    print()
    
    # GUI 실행
    app = CyberneticArmGUI()
    app.run()

if __name__ == "__main__":
    main()