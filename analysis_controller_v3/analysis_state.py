"""
Analysis State Management for CXR Analysis v3
=============================================

State management for the analysis pipeline with progress tracking.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import threading
import time


class AnalysisStatus(Enum):
    """Analysis status enumeration"""
    IDLE = auto()
    INITIALIZING = auto()
    LOADING_IMAGE = auto()
    VALIDATING_IMAGE = auto()
    RUNNING_SEGMENTATION = auto()
    APPLYING_MASKS = auto()
    RUNNING_CLASSIFICATION = auto()
    GENERATING_GRADCAM = auto()
    COMPILING_RESULTS = auto()
    COMPLETED = auto()
    ERROR = auto()
    CANCELLED = auto()


@dataclass
class AnalysisProgress:
    """Analysis progress information"""
    current_step: int = 0
    total_steps: int = 7
    step_name: str = ""
    step_description: str = ""
    progress_percentage: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0


@dataclass
class AnalysisState:
    """Complete analysis state"""
    status: AnalysisStatus = AnalysisStatus.IDLE
    progress: AnalysisProgress = field(default_factory=AnalysisProgress)

    # Input data
    image_path: Optional[str] = None
    selected_models: Dict[str, str] = field(default_factory=dict)

    # Intermediate results
    original_image: Optional[Any] = None
    segmentation_mask: Optional[Any] = None
    masked_images: Dict[str, Any] = field(default_factory=dict)
    classification_results: Dict[str, Any] = field(default_factory=dict)
    gradcam_result: Optional[Any] = None

    # Final results
    final_results: Optional[Dict[str, Any]] = None

    # Error information
    error_message: str = ""
    error_traceback: str = ""

    # Timing information
    start_time: float = 0.0
    end_time: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)

    # Cancellation
    cancel_requested: bool = False

    def __post_init__(self):
        """Initialize state after creation"""
        self.lock = threading.Lock()

    def reset(self):
        """Reset state to initial values"""
        with self.lock:
            self.status = AnalysisStatus.IDLE
            self.progress = AnalysisProgress()
            self.image_path = None
            self.selected_models = {}
            self.original_image = None
            self.segmentation_mask = None
            self.masked_images = {}
            self.classification_results = {}
            self.gradcam_result = None
            self.final_results = None
            self.error_message = ""
            self.error_traceback = ""
            self.start_time = 0.0
            self.end_time = 0.0
            self.step_times = {}
            self.cancel_requested = False

    def set_status(self, status: AnalysisStatus):
        """Set analysis status thread-safely"""
        with self.lock:
            self.status = status

    def set_error(self, message: str, traceback: str = ""):
        """Set error state"""
        with self.lock:
            self.status = AnalysisStatus.ERROR
            self.error_message = message
            self.error_traceback = traceback
            self.end_time = time.time()

    def set_cancelled(self):
        """Set cancelled state"""
        with self.lock:
            self.status = AnalysisStatus.CANCELLED
            self.cancel_requested = True
            self.end_time = time.time()

    def request_cancellation(self):
        """Request analysis cancellation"""
        with self.lock:
            self.cancel_requested = True

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested"""
        with self.lock:
            return self.cancel_requested

    def update_progress(self, step: int, step_name: str, step_description: str = ""):
        """Update progress information"""
        with self.lock:
            self.progress.current_step = step
            self.progress.step_name = step_name
            self.progress.step_description = step_description
            self.progress.progress_percentage = (step / self.progress.total_steps) * 100

            # Update timing
            current_time = time.time()
            if self.start_time > 0:
                self.progress.elapsed_time = current_time - self.start_time

                # Estimate remaining time
                if step > 0:
                    avg_step_time = self.progress.elapsed_time / step
                    remaining_steps = self.progress.total_steps - step
                    self.progress.estimated_remaining = avg_step_time * remaining_steps

    def start_analysis(self):
        """Start analysis timing"""
        with self.lock:
            self.start_time = time.time()
            self.end_time = 0.0
            self.step_times = {}
            self.cancel_requested = False

    def complete_analysis(self):
        """Complete analysis timing"""
        with self.lock:
            self.end_time = time.time()
            self.status = AnalysisStatus.COMPLETED
            self.progress.progress_percentage = 100.0

    def start_step(self, step_name: str):
        """Start timing for a specific step"""
        with self.lock:
            self.step_times[f"{step_name}_start"] = time.time()

    def end_step(self, step_name: str):
        """End timing for a specific step"""
        with self.lock:
            start_key = f"{step_name}_start"
            end_key = f"{step_name}_duration"

            if start_key in self.step_times:
                self.step_times[end_key] = time.time() - self.step_times[start_key]

    def get_total_duration(self) -> float:
        """Get total analysis duration"""
        with self.lock:
            if self.start_time > 0:
                end_time = self.end_time if self.end_time > 0 else time.time()
                return end_time - self.start_time
            return 0.0

    def get_step_duration(self, step_name: str) -> float:
        """Get duration for a specific step"""
        with self.lock:
            duration_key = f"{step_name}_duration"
            return self.step_times.get(duration_key, 0.0)

    def get_progress_info(self) -> Dict[str, Any]:
        """Get progress information as dictionary"""
        with self.lock:
            return {
                'status': self.status.name,
                'current_step': self.progress.current_step,
                'total_steps': self.progress.total_steps,
                'step_name': self.progress.step_name,
                'step_description': self.progress.step_description,
                'progress_percentage': self.progress.progress_percentage,
                'elapsed_time': self.progress.elapsed_time,
                'estimated_remaining': self.progress.estimated_remaining,
                'total_duration': self.get_total_duration()
            }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary"""
        with self.lock:
            return {
                'status': self.status.name,
                'progress': self.get_progress_info(),
                'has_image': self.image_path is not None,
                'has_results': self.final_results is not None,
                'has_error': bool(self.error_message),
                'error_message': self.error_message,
                'selected_models': self.selected_models.copy(),
                'step_durations': {k: v for k, v in self.step_times.items() if k.endswith('_duration')},
                'cancellation_requested': self.cancel_requested
            }

    def is_running(self) -> bool:
        """Check if analysis is currently running"""
        with self.lock:
            return self.status not in [
                AnalysisStatus.IDLE,
                AnalysisStatus.COMPLETED,
                AnalysisStatus.ERROR,
                AnalysisStatus.CANCELLED
            ]

    def is_finished(self) -> bool:
        """Check if analysis is finished (completed, error, or cancelled)"""
        with self.lock:
            return self.status in [
                AnalysisStatus.COMPLETED,
                AnalysisStatus.ERROR,
                AnalysisStatus.CANCELLED
            ]

    def can_start_analysis(self) -> bool:
        """Check if analysis can be started"""
        with self.lock:
            return (self.status == AnalysisStatus.IDLE and
                    self.image_path is not None and
                    not self.cancel_requested)

    def can_cancel_analysis(self) -> bool:
        """Check if analysis can be cancelled"""
        with self.lock:
            return self.is_running() and not self.cancel_requested


class StateManager:
    """Thread-safe state manager for analysis pipeline"""

    def __init__(self):
        """Initialize state manager"""
        self.state = AnalysisState()
        self.observers = []  # List of observer callbacks
        self.observer_lock = threading.Lock()

    def add_observer(self, callback):
        """Add state change observer"""
        with self.observer_lock:
            if callback not in self.observers:
                self.observers.append(callback)

    def remove_observer(self, callback):
        """Remove state change observer"""
        with self.observer_lock:
            if callback in self.observers:
                self.observers.remove(callback)

    def notify_observers(self):
        """Notify all observers of state change"""
        with self.observer_lock:
            observers_copy = self.observers.copy()

        for callback in observers_copy:
            try:
                callback(self.state)
            except Exception as e:
                print(f"Error notifying observer: {e}")

    def update_status(self, status: AnalysisStatus):
        """Update status and notify observers"""
        self.state.set_status(status)
        self.notify_observers()

    def update_progress(self, step: int, step_name: str, step_description: str = ""):
        """Update progress and notify observers"""
        self.state.update_progress(step, step_name, step_description)
        self.notify_observers()

    def set_error(self, message: str, traceback: str = ""):
        """Set error state and notify observers"""
        self.state.set_error(message, traceback)
        self.notify_observers()

    def request_cancellation(self):
        """Request cancellation and notify observers"""
        self.state.request_cancellation()
        self.notify_observers()

    def reset_state(self):
        """Reset state and notify observers"""
        self.state.reset()
        self.notify_observers()

    def get_state(self) -> AnalysisState:
        """Get current state (read-only access recommended)"""
        return self.state

    def get_state_copy(self) -> Dict[str, Any]:
        """Get a copy of current state as dictionary"""
        return self.state.get_state_summary()

    def start_analysis(self, image_path: str, selected_models: Dict[str, str]):
        """Start analysis with given parameters"""
        self.state.reset()
        self.state.image_path = image_path
        self.state.selected_models = selected_models
        self.state.start_analysis()
        self.update_status(AnalysisStatus.INITIALIZING)

    def complete_analysis(self, results: Dict[str, Any]):
        """Complete analysis with results"""
        self.state.final_results = results
        self.state.complete_analysis()
        self.notify_observers()

    def store_intermediate_result(self, key: str, value: Any):
        """Store intermediate analysis result"""
        with self.state.lock:
            if key == 'original_image':
                self.state.original_image = value
            elif key == 'segmentation_mask':
                self.state.segmentation_mask = value
            elif key == 'masked_images':
                self.state.masked_images = value
            elif key == 'classification_results':
                self.state.classification_results = value
            elif key == 'gradcam_result':
                self.state.gradcam_result = value

    def get_intermediate_result(self, key: str) -> Any:
        """Get intermediate analysis result"""
        with self.state.lock:
            if key == 'original_image':
                return self.state.original_image
            elif key == 'segmentation_mask':
                return self.state.segmentation_mask
            elif key == 'masked_images':
                return self.state.masked_images
            elif key == 'classification_results':
                return self.state.classification_results
            elif key == 'gradcam_result':
                return self.state.gradcam_result
            return None