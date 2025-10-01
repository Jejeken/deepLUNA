"""
Pipeline Controller for CXR Analysis v3
=======================================

Main controller that orchestrates the analysis pipeline with UI integration.
"""

import threading
import time
from typing import Dict, Any, Optional, Callable, List
import traceback

from .analysis_pipeline import AnalysisPipeline
from .analysis_state import StateManager, AnalysisStatus


class PipelineController:
    """Main controller for the analysis pipeline"""

    def __init__(self):
        """Initialize pipeline controller"""
        # Core components
        self.state_manager = StateManager()
        self.pipeline = AnalysisPipeline(self.state_manager)

        # Threading
        self.analysis_thread = None
        self.analysis_lock = threading.Lock()

        # Callbacks for UI integration
        self.callbacks = {
            'on_status_change': [],
            'on_progress_update': [],
            'on_analysis_complete': [],
            'on_analysis_error': [],
            'on_analysis_cancelled': []
        }

        # Setup state observer
        self.state_manager.add_observer(self._on_state_change)

        # Configuration
        self.config = {
            'auto_cleanup': True,
            'max_concurrent_analyses': 1,
            'enable_detailed_logging': True
        }

    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific events"""
        if event_type in self.callbacks:
            if callback not in self.callbacks[event_type]:
                self.callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def remove_callback(self, event_type: str, callback: Callable):
        """Remove callback for specific events"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger callbacks for specific event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type].copy():  # Copy to avoid modification during iteration
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    print(f"Error in callback for {event_type}: {e}")

    def _on_state_change(self, state):
        """Handle state changes from state manager"""
        try:
            # Trigger appropriate callbacks based on state
            if state.status == AnalysisStatus.COMPLETED:
                self._trigger_callbacks('on_analysis_complete', state.final_results)
            elif state.status == AnalysisStatus.ERROR:
                self._trigger_callbacks('on_analysis_error', state.error_message, state.error_traceback)
            elif state.status == AnalysisStatus.CANCELLED:
                self._trigger_callbacks('on_analysis_cancelled')

            # Always trigger status and progress callbacks
            self._trigger_callbacks('on_status_change', state.status, state.get_progress_info())
            self._trigger_callbacks('on_progress_update', state.progress)

        except Exception as e:
            print(f"Error handling state change: {e}")

    def set_models(self, segmentation_model, classification_model, gradcam=None):
        """Set models for the pipeline"""
        self.pipeline.set_models(segmentation_model, classification_model, gradcam)

    def start_analysis(self, image_path: str, selected_models: Dict[str, str]) -> bool:
        """
        Start analysis in background thread

        Returns:
            bool: True if analysis started successfully, False otherwise
        """
        with self.analysis_lock:
            # Check if analysis is already running
            if self.is_analysis_running():
                print("Analysis already in progress")
                return False

            # Validate inputs
            if not image_path or not selected_models:
                self._trigger_callbacks('on_analysis_error', "Invalid input parameters", "")
                return False

            # Check if models are set
            if not self.pipeline._validate_models():
                self._trigger_callbacks('on_analysis_error', "Models not loaded", "")
                return False

            # Start analysis thread
            self.analysis_thread = threading.Thread(
                target=self._run_analysis_thread,
                args=(image_path, selected_models),
                daemon=True
            )
            self.analysis_thread.start()

            return True

    def _run_analysis_thread(self, image_path: str, selected_models: Dict[str, str]):
        """Run analysis in background thread"""
        try:
            # Run the complete analysis pipeline
            result = self.pipeline.run_complete_analysis(image_path, selected_models)

            # Handle result
            if result.get('success'):
                # Analysis completed successfully - callbacks already triggered by state manager
                pass
            elif result.get('cancelled'):
                # Analysis was cancelled - callbacks already triggered by state manager
                pass
            else:
                # Analysis failed
                error_msg = result.get('error', 'Unknown error')
                error_trace = result.get('traceback', '')
                self.state_manager.set_error(error_msg, error_trace)

        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected analysis error: {str(e)}"
            error_trace = traceback.format_exc()
            self.state_manager.set_error(error_msg, error_trace)

        finally:
            # Cleanup
            if self.config['auto_cleanup']:
                self._cleanup_after_analysis()

    def cancel_analysis(self) -> bool:
        """
        Cancel current analysis

        Returns:
            bool: True if cancellation was requested, False if no analysis running
        """
        with self.analysis_lock:
            if not self.is_analysis_running():
                return False

            # Request cancellation
            self.pipeline.cancel_analysis()
            return True

    def is_analysis_running(self) -> bool:
        """Check if analysis is currently running"""
        return (self.analysis_thread is not None and
                self.analysis_thread.is_alive() and
                self.state_manager.get_state().is_running())

    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        state = self.state_manager.get_state()
        return {
            'is_running': self.is_analysis_running(),
            'status': state.status.name,
            'progress': state.get_progress_info(),
            'has_results': state.final_results is not None,
            'has_error': bool(state.error_message),
            'can_start': state.can_start_analysis(),
            'can_cancel': state.can_cancel_analysis()
        }

    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get analysis results if available"""
        state = self.state_manager.get_state()
        return state.final_results

    def get_error_info(self) -> Optional[Dict[str, str]]:
        """Get error information if analysis failed"""
        state = self.state_manager.get_state()
        if state.error_message:
            return {
                'message': state.error_message,
                'traceback': state.error_traceback
            }
        return None

    def reset_analysis(self):
        """Reset analysis state"""
        with self.analysis_lock:
            if self.is_analysis_running():
                self.cancel_analysis()
                # Wait briefly for cancellation
                time.sleep(0.1)

            self.state_manager.reset_state()
            self.analysis_thread = None

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for analysis to complete

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if analysis completed, False if timeout
        """
        if not self.analysis_thread:
            return True

        try:
            self.analysis_thread.join(timeout=timeout)
            return not self.analysis_thread.is_alive()
        except Exception:
            return False

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get comprehensive pipeline information"""
        return {
            'controller_config': self.config.copy(),
            'pipeline_info': self.pipeline.get_pipeline_info(),
            'analysis_status': self.get_analysis_status(),
            'thread_info': {
                'has_thread': self.analysis_thread is not None,
                'thread_alive': self.analysis_thread.is_alive() if self.analysis_thread else False,
                'thread_name': self.analysis_thread.name if self.analysis_thread else None
            }
        }

    def update_pipeline_config(self, new_config: Dict[str, Any]):
        """Update pipeline configuration"""
        self.pipeline.update_config(new_config)

    def update_controller_config(self, new_config: Dict[str, Any]):
        """Update controller configuration"""
        self.config.update(new_config)

    def _cleanup_after_analysis(self):
        """Cleanup after analysis completion"""
        try:
            # Additional cleanup can be added here
            pass
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of recent analyses (placeholder for future implementation)"""
        # This could be extended to maintain a history of analyses
        return []

    def export_analysis_report(self, format_type: str = 'json') -> Optional[str]:
        """Export analysis report (placeholder for future implementation)"""
        # This could be extended to export detailed reports
        results = self.get_results()
        if not results:
            return None

        if format_type == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        else:
            return str(results)

    def validate_analysis_inputs(self, image_path: str, selected_models: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate inputs before starting analysis

        Returns:
            Dict with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check image path
        if not image_path:
            validation['errors'].append("No image path provided")
            validation['valid'] = False

        # Check if file exists
        import os
        if image_path and not os.path.exists(image_path):
            validation['errors'].append(f"Image file not found: {image_path}")
            validation['valid'] = False

        # Check selected models
        if not selected_models:
            validation['errors'].append("No models selected")
            validation['valid'] = False
        else:
            required_keys = ['segmentation', 'classification']
            for key in required_keys:
                if key not in selected_models:
                    validation['errors'].append(f"Missing {key} model selection")
                    validation['valid'] = False

        # Check if models are loaded
        if not self.pipeline._validate_models():
            validation['errors'].append("Models not loaded in pipeline")
            validation['valid'] = False

        # Check if analysis is already running
        if self.is_analysis_running():
            validation['errors'].append("Analysis already in progress")
            validation['valid'] = False

        return validation

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the analysis"""
        state = self.state_manager.get_state()
        return {
            'total_duration': state.get_total_duration(),
            'step_durations': {k: v for k, v in state.step_times.items() if k.endswith('_duration')},
            'status': state.status.name,
            'has_completed': state.status == AnalysisStatus.COMPLETED
        }