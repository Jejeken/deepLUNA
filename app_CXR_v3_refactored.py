"""
CXR Segmentation and Classification Desktop Application - Enhanced v3
===================================================================

Modular and enhanced implementation with improved user experience:
- Model selection dropdowns
- Separated Load Image and Start Analysis functions
- Image preview functionality
- Enhanced drag & drop support
- Disabled Load button after analysis completion
- Better error handling and state management

Dependencies to install:
pip install customtkinter pillow torch torchxrayvision tensorflow opencv-python-headless numpy albumentations scikit-learn matplotlib timm

Required model files (place in models/segmentation and models/clf folders):
- models/segmentation/best_resunet_lung_segmentation.h5
- models/clf/optimized_classifier_efficientB2.pth

Author: Enhanced Modular Architecture v3
Version: 3.0 - Modular with Enhanced UX
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import warnings
import traceback
import tkinterdnd2 as tkdnd

# Import modular components
from ui_v3 import SidebarFrame, MainPanel
from model_management_v3 import ModelLoader
from image_handling_v3 import ImageLoader, DragDropHandler, ImageValidator
from analysis_controller_v3 import PipelineController, AnalysisStatus

warnings.filterwarnings('ignore')

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CXRAnalysisAppV3(ctk.CTk):
    """
    Enhanced CXR Analysis Application v3 with modular architecture

    Key improvements:
    - Model selection via dropdowns
    - Separated image loading and analysis
    - Image preview with drag & drop
    - Better state management
    - Disabled load button after analysis completion
    """

    def __init__(self):
        """Initialize the enhanced application"""
        super().__init__()

        # Configure window
        self.title("Enhanced CXR Analysis v3 - Multi-Model Pipeline")
        self.geometry("1800x1000")
        self.resizable(True, True)

        # Set consistent background color like v2
        self.configure(fg_color=("gray92", "gray14"))

        # Initialize tkinterdnd2 after window is created
        try:
            # Try different methods to enable tkinterdnd2
            if hasattr(tkdnd, 'TkinterDnD'):
                if hasattr(tkdnd.TkinterDnD, 'enable'):
                    tkdnd.TkinterDnD.enable(self)
                else:
                    # Alternative initialization
                    self.tk.call('package', 'require', 'tkdnd')
                print("DEBUG: tkinterdnd2 enabled successfully")
            else:
                raise AttributeError("TkinterDnD not available")
        except Exception as e:
            print(f"DEBUG: tkinterdnd2 initialization failed: {e}")
            print("DEBUG: Continuing without main window drag & drop")

        # Initialize core components
        self._init_core_components()

        # Initialize state
        self._init_application_state()

        # Setup UI
        self._setup_ui()

        # Setup callbacks and integrations
        self._setup_integrations()

        # Initialize models in background
        self._init_models_async()

    def _init_core_components(self):
        """Initialize core modular components"""
        # Image handling
        self.image_loader = ImageLoader()
        self.image_validator = ImageValidator()

        # Analysis pipeline
        self.pipeline_controller = PipelineController()

        # Model management will be handled by UI components

    def _init_application_state(self):
        """Initialize application state variables"""
        # Current state
        self.current_image_path = None
        self.analysis_completed = False

        # State locks
        self.state_lock = threading.Lock()

        # Drag and drop handler
        self.drag_drop_handler = DragDropHandler(
            on_image_loaded=self.handle_image_drop,
            on_validation_failed=self.handle_drop_validation_failed,
            on_loading_started=self.handle_drop_loading_started,
            on_loading_finished=self.handle_drop_loading_finished
        )

    def _setup_ui(self):
        """Setup the enhanced user interface"""
        # Configure main grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create enhanced sidebar with model selection
        self.sidebar = SidebarFrame(self)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Create main panel with enhanced features
        self.main_panel = MainPanel(self)
        self.main_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self,
            text="Ready - Enhanced Pipeline v3 Initializing...",
            anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.status_bar.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="ew")

        # Model selector is now integrated into sidebar
        # No need for separate ModelSelector widget

    def _setup_integrations(self):
        """Setup integrations between components"""
        # Setup pipeline controller callbacks
        self.pipeline_controller.add_callback('on_status_change', self.on_analysis_status_change)
        self.pipeline_controller.add_callback('on_progress_update', self.on_analysis_progress_update)
        self.pipeline_controller.add_callback('on_analysis_complete', self.on_analysis_complete)
        self.pipeline_controller.add_callback('on_analysis_error', self.on_analysis_error)
        self.pipeline_controller.add_callback('on_analysis_cancelled', self.on_analysis_cancelled)

        # Connect drag and drop to main panel
        self.main_panel.preview_frame.drag_drop_frame.parent = self

    def _init_models_async(self):
        """Initialize models in background thread"""
        def init_models():
            try:
                self.update_status("Initializing enhanced pipeline...")
                # Model scanning is handled by sidebar dropdowns
                self.after(1000, lambda: self.update_status("Ready - Select models from dropdowns and load image"))
            except Exception as e:
                error_msg = f"Initialization failed: {str(e)}"
                self.after(0, lambda: self.update_status(error_msg))

        thread = threading.Thread(target=init_models, daemon=True)
        thread.start()

    # Event Handlers
    def load_image(self):
        """Load image via file dialog (key requirement: disabled after analysis)"""
        with self.state_lock:
            # Check if analysis is completed (key requirement)
            if self.analysis_completed:
                messagebox.showinfo(
                    "Load Disabled",
                    "Load X-Ray Image is disabled after analysis completion.\n"
                    "Use 'Change Image' to load a new image and reset the application."
                )
                return

            # Check if models are selected
            selected_models = self.sidebar.get_selected_model_info()
            if not selected_models['segmentation'] or not selected_models['classification']:
                messagebox.showwarning("Warning", "Please select models first before selecting an image.")
                return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select X-Ray Image",
            filetypes=self.drag_drop_handler.create_file_filter_string()
        )

        if file_path:
            self._load_specific_image(file_path)

    def _load_specific_image(self, file_path: str):
        """Load specific image file"""
        def load_worker():
            try:
                self.update_status("Loading image...")

                # Load and validate image
                load_result = self.image_loader.load_image(file_path, validate=True)

                if load_result['success']:
                    # Update state
                    with self.state_lock:
                        self.current_image_path = file_path

                    # Update UI on main thread
                    self.after(0, lambda: self._handle_image_loaded_success(load_result))
                else:
                    # Handle failure
                    self.after(0, lambda: self._handle_image_load_failure(load_result['error_message']))

            except Exception as e:
                error_msg = f"Failed to load image: {str(e)}"
                self.after(0, lambda: self._handle_image_load_failure(error_msg))

        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()

    def _handle_image_loaded_success(self, load_result):
        """Handle successful image loading"""
        try:
            # Update sidebar
            self.sidebar.set_image_loaded(self.current_image_path)

            # Load image preview in main panel
            success = self.main_panel.load_image_preview(self.current_image_path)

            if success:
                self.update_status(f"Image loaded: {os.path.basename(self.current_image_path)}")

                # Show validation warnings if any
                validation = load_result.get('validation')
                if validation and validation.get('warnings'):
                    warning_msg = "Image loaded with warnings:\n"
                    for warning in validation['warnings'][:3]:
                        warning_msg += f"• {warning}\n"
                    if len(validation['warnings']) > 3:
                        warning_msg += f"... and {len(validation['warnings']) - 3} more"

                    messagebox.showwarning("Image Validation", warning_msg)
            else:
                self._handle_image_load_failure("Failed to display image preview")

        except Exception as e:
            self._handle_image_load_failure(f"Error handling loaded image: {str(e)}")

    def _handle_image_load_failure(self, error_message):
        """Handle image loading failure"""
        messagebox.showerror("Image Loading Error", error_message)
        self.update_status("Ready - Select an image to analyze")

        # Reset state
        with self.state_lock:
            self.current_image_path = None

    def change_image_with_confirmation(self):
        """Show confirmation dialog before changing image (key requirement)"""
        result = messagebox.askyesno(
            "Change Image Confirmation",
            "Are you sure you want to change the current image?\n\n"
            "This will:\n"
            "• Stop any ongoing analysis\n"
            "• Clear all current results\n"
            "• Reset the application to initial state\n"
            "• Re-enable the Load X-Ray Image button\n\n"
            "Do you want to continue?",
            icon="warning"
        )
        if result:
            self.reset_application()

    def reset_application(self):
        """Reset application to initial state (key requirement)"""
        try:
            # Cancel any running analysis
            if self.pipeline_controller.is_analysis_running():
                self.pipeline_controller.cancel_analysis()

            # Reset state
            with self.state_lock:
                self.current_image_path = None
                self.analysis_completed = False  # Key: re-enable load button

            # Reset UI components
            self.sidebar.reset_ui()
            self.main_panel.clear_all_displays()

            # Reset image loader
            self.image_loader.clear_current_image()

            # Reset pipeline
            self.pipeline_controller.reset_analysis()

            self.update_status("Application reset - Ready to load new image")

        except Exception as e:
            self.update_status(f"Reset error: {str(e)}")

    def start_analysis(self):
        """Start analysis (key requirement: separated from load image)"""
        with self.state_lock:
            # Validate state
            if not self.current_image_path:
                messagebox.showwarning("Warning", "Please load an image first")
                return

            # Check if models are selected (no longer need separate models_loaded flag)
            selected_models_check = self.sidebar.get_selected_model_info()
            if not selected_models_check['segmentation'] or not selected_models_check['classification']:
                messagebox.showwarning("Warning", "Please select models first")
                return

            if self.pipeline_controller.is_analysis_running():
                messagebox.showinfo("Info", "Analysis is already in progress")
                return

        # Get selected models from sidebar
        selected_model_info = self.sidebar.get_selected_model_info()
        if not selected_model_info['segmentation'] or not selected_model_info['classification']:
            messagebox.showwarning("Warning", "Please select both segmentation and classification models")
            return

        # For now, we'll need to load the models on-demand
        # This is a simplified approach - in a full implementation you'd want to cache loaded models
        from model_management_v3 import ModelLoader
        model_loader = ModelLoader()

        # Load segmentation model
        seg_model, seg_status = model_loader.load_segmentation_model(selected_model_info['segmentation'])
        if seg_model is None:
            messagebox.showerror("Error", f"Failed to load segmentation model: {seg_status}")
            return

        # Load classification model
        clf_model, gradcam, clf_status = model_loader.load_classification_model(selected_model_info['classification'])
        if clf_model is None:
            messagebox.showerror("Error", f"Failed to load classification model: {clf_status}")
            return

        selected_models = {
            'segmentation_model': seg_model,
            'classification_model': clf_model,
            'gradcam': gradcam,
            'seg_model_name': selected_model_info['segmentation'],
            'clf_model_name': selected_model_info['classification']
        }

        # Validate inputs
        validation = self.pipeline_controller.validate_analysis_inputs(
            self.current_image_path,
            {
                'segmentation': selected_models['seg_model_name'],
                'classification': selected_models['clf_model_name']
            }
        )

        if not validation['valid']:
            error_msg = "Cannot start analysis:\n" + "\n".join(validation['errors'])
            messagebox.showerror("Validation Error", error_msg)
            return

        # Set models in pipeline
        self.pipeline_controller.set_models(
            selected_models['segmentation_model'],
            selected_models['classification_model'],
            selected_models['gradcam']
        )

        # Update UI state
        self.sidebar.set_analysis_started()
        self.main_panel.show_analysis_state()

        # Start analysis
        success = self.pipeline_controller.start_analysis(
            self.current_image_path,
            {
                'segmentation': selected_models['seg_model_name'],
                'classification': selected_models['clf_model_name']
            }
        )

        if not success:
            messagebox.showerror("Error", "Failed to start analysis")
            self.sidebar.set_analysis_stopped()

    def stop_analysis(self):
        """Stop ongoing analysis"""
        if self.pipeline_controller.is_analysis_running():
            self.pipeline_controller.cancel_analysis()
            self.update_status("Stopping analysis...")
        else:
            messagebox.showinfo("Info", "No analysis is currently running")

    # Drag and Drop Handlers
    def handle_image_drop(self, load_result):
        """Handle successful image drop"""
        try:
            file_path = load_result['info']['file_path']

            with self.state_lock:
                # Check if analysis is completed (key requirement)
                if self.analysis_completed:
                    messagebox.showinfo(
                        "Drop Disabled",
                        "Drag & Drop is disabled after analysis completion.\n"
                        "Use 'Change Image' to reset and load a new image."
                    )
                    return

                # Check if models are loaded
                if not self.models_loaded:
                    messagebox.showwarning("Warning", "Please load models first before dropping an image.")
                    return

                self.current_image_path = file_path

            # Update UI
            self._handle_image_loaded_success(load_result)

        except Exception as e:
            self._handle_image_load_failure(f"Drag and drop error: {str(e)}")

    def handle_image_specific_drop(self, file_path):
        """Handle specific file drop (alternative entry point)"""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please load models first")
            return

        self._load_specific_image(file_path)

    def handle_drop_validation_failed(self, file_path, validation_result):
        """Handle validation failure from drag and drop"""
        errors = validation_result.get('errors', ['Unknown validation error'])
        error_msg = f"Cannot load dropped image:\n" + "\n".join(errors)
        messagebox.showerror("Validation Error", error_msg)

    def handle_drop_loading_started(self, file_path):
        """Handle drag and drop loading started"""
        self.update_status("Loading dropped image...")

    def handle_drop_loading_finished(self, file_path, success):
        """Handle drag and drop loading finished"""
        if not success:
            self.update_status("Ready - Select an image to analyze")

    # Analysis Pipeline Callbacks
    def on_analysis_status_change(self, status, progress_info):
        """Handle analysis status changes"""
        status_messages = {
            AnalysisStatus.INITIALIZING: "Initializing analysis...",
            AnalysisStatus.LOADING_IMAGE: "Loading image...",
            AnalysisStatus.RUNNING_SEGMENTATION: "Running segmentation...",
            AnalysisStatus.APPLYING_MASKS: "Applying masks...",
            AnalysisStatus.RUNNING_CLASSIFICATION: "Running classification...",
            AnalysisStatus.GENERATING_GRADCAM: "Generating visualizations...",
            AnalysisStatus.COMPILING_RESULTS: "Compiling results...",
            AnalysisStatus.COMPLETED: "Analysis completed",
            AnalysisStatus.ERROR: "Analysis failed",
            AnalysisStatus.CANCELLED: "Analysis cancelled"
        }

        message = status_messages.get(status, f"Status: {status.name}")
        self.update_status(message)

    def on_analysis_progress_update(self, progress):
        """Handle analysis progress updates"""
        try:
            progress_text = f"{progress.step_name} ({progress.current_step}/{progress.total_steps})"
            if progress.progress_percentage > 0:
                progress_text += f" - {progress.progress_percentage:.0f}%"

            self.update_status(progress_text)

        except Exception as e:
            print(f"Error updating progress: {e}")

    def on_analysis_complete(self, results):
        """Handle analysis completion"""
        try:
            # Update state (key requirement: disable load button)
            with self.state_lock:
                self.analysis_completed = True

            # Update UI
            self.sidebar.set_analysis_completed()
            self.sidebar.update_results(results)

            # Update main panel with results
            self.main_panel.update_processing_steps(results)
            self.main_panel.update_final_results(results)
            self.main_panel.update_detailed_analysis(results)

            # Final status
            prediction = results['prediction_label']
            confidence = results['confidence']
            self.update_status(f"Analysis Complete: {prediction} ({confidence:.1%} confidence)")

        except Exception as e:
            self.update_status(f"Error displaying results: {str(e)}")

    def on_analysis_error(self, error_message, error_traceback):
        """Handle analysis error"""
        self.sidebar.set_analysis_stopped()
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_message}")
        self.update_status("Analysis failed")

    def on_analysis_cancelled(self):
        """Handle analysis cancellation"""
        self.sidebar.set_analysis_stopped()
        self.update_status("Analysis cancelled")

    # Model Management - now handled by sidebar integration
    # Models are selected via dropdown and loaded on-demand during analysis

    # Utility Methods
    def update_status(self, message: str):
        """Update status bar message thread-safely"""
        try:
            # Schedule status update on main thread
            def update():
                self.status_bar.configure(text=message)

            if threading.current_thread() == threading.main_thread():
                update()
            else:
                self.after(0, update)

        except Exception as e:
            print(f"Error updating status: {e}")

    def export_results(self):
        """Export analysis results"""
        try:
            results = self.pipeline_controller.get_results()
            if not results:
                messagebox.showinfo("Info", "No results to export")
                return

            # Export as JSON
            file_path = filedialog.asksaveasfilename(
                title="Export Analysis Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                report = self.pipeline_controller.export_analysis_report('json')
                with open(file_path, 'w') as f:
                    f.write(report)

                messagebox.showinfo("Success", f"Results exported to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """Enhanced CXR Analysis v3
Multi-Model Pipeline with Advanced UI

Features:
• Model selection via dropdowns
• Separated Load Image and Start Analysis
• Image preview with drag & drop support
• Enhanced error handling and validation
• Disabled Load button after analysis completion
• Comprehensive analysis reporting

Developed with modular architecture for
better maintainability and extensibility."""

        messagebox.showinfo("About", about_text)


def main():
    """Main function to start the enhanced application"""
    try:
        print("Starting Enhanced CXR Analysis Application v3")
        print("Multi-Model Pipeline with Advanced Features")
        print("=" * 65)

        # Create and run application
        app = CXRAnalysisAppV3()

        # Setup menu (if needed)
        # Could add menu bar here for additional features

        app.mainloop()

    except Exception as e:
        print(f"Application startup failed: {e}")
        print(traceback.format_exc())
        try:
            messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
        except:
            pass


if __name__ == "__main__":
    main()