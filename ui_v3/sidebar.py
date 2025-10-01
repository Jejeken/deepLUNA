"""
Enhanced Sidebar Component for CXR Analysis v3
==============================================

Includes model selection, file management, and analysis results display.
"""

import customtkinter as ctk
import os
import threading
from model_management_v3 import ModelConfig


class SidebarFrame(ctk.CTkFrame):
    """Enhanced sidebar with model selection and improved controls"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=380, corner_radius=0,
                        fg_color=("gray90", "gray17"), **kwargs)

        # Store parent reference for callbacks
        self.parent = parent

        # Initialize model configuration
        self.model_config = ModelConfig()
        self.selected_seg_model = None
        self.selected_clf_model = None

        # Configure grid
        self.grid_rowconfigure(9, weight=1)

        # Setup UI components
        self._setup_header()
        self._setup_model_selection()
        self._setup_file_controls()
        self._setup_analysis_controls()
        self._setup_results_section()
        self._setup_model_status()

        # Initialize states
        self._reset_to_initial_state()

        # Initialize models with a delay
        self.after(500, self._load_models_async)

    def _setup_header(self):
        """Setup header section with title and version"""
        self.logo_label = ctk.CTkLabel(
            self,
            text="Enhanced CXR Analysis v3",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.version_label = ctk.CTkLabel(
            self,
            text="Multi-Model Pipeline with Enhanced UI",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 10))

    def _setup_model_selection(self):
        """Setup model selection dropdowns"""
        # Model Selection Section
        self.model_section_label = ctk.CTkLabel(
            self,
            text="Model Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.model_section_label.grid(row=2, column=0, padx=20, pady=(15, 5))

        # Segmentation model dropdown
        self.seg_model_label = ctk.CTkLabel(
            self,
            text="Segmentation Model:",
            font=ctk.CTkFont(size=12)
        )
        self.seg_model_label.grid(row=3, column=0, padx=20, pady=(5, 2), sticky="w")

        self.seg_model_dropdown = ctk.CTkComboBox(
            self,
            values=["Loading..."],
            state="disabled",
            width=340,
            command=self._on_seg_model_change
        )
        self.seg_model_dropdown.grid(row=4, column=0, padx=20, pady=(0, 8))

        # Classification model dropdown
        self.clf_model_label = ctk.CTkLabel(
            self,
            text="Classification Model:",
            font=ctk.CTkFont(size=12)
        )
        self.clf_model_label.grid(row=5, column=0, padx=20, pady=(5, 2), sticky="w")

        self.clf_model_dropdown = ctk.CTkComboBox(
            self,
            values=["Loading..."],
            state="disabled",
            width=340,
            command=self._on_clf_model_change
        )
        self.clf_model_dropdown.grid(row=6, column=0, padx=20, pady=(0, 8))

    def _setup_file_controls(self):
        """Setup file management controls"""
        # Separator
        self.separator1 = ctk.CTkFrame(self, height=2)
        self.separator1.grid(row=7, column=0, padx=20, pady=10, sticky="ew")

        # File Management Section
        self.file_section_label = ctk.CTkLabel(
            self,
            text="Image Management",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.file_section_label.grid(row=8, column=0, padx=20, pady=(10, 5))

        # Load image button
        self.load_button = ctk.CTkButton(
            self,
            text="Load X-Ray Image",
            command=self._on_load_image,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.load_button.grid(row=9, column=0, padx=20, pady=5)

        # Change image button
        self.change_button = ctk.CTkButton(
            self,
            text="Change Image",
            command=self._on_change_image,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            border_width=2,
            border_color="#3B82F6",
            text_color="#3B82F6",
            hover_color="#DC2626",
            state="disabled"
        )
        self.change_button.grid(row=10, column=0, padx=20, pady=(5, 10))

        # Current file label
        self.file_label = ctk.CTkLabel(
            self,
            text="No image selected",
            wraplength=340,
            font=ctk.CTkFont(size=12)
        )
        self.file_label.grid(row=11, column=0, padx=20, pady=5)

    def _setup_analysis_controls(self):
        """Setup analysis control buttons"""
        # Separator
        self.separator2 = ctk.CTkFrame(self, height=2)
        self.separator2.grid(row=12, column=0, padx=20, pady=10, sticky="ew")

        # Analysis Control Section
        self.analysis_section_label = ctk.CTkLabel(
            self,
            text="Analysis Control",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.analysis_section_label.grid(row=13, column=0, padx=20, pady=(10, 5))

        # Start analysis button
        self.start_analysis_button = ctk.CTkButton(
            self,
            text="🔬 Start Analysis",
            command=self._on_start_analysis,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            state="disabled"
        )
        self.start_analysis_button.grid(row=14, column=0, padx=20, pady=5)

        # Stop analysis button
        self.stop_analysis_button = ctk.CTkButton(
            self,
            text="⏹️ Stop Analysis",
            command=self._on_stop_analysis,
            height=35,
            font=ctk.CTkFont(size=12),
            fg_color="#EF4444",
            hover_color="#DC2626",
            state="disabled"
        )
        self.stop_analysis_button.grid(row=15, column=0, padx=20, pady=(5, 10))

    def _setup_results_section(self):
        """Setup analysis results display"""
        # Separator
        self.separator3 = ctk.CTkFrame(self, height=2)
        self.separator3.grid(row=16, column=0, padx=20, pady=10, sticky="ew")

        # Results section
        self.results_title = ctk.CTkLabel(
            self,
            text="Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.results_title.grid(row=17, column=0, padx=20, pady=(10, 5))

        # Results frame
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row=18, column=0, padx=20, pady=5, sticky="ew")

        # Result labels
        self.prediction_label = ctk.CTkLabel(
            self.results_frame,
            text="Prediction: --",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.prediction_label.grid(row=0, column=0, padx=15, pady=5, sticky="w")

        self.confidence_label = ctk.CTkLabel(
            self.results_frame,
            text="Confidence: --",
            font=ctk.CTkFont(size=12)
        )
        self.confidence_label.grid(row=1, column=0, padx=15, pady=2, sticky="w")

        self.prob_normal_label = ctk.CTkLabel(
            self.results_frame,
            text="Normal: --",
            font=ctk.CTkFont(size=11)
        )
        self.prob_normal_label.grid(row=2, column=0, padx=15, pady=2, sticky="w")

        self.prob_abnormal_label = ctk.CTkLabel(
            self.results_frame,
            text="Abnormal: --",
            font=ctk.CTkFont(size=11)
        )
        self.prob_abnormal_label.grid(row=3, column=0, padx=15, pady=2, sticky="w")

        self.ensemble_info = ctk.CTkLabel(
            self.results_frame,
            text="Method: Multi-Model Ensemble",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.ensemble_info.grid(row=4, column=0, padx=15, pady=2, sticky="w")

    def _setup_model_status(self):
        """Setup model status indicators"""
        # Model status frame
        self.model_info_frame = ctk.CTkFrame(self)
        self.model_info_frame.grid(row=19, column=0, padx=20, pady=(10, 20), sticky="ew")

        self.model_info_label = ctk.CTkLabel(
            self.model_info_frame,
            text="Model Status",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.model_info_label.grid(row=0, column=0, padx=15, pady=5)

        self.segmentation_status = ctk.CTkLabel(
            self.model_info_frame,
            text="Segmentation: Loading...",
            font=ctk.CTkFont(size=11)
        )
        self.segmentation_status.grid(row=1, column=0, padx=15, pady=2, sticky="w")

        self.classification_status = ctk.CTkLabel(
            self.model_info_frame,
            text="Classification: Loading...",
            font=ctk.CTkFont(size=11)
        )
        self.classification_status.grid(row=2, column=0, padx=15, pady=2, sticky="w")

    def _reset_to_initial_state(self):
        """Reset sidebar to initial state"""
        self.current_image_path = None
        self.analysis_complete = False

        # Reset button states
        self.load_button.configure(state="normal")
        self.change_button.configure(state="disabled")
        self.start_analysis_button.configure(state="disabled")
        self.stop_analysis_button.configure(state="disabled")

        # Reset results
        self.prediction_label.configure(text="Prediction: --", text_color="white")
        self.confidence_label.configure(text="Confidence: --")
        self.prob_normal_label.configure(text="Normal: --")
        self.prob_abnormal_label.configure(text="Abnormal: --")

    # Event handlers (to be connected to main app)
    def _on_load_image(self):
        """Handle load image button click"""
        if hasattr(self.parent, 'load_image'):
            self.parent.load_image()

    def _on_change_image(self):
        """Handle change image button click"""
        if hasattr(self.parent, 'change_image_with_confirmation'):
            self.parent.change_image_with_confirmation()

    def _on_start_analysis(self):
        """Handle start analysis button click"""
        if hasattr(self.parent, 'start_analysis'):
            self.parent.start_analysis()

    def _on_stop_analysis(self):
        """Handle stop analysis button click"""
        if hasattr(self.parent, 'stop_analysis'):
            self.parent.stop_analysis()

    # Public methods for state management
    def set_image_loaded(self, file_path):
        """Update UI when image is loaded"""
        self.current_image_path = file_path
        filename = os.path.basename(file_path)
        self.file_label.configure(text=f"Selected: {filename}")

        # Enable appropriate buttons
        self.change_button.configure(state="normal")
        self.start_analysis_button.configure(state="normal")

        # Disable load button if analysis is complete
        if self.analysis_complete:
            self.load_button.configure(state="disabled")

    def set_analysis_started(self):
        """Update UI when analysis starts"""
        self.start_analysis_button.configure(state="disabled")
        self.stop_analysis_button.configure(state="normal")
        self.load_button.configure(state="disabled")
        self.change_button.configure(state="normal")

    def set_analysis_completed(self):
        """Update UI when analysis completes"""
        self.analysis_complete = True
        self.start_analysis_button.configure(state="disabled")
        self.stop_analysis_button.configure(state="disabled")
        self.load_button.configure(state="disabled")  # Key requirement: disable load after analysis
        self.change_button.configure(state="normal")

    def set_analysis_stopped(self):
        """Update UI when analysis is stopped"""
        self.start_analysis_button.configure(state="normal" if self.current_image_path else "disabled")
        self.stop_analysis_button.configure(state="disabled")

        # Only enable load if no analysis has been completed
        if not self.analysis_complete:
            self.load_button.configure(state="normal")

    def reset_ui(self):
        """Reset UI to initial state"""
        self._reset_to_initial_state()
        self.file_label.configure(text="No image selected")

    def update_model_dropdowns(self, seg_models, clf_models):
        """Update model dropdown options"""
        self.seg_model_dropdown.configure(values=seg_models, state="readonly")
        self.clf_model_dropdown.configure(values=clf_models, state="readonly")

        # Set default selections
        if seg_models:
            self.seg_model_dropdown.set(seg_models[0])
        if clf_models:
            self.clf_model_dropdown.set(clf_models[0])

    def get_selected_models(self):
        """Get currently selected models"""
        return {
            'segmentation': self.seg_model_dropdown.get(),
            'classification': self.clf_model_dropdown.get()
        }

    def update_model_status(self, seg_status, clf_status):
        """Update model loading status"""
        self.segmentation_status.configure(text=seg_status)
        self.classification_status.configure(text=clf_status)

    def update_results(self, results):
        """Update analysis results display"""
        self.prediction_label.configure(
            text=f"Prediction: {results['prediction_label']}",
            text_color="green" if results['prediction_label'] == 'Normal' else "red"
        )
        self.confidence_label.configure(text=f"Confidence: {results['confidence']:.1%}")
        self.prob_normal_label.configure(text=f"Normal: {results['ensemble_probabilities'][0]:.1%}")
        self.prob_abnormal_label.configure(text=f"Abnormal: {results['ensemble_probabilities'][1]:.1%}")

    # Model loading methods
    def _load_models_async(self):
        """Load available models in background thread"""
        def load_models():
            try:
                print("DEBUG: Sidebar - Loading models...")

                # Get available models
                seg_models = self.model_config.get_model_names('segmentation')
                clf_models = self.model_config.get_model_names('classification')

                print(f"DEBUG: Sidebar - Found {len(seg_models)} segmentation models: {seg_models}")
                print(f"DEBUG: Sidebar - Found {len(clf_models)} classification models: {clf_models}")

                # Update UI on main thread
                self.after(0, lambda: self._update_sidebar_dropdowns(seg_models, clf_models))

            except Exception as e:
                print(f"DEBUG: Sidebar - Error loading models: {str(e)}")
                import traceback
                print(f"DEBUG: Sidebar - Traceback: {traceback.format_exc()}")

        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()

    def _update_sidebar_dropdowns(self, seg_models, clf_models):
        """Update sidebar dropdown menus with available models"""
        try:
            print(f"DEBUG: Sidebar - Updating dropdowns with seg={len(seg_models)}, clf={len(clf_models)} models")

            # Update segmentation dropdown
            if seg_models:
                print(f"DEBUG: Sidebar - Segmentation dropdown before: {self.seg_model_dropdown.cget('state')}")
                self.seg_model_dropdown.configure(values=seg_models, state="normal")
                print(f"DEBUG: Sidebar - Segmentation dropdown after: {self.seg_model_dropdown.cget('state')}")

                # Set default
                default_seg = self.model_config.get_default_model('segmentation')
                if default_seg and default_seg.name in seg_models:
                    self.seg_model_dropdown.set(default_seg.name)
                    self.selected_seg_model = default_seg.name
                    print(f"DEBUG: Sidebar - Set default segmentation: {default_seg.name}")
                elif seg_models:
                    self.seg_model_dropdown.set(seg_models[0])
                    self.selected_seg_model = seg_models[0]
                    print(f"DEBUG: Sidebar - Set first segmentation: {seg_models[0]}")
            else:
                self.seg_model_dropdown.configure(values=["No models found"], state="disabled")

            # Update classification dropdown
            if clf_models:
                print(f"DEBUG: Sidebar - Classification dropdown before: {self.clf_model_dropdown.cget('state')}")
                self.clf_model_dropdown.configure(values=clf_models, state="normal")
                print(f"DEBUG: Sidebar - Classification dropdown after: {self.clf_model_dropdown.cget('state')}")

                # Set default
                default_clf = self.model_config.get_default_model('classification')
                if default_clf and default_clf.name in clf_models:
                    self.clf_model_dropdown.set(default_clf.name)
                    self.selected_clf_model = default_clf.name
                    print(f"DEBUG: Sidebar - Set default classification: {default_clf.name}")
                elif clf_models:
                    self.clf_model_dropdown.set(clf_models[0])
                    self.selected_clf_model = clf_models[0]
                    print(f"DEBUG: Sidebar - Set first classification: {clf_models[0]}")
            else:
                self.clf_model_dropdown.configure(values=["No models found"], state="disabled")

            # Update status
            if seg_models and clf_models:
                self.segmentation_status.configure(text=f"Segmentation: {len(seg_models)} models found")
                self.classification_status.configure(text=f"Classification: {len(clf_models)} models found")
                print(f"DEBUG: Sidebar - Successfully updated both dropdowns")
            else:
                self.segmentation_status.configure(text="Segmentation: No models found")
                self.classification_status.configure(text="Classification: No models found")

        except Exception as e:
            print(f"DEBUG: Sidebar - Error updating dropdowns: {str(e)}")
            import traceback
            print(f"DEBUG: Sidebar - Traceback: {traceback.format_exc()}")

    def _on_seg_model_change(self, value):
        """Handle segmentation model selection change"""
        print(f"DEBUG: Sidebar - Segmentation model changed to: {value}")
        self.selected_seg_model = value
        self.segmentation_status.configure(text=f"Segmentation: {value}")

    def _on_clf_model_change(self, value):
        """Handle classification model selection change"""
        print(f"DEBUG: Sidebar - Classification model changed to: {value}")
        self.selected_clf_model = value
        self.classification_status.configure(text=f"Classification: {value}")

    def get_selected_model_info(self):
        """Get information about selected models"""
        return {
            'segmentation': self.selected_seg_model,
            'classification': self.selected_clf_model,
            'seg_info': self.model_config.get_model_by_name('segmentation', self.selected_seg_model) if self.selected_seg_model else None,
            'clf_info': self.model_config.get_model_by_name('classification', self.selected_clf_model) if self.selected_clf_model else None
        }