"""
Model Selector Widget for CXR Analysis v3
=========================================

Dropdown widget for model selection with validation and status updates.
"""

import customtkinter as ctk
import threading
from typing import Dict, List, Callable, Optional, Any
from .model_loader import ModelLoader
from .model_config import ModelConfig


class ModelSelector(ctk.CTkFrame):
    """Enhanced model selector with dropdown and status display"""

    def __init__(self, parent, callback: Optional[Callable] = None, **kwargs):
        super().__init__(parent, fg_color=("gray88", "gray19"), **kwargs)

        self.parent = parent
        self.callback = callback  # Callback function when models are loaded
        self.model_loader = ModelLoader()

        # Current selections
        self.selected_seg_model = None
        self.selected_clf_model = None
        self.loaded_seg_model = None
        self.loaded_clf_model = None
        self.loaded_gradcam = None

        # Loading states
        self.loading_in_progress = False
        self.loading_thread = None

        # Setup UI
        self._setup_ui()

        # Initialize model scanning with a small delay to ensure UI is ready
        self.after(500, self._scan_models_async)

    def _setup_ui(self):
        """Setup the model selector UI"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="Model Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w")

        # Segmentation model section
        self._setup_segmentation_selector()

        # Classification model section
        self._setup_classification_selector()

        # Action buttons
        self._setup_action_buttons()

        # Status section
        self._setup_status_section()

        # Model info section
        self._setup_info_section()

    def _setup_segmentation_selector(self):
        """Setup segmentation model selector"""
        self.seg_label = ctk.CTkLabel(
            self,
            text="Segmentation Model:",
            font=ctk.CTkFont(size=12)
        )
        self.seg_label.grid(row=1, column=0, padx=15, pady=(5, 2), sticky="w")

        self.seg_dropdown = ctk.CTkComboBox(
            self,
            values=["Scanning models..."],
            state="disabled",
            width=320,
            command=self._on_seg_model_change,
            fg_color=("gray85", "gray22"),
            border_color=("gray70", "gray28")
        )
        self.seg_dropdown.grid(row=2, column=0, padx=15, pady=(0, 8), sticky="ew")

        # Segmentation status
        self.seg_status_label = ctk.CTkLabel(
            self,
            text="Status: Not loaded",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.seg_status_label.grid(row=3, column=0, padx=15, pady=(0, 5), sticky="w")

    def _setup_classification_selector(self):
        """Setup classification model selector"""
        self.clf_label = ctk.CTkLabel(
            self,
            text="Classification Model:",
            font=ctk.CTkFont(size=12)
        )
        self.clf_label.grid(row=4, column=0, padx=15, pady=(5, 2), sticky="w")

        self.clf_dropdown = ctk.CTkComboBox(
            self,
            values=["Scanning models..."],
            state="disabled",
            width=320,
            command=self._on_clf_model_change,
            fg_color=("gray85", "gray22"),
            border_color=("gray70", "gray28")
        )
        self.clf_dropdown.grid(row=5, column=0, padx=15, pady=(0, 8), sticky="ew")

        # Classification status
        self.clf_status_label = ctk.CTkLabel(
            self,
            text="Status: Not loaded",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.clf_status_label.grid(row=6, column=0, padx=15, pady=(0, 5), sticky="w")

    def _setup_action_buttons(self):
        """Setup action buttons"""
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=7, column=0, padx=15, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        # Load models button
        self.load_button = ctk.CTkButton(
            self.button_frame,
            text="🔄 Load Models",
            command=self._load_selected_models,
            state="disabled",
            height=35
        )
        self.load_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")

        # Refresh button
        self.refresh_button = ctk.CTkButton(
            self.button_frame,
            text="↻ Refresh",
            command=self._refresh_models,
            height=35,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90")
        )
        self.refresh_button.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")

    def _setup_status_section(self):
        """Setup overall status section"""
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=8, column=0, padx=15, pady=(5, 10), sticky="ew")

        self.status_title = ctk.CTkLabel(
            self.status_frame,
            text="Loading Status",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready to load models",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.status_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")

    def _setup_info_section(self):
        """Setup model information section"""
        self.info_frame = ctk.CTkFrame(self)
        self.info_frame.grid(row=9, column=0, padx=15, pady=(0, 15), sticky="ew")

        self.info_title = ctk.CTkLabel(
            self.info_frame,
            text="Model Information",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.info_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.info_text = ctk.CTkTextbox(
            self.info_frame,
            height=80,
            font=ctk.CTkFont(size=9)
        )
        self.info_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

    def _scan_models_async(self):
        """Scan for available models in background thread"""
        def scan_models():
            try:
                self._update_status("Scanning for available models...")

                # Get available models
                seg_models = self.model_loader.config.get_model_names('segmentation')
                clf_models = self.model_loader.config.get_model_names('classification')

                print(f"DEBUG: Found {len(seg_models)} segmentation models: {seg_models}")
                print(f"DEBUG: Found {len(clf_models)} classification models: {clf_models}")

                # Update UI on main thread
                self.after(0, lambda: self._update_model_dropdowns(seg_models, clf_models))

            except Exception as e:
                print(f"DEBUG: Error scanning models: {str(e)}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                self.after(0, lambda: self._update_status(f"Error scanning models: {str(e)}"))

        thread = threading.Thread(target=scan_models, daemon=True)
        thread.start()

    def _update_model_dropdowns(self, seg_models: List[str], clf_models: List[str]):
        """Update dropdown menus with available models"""
        try:
            # Update segmentation dropdown
            if seg_models:
                print(f"DEBUG: Updating segmentation dropdown with {seg_models}")
                print(f"DEBUG: Segmentation dropdown state before update: {self.seg_dropdown.cget('state')}")
                # Use "normal" state instead of "readonly" to ensure dropdown is clickable
                self.seg_dropdown.configure(values=seg_models, state="normal")
                print(f"DEBUG: Segmentation dropdown state after update: {self.seg_dropdown.cget('state')}")

                default_seg = self.model_loader.config.get_default_model('segmentation')
                if default_seg and default_seg.name in seg_models:
                    self.seg_dropdown.set(default_seg.name)
                    self.selected_seg_model = default_seg.name
                    print(f"DEBUG: Set default segmentation model: {default_seg.name}")
                elif seg_models:
                    # If no default, set first available
                    self.seg_dropdown.set(seg_models[0])
                    self.selected_seg_model = seg_models[0]
                    print(f"DEBUG: Set first segmentation model: {seg_models[0]}")
            else:
                print("DEBUG: No segmentation models found")
                self.seg_dropdown.configure(values=["No models found"], state="disabled")

            # Update classification dropdown
            if clf_models:
                print(f"DEBUG: Updating classification dropdown with {clf_models}")
                print(f"DEBUG: Classification dropdown state before update: {self.clf_dropdown.cget('state')}")
                # Use "normal" state instead of "readonly" to ensure dropdown is clickable
                self.clf_dropdown.configure(values=clf_models, state="normal")
                print(f"DEBUG: Classification dropdown state after update: {self.clf_dropdown.cget('state')}")

                default_clf = self.model_loader.config.get_default_model('classification')
                if default_clf and default_clf.name in clf_models:
                    self.clf_dropdown.set(default_clf.name)
                    self.selected_clf_model = default_clf.name
                    print(f"DEBUG: Set default classification model: {default_clf.name}")
                elif clf_models:
                    # If no default, set first available
                    self.clf_dropdown.set(clf_models[0])
                    self.selected_clf_model = clf_models[0]
                    print(f"DEBUG: Set first classification model: {clf_models[0]}")
            else:
                print("DEBUG: No classification models found")
                self.clf_dropdown.configure(values=["No models found"], state="disabled")

            # Enable load button if both models are available and selected
            if seg_models and clf_models and self.selected_seg_model and self.selected_clf_model:
                self.load_button.configure(state="normal")
                self._update_status(f"Found {len(seg_models)} segmentation and {len(clf_models)} classification models")
                self._update_model_info()
                print(f"DEBUG: Load button enabled. Selected models: seg={self.selected_seg_model}, clf={self.selected_clf_model}")

                # Force a final check on dropdown states
                self.after(100, self._force_dropdown_enable)
            else:
                self._update_status("No valid model pairs found")
                print(f"DEBUG: Load button not enabled. seg_models={len(seg_models) if seg_models else 0}, clf_models={len(clf_models) if clf_models else 0}")
                print(f"DEBUG: Selected models: seg={self.selected_seg_model}, clf={self.selected_clf_model}")

        except Exception as e:
            self._update_status(f"Error updating dropdowns: {str(e)}")
            print(f"DEBUG: Exception in _update_model_dropdowns: {str(e)}")

    def _force_dropdown_enable(self):
        """Force enable dropdowns as a final check"""
        try:
            print("DEBUG: Forcing dropdown enable as final check...")

            # Check current states
            seg_state = self.seg_dropdown.cget("state")
            clf_state = self.clf_dropdown.cget("state")
            seg_values = self.seg_dropdown.cget("values")
            clf_values = self.clf_dropdown.cget("values")

            print(f"DEBUG: Current states - Seg: {seg_state}, Clf: {clf_state}")
            print(f"DEBUG: Current values - Seg: {seg_values}, Clf: {clf_values}")

            # Force enable if they have values but are not in normal state
            if seg_values and len(seg_values) > 0 and seg_values[0] != "No models found":
                if seg_state != "normal":
                    self.seg_dropdown.configure(state="normal")
                    print("DEBUG: Forced segmentation dropdown to normal state")

            if clf_values and len(clf_values) > 0 and clf_values[0] != "No models found":
                if clf_state != "normal":
                    self.clf_dropdown.configure(state="normal")
                    print("DEBUG: Forced classification dropdown to normal state")

            # Final state verification
            final_seg_state = self.seg_dropdown.cget("state")
            final_clf_state = self.clf_dropdown.cget("state")
            print(f"DEBUG: Final states - Seg: {final_seg_state}, Clf: {final_clf_state}")

        except Exception as e:
            print(f"DEBUG: Error in force dropdown enable: {e}")

    def _on_seg_model_change(self, value):
        """Handle segmentation model selection change"""
        print(f"DEBUG: Segmentation model changed to: {value}")
        # Validate the selection is from available models
        if hasattr(self.seg_dropdown, 'cget'):
            available_values = self.seg_dropdown.cget("values")
            if value not in available_values:
                print(f"DEBUG: Invalid segmentation model selection: {value}, available: {available_values}")
                return

        self.selected_seg_model = value
        self.loaded_seg_model = None  # Clear loaded model
        self.seg_status_label.configure(text="Status: Not loaded", text_color="gray")
        self._update_model_info()
        print(f"DEBUG: Segmentation model selection confirmed: {value}")

    def _on_clf_model_change(self, value):
        """Handle classification model selection change"""
        print(f"DEBUG: Classification model changed to: {value}")
        # Validate the selection is from available models
        if hasattr(self.clf_dropdown, 'cget'):
            available_values = self.clf_dropdown.cget("values")
            if value not in available_values:
                print(f"DEBUG: Invalid classification model selection: {value}, available: {available_values}")
                return

        self.selected_clf_model = value
        self.loaded_clf_model = None  # Clear loaded model
        self.loaded_gradcam = None
        self.clf_status_label.configure(text="Status: Not loaded", text_color="gray")
        self._update_model_info()
        print(f"DEBUG: Classification model selection confirmed: {value}")

    def _load_selected_models(self):
        """Load the selected models in background thread"""
        if self.loading_in_progress:
            return

        if not self.selected_seg_model or not self.selected_clf_model:
            self._update_status("Please select both models")
            return

        self.loading_in_progress = True
        self.load_button.configure(state="disabled", text="🔄 Loading...")

        def load_models():
            try:
                # Validate model pair first
                validation = self.model_loader.validate_model_pair(
                    self.selected_seg_model,
                    self.selected_clf_model
                )

                if not validation['valid']:
                    self.after(0, lambda: self._handle_loading_error(validation['error']))
                    return

                # Load segmentation model
                self.after(0, lambda: self._update_status("Loading segmentation model..."))
                seg_model, seg_status = self.model_loader.load_segmentation_model(self.selected_seg_model)

                if seg_model is None:
                    self.after(0, lambda: self._handle_loading_error(f"Failed to load segmentation model: {seg_status}"))
                    return

                self.loaded_seg_model = seg_model
                self.after(0, lambda: self.seg_status_label.configure(text=seg_status, text_color="green"))

                # Load classification model
                self.after(0, lambda: self._update_status("Loading classification model..."))
                clf_model, gradcam, clf_status = self.model_loader.load_classification_model(self.selected_clf_model)

                if clf_model is None:
                    self.after(0, lambda: self._handle_loading_error(f"Failed to load classification model: {clf_status}"))
                    return

                self.loaded_clf_model = clf_model
                self.loaded_gradcam = gradcam
                self.after(0, lambda: self.clf_status_label.configure(text=clf_status, text_color="green"))

                # Success
                self.after(0, lambda: self._handle_loading_success(validation.get('warnings', [])))

            except Exception as e:
                self.after(0, lambda: self._handle_loading_error(f"Loading failed: {str(e)}"))

        self.loading_thread = threading.Thread(target=load_models, daemon=True)
        self.loading_thread.start()

    def _handle_loading_success(self, warnings: List[str]):
        """Handle successful model loading"""
        self.loading_in_progress = False
        self.load_button.configure(state="normal", text="✅ Models Loaded")

        status_msg = "Models loaded successfully"
        if warnings:
            status_msg += f" (Warnings: {len(warnings)})"

        self._update_status(status_msg)

        # Update model info
        self._update_model_info()

        # Notify parent if callback is set
        if self.callback:
            self.callback({
                'segmentation_model': self.loaded_seg_model,
                'classification_model': self.loaded_clf_model,
                'gradcam': self.loaded_gradcam,
                'seg_model_name': self.selected_seg_model,
                'clf_model_name': self.selected_clf_model
            })

    def _handle_loading_error(self, error_msg: str):
        """Handle model loading error"""
        self.loading_in_progress = False
        self.load_button.configure(state="normal", text="🔄 Load Models")
        self._update_status(f"Error: {error_msg}")

        # Reset status labels
        if self.loaded_seg_model is None:
            self.seg_status_label.configure(text="Status: Failed to load", text_color="red")
        if self.loaded_clf_model is None:
            self.clf_status_label.configure(text="Status: Failed to load", text_color="red")

    def _refresh_models(self):
        """Refresh available models"""
        self.refresh_button.configure(state="disabled", text="⟳ Refreshing...")

        def refresh():
            try:
                self.model_loader.refresh_available_models()
                self.after(0, lambda: self._scan_models_async())
                self.after(1000, lambda: self.refresh_button.configure(state="normal", text="↻ Refresh"))
            except Exception as e:
                self.after(0, lambda: self._update_status(f"Refresh failed: {str(e)}"))
                self.after(1000, lambda: self.refresh_button.configure(state="normal", text="↻ Refresh"))

        thread = threading.Thread(target=refresh, daemon=True)
        thread.start()

    def _update_status(self, message: str):
        """Update status message"""
        self.status_label.configure(text=message)

    def _update_model_info(self):
        """Update model information display"""
        try:
            info_text = ""

            if self.selected_seg_model:
                seg_info = self.model_loader.get_model_info('segmentation', self.selected_seg_model)
                if seg_info:
                    info_text += f"Segmentation: {seg_info.architecture}\n"
                    info_text += f"Input: {seg_info.input_size}\n"

            if self.selected_clf_model:
                clf_info = self.model_loader.get_model_info('classification', self.selected_clf_model)
                if clf_info:
                    info_text += f"Classification: {clf_info.architecture}\n"
                    info_text += f"Input: {clf_info.input_size}\n"
                    info_text += f"Classes: {clf_info.output_classes}\n"

            if not info_text:
                info_text = "Select models to see information"

            self.info_text.delete("1.0", "end")
            self.info_text.insert("1.0", info_text)

        except Exception as e:
            self.info_text.delete("1.0", "end")
            self.info_text.insert("1.0", f"Error displaying info: {str(e)}")

    def get_loaded_models(self) -> Dict[str, Any]:
        """Get currently loaded models"""
        return {
            'segmentation_model': self.loaded_seg_model,
            'classification_model': self.loaded_clf_model,
            'gradcam': self.loaded_gradcam,
            'seg_model_name': self.selected_seg_model if self.loaded_seg_model else None,
            'clf_model_name': self.selected_clf_model if self.loaded_clf_model else None
        }

    def are_models_loaded(self) -> bool:
        """Check if both models are loaded"""
        return (self.loaded_seg_model is not None and
                self.loaded_clf_model is not None)

    def unload_models(self):
        """Unload all models"""
        if self.selected_seg_model:
            self.model_loader.unload_model('segmentation', self.selected_seg_model)
        if self.selected_clf_model:
            self.model_loader.unload_model('classification', self.selected_clf_model)

        self.loaded_seg_model = None
        self.loaded_clf_model = None
        self.loaded_gradcam = None

        self.seg_status_label.configure(text="Status: Not loaded", text_color="gray")
        self.clf_status_label.configure(text="Status: Not loaded", text_color="gray")
        self.load_button.configure(text="🔄 Load Models")

        self._update_status("Models unloaded")

    def set_loading_enabled(self, enabled: bool):
        """Enable or disable model loading"""
        state = "normal" if enabled else "disabled"
        self.load_button.configure(state=state)
        self.seg_dropdown.configure(state="normal" if enabled else "disabled")
        self.clf_dropdown.configure(state="normal" if enabled else "disabled")