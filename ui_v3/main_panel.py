"""
Main Panel Component for CXR Analysis v3
========================================

Enhanced main panel with improved tabbed interface and better organization.
"""

import customtkinter as ctk
import numpy as np
import cv2
from PIL import Image, ImageTk


class MainPanel(ctk.CTkTabview):
    """Enhanced main panel with tabbed interface"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=1000,
                        fg_color=("gray92", "gray14"), **kwargs)

        self.parent = parent
        self.current_results = None

        # Create tabs
        self._create_tabs()
        self._setup_tabs()

    def _create_tabs(self):
        """Create main tabs"""
        self.add("Image Preview")
        self.add("Processing Steps")
        self.add("Final Results")
        self.add("Detailed Analysis")

    def _setup_tabs(self):
        """Setup each tab"""
        self._setup_preview_tab()
        self._setup_steps_tab()
        self._setup_results_tab()
        self._setup_analysis_tab()

    def _setup_preview_tab(self):
        """Setup image preview tab"""
        # This tab will contain the ImagePreviewFrame
        from .image_preview import ImagePreviewFrame

        self.preview_frame = ImagePreviewFrame(self.tab("Image Preview"))
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def _setup_steps_tab(self):
        """Setup processing steps tab"""
        self.steps_main_frame = ctk.CTkFrame(self.tab("Processing Steps"))
        self.steps_main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid for steps
        self.steps_main_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.steps_main_frame.grid_rowconfigure((0, 1), weight=1)

        # Create step frames
        self._create_processing_steps()

    def _create_processing_steps(self):
        """Create individual processing step frames"""
        step_configs = [
            ("1. Original Image", 0, 0, "step1"),
            ("2. Lung Segmentation", 0, 1, "step2"),
            ("3. Soft Masking", 0, 2, "step3"),
            ("4. Attention Masking", 1, 0, "step4"),
            ("5. GradCAM Heatmap", 1, 1, "step5"),
            ("6. Final Overlay", 1, 2, "step6")
        ]

        for title, row, col, attr_prefix in step_configs:
            # Create step container
            step_frame = ctk.CTkFrame(self.steps_main_frame)
            step_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            # Step title
            step_title = ctk.CTkLabel(
                step_frame,
                text=title,
                font=ctk.CTkFont(size=12, weight="bold")
            )
            step_title.pack(pady=(10, 5))

            # Image placeholder
            step_image = ctk.CTkLabel(
                step_frame,
                text="Processing...",
                width=180,
                height=180
            )
            step_image.pack(expand=True, pady=(5, 10))

            # Store references
            setattr(self, f"{attr_prefix}_frame", step_frame)
            setattr(self, f"{attr_prefix}_title", step_title)
            setattr(self, f"{attr_prefix}_image", step_image)

    def _setup_results_tab(self):
        """Setup final results tab with overlay visualization"""
        self.results_main_frame = ctk.CTkFrame(self.tab("Final Results"))
        self.results_main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid
        self.results_main_frame.grid_columnconfigure(0, weight=1)
        self.results_main_frame.grid_rowconfigure(1, weight=1)

        # Results header
        self.results_header = ctk.CTkFrame(self.results_main_frame, height=60)
        self.results_header.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.results_header.grid_columnconfigure(1, weight=1)

        # Title and controls
        self.results_title = ctk.CTkLabel(
            self.results_header,
            text="Final Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.results_title.grid(row=0, column=0, padx=15, pady=15, sticky="w")

        # Export button
        self.export_button = ctk.CTkButton(
            self.results_header,
            text="📊 Export Results",
            command=self._export_results,
            state="disabled"
        )
        self.export_button.grid(row=0, column=1, padx=15, pady=15, sticky="e")

        # Main results display
        self.results_display_frame = ctk.CTkFrame(self.results_main_frame)
        self.results_display_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.results_display_frame.grid_columnconfigure(0, weight=1)
        self.results_display_frame.grid_rowconfigure(0, weight=1)

        # Large overlay image
        self.overlay_image_label = ctk.CTkLabel(
            self.results_display_frame,
            text="Load an image and run analysis to see results",
            font=ctk.CTkFont(size=16)
        )
        self.overlay_image_label.pack(expand=True)

    def _setup_analysis_tab(self):
        """Setup detailed analysis tab"""
        self.analysis_main_frame = ctk.CTkFrame(self.tab("Detailed Analysis"))
        self.analysis_main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid
        self.analysis_main_frame.grid_columnconfigure(0, weight=1)
        self.analysis_main_frame.grid_rowconfigure(1, weight=1)

        # Analysis header
        self.analysis_header = ctk.CTkLabel(
            self.analysis_main_frame,
            text="Detailed Analysis Report",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.analysis_header.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w")

        # Analysis text display
        self.analysis_text = ctk.CTkTextbox(
            self.analysis_main_frame,
            font=ctk.CTkFont(family="Courier", size=12),
            wrap="word"
        )
        self.analysis_text.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

    def load_image_preview(self, image_path):
        """Load image preview in the preview tab"""
        return self.preview_frame.load_image_preview(image_path)

    def show_processing_state(self):
        """Show processing state across tabs"""
        # Update preview tab
        self.preview_frame.show_processing_state()

        # Update processing steps
        for i in range(1, 7):
            step_image = getattr(self, f"step{i}_image")
            step_image.configure(text="⏳ Processing...", image="")

        # Update results tab
        self.overlay_image_label.configure(text="🔄 Analysis in progress...", image="")

        # Update analysis tab
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "Analysis in progress...\n\nPlease wait while the enhanced pipeline processes your image.")

    def show_analysis_state(self):
        """Show analysis in progress state"""
        self.preview_frame.show_analysis_state()

        # Update step indicators
        for i in range(1, 7):
            step_image = getattr(self, f"step{i}_image")
            step_image.configure(text="🔬 Analyzing...", image="")

    def update_processing_steps(self, results):
        """Update processing steps with results"""
        try:
            img_size = 180

            # Define step data
            step_data = [
                ('step1', results.get('original_resized')),
                ('step2', results.get('segmentation_mask')),
                ('step3', results.get('segmented_soft')),
                ('step4', results.get('segmented_attention')),
                ('step5', self._create_gradcam_visualization(results.get('gradcam'), img_size)),
                ('step6', self._create_final_overlay(results, img_size))
            ]

            for step_name, image_data in step_data:
                if image_data is not None:
                    self._update_step_image(step_name, image_data, img_size)

        except Exception as e:
            print(f"Error updating processing steps: {e}")

    def _update_step_image(self, step_name, image_data, size):
        """Update individual step image"""
        try:
            step_image = getattr(self, f"{step_name}_image")

            if isinstance(image_data, np.ndarray):
                # Handle different image types
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    # RGB image
                    pil_image = Image.fromarray(image_data).resize((size, size))
                else:
                    # Grayscale image
                    if image_data.dtype != np.uint8:
                        # Normalize if needed
                        if image_data.max() <= 1.0:
                            image_data = (image_data * 255).astype(np.uint8)
                        else:
                            image_data = image_data.astype(np.uint8)
                    pil_image = Image.fromarray(image_data).resize((size, size))

                ctk_image = ctk.CTkImage(
                    light_image=pil_image,
                    dark_image=pil_image,
                    size=(size, size)
                )
                step_image.configure(image=ctk_image, text="")

        except Exception as e:
            print(f"Error updating step image {step_name}: {e}")
            step_image.configure(text="Error", image="")

    def _create_gradcam_visualization(self, cam, size):
        """Create colored GradCAM visualization"""
        if cam is None:
            return None

        try:
            cam_resized = cv2.resize(cam, (size, size))
            cam_normalized = np.clip(cam_resized, 0, 1)
            cam_colored = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
            return cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error creating GradCAM visualization: {e}")
            return None

    def _create_final_overlay(self, results, size):
        """Create final overlay visualization"""
        try:
            if not all(key in results for key in ['original_resized', 'gradcam', 'segmentation_mask']):
                return None

            # Resize components
            original = cv2.resize(results['original_resized'], (size, size))
            cam = cv2.resize(results['gradcam'], (size, size))
            mask = cv2.resize(results['segmentation_mask'], (size, size))

            # Create colored overlays
            heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Segmentation overlay (green for lungs)
            seg_overlay = np.zeros((size, size, 3), dtype=np.uint8)
            seg_overlay[:, :, 1] = mask * 100

            # Combine overlays
            base_rgb = np.stack([original] * 3, axis=-1)
            final_overlay = (base_rgb * 0.5 + seg_overlay * 0.2 + heatmap_rgb * 0.3).astype(np.uint8)

            return final_overlay

        except Exception as e:
            print(f"Error creating final overlay: {e}")
            return None

    def update_final_results(self, results):
        """Update final results display with large overlay"""
        try:
            img_size = 512

            # Create large overlay
            overlay = self._create_final_overlay(results, img_size)
            if overlay is not None:
                overlay_pil = Image.fromarray(overlay)
                overlay_tk = ctk.CTkImage(
                    light_image=overlay_pil,
                    dark_image=overlay_pil,
                    size=(img_size, img_size)
                )
                self.overlay_image_label.configure(image=overlay_tk, text="")

            # Enable export button
            self.export_button.configure(state="normal")

        except Exception as e:
            print(f"Error updating final results: {e}")
            self.overlay_image_label.configure(text="Error displaying results", image="")

    def update_detailed_analysis(self, results):
        """Update detailed analysis text"""
        try:
            detailed_text = self._generate_detailed_report(results)
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("1.0", detailed_text)

        except Exception as e:
            print(f"Error updating detailed analysis: {e}")
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("1.0", f"Error generating detailed analysis: {str(e)}")

    def _generate_detailed_report(self, results):
        """Generate comprehensive analysis report"""
        report = "ENHANCED CXR ANALYSIS REPORT - MULTI-MODEL PIPELINE v3\n"
        report += "=" * 70 + "\n\n"

        # Executive Summary
        report += "EXECUTIVE SUMMARY:\n"
        report += "-" * 20 + "\n"
        report += f"Final Prediction: {results['prediction_label']}\n"
        report += f"Confidence Level: {results['confidence']:.1%}\n"
        report += f"Recommendation: {self._get_clinical_recommendation(results)}\n\n"

        # Ensemble Analysis
        report += "ENSEMBLE ANALYSIS:\n"
        report += "-" * 20 + "\n"
        report += f"Normal Probability:   {results['ensemble_probabilities'][0]:.3f} ({results['ensemble_probabilities'][0]:.1%})\n"
        report += f"Abnormal Probability: {results['ensemble_probabilities'][1]:.3f} ({results['ensemble_probabilities'][1]:.1%})\n\n"

        # Individual Model Contributions
        report += "INDIVIDUAL MODEL CONTRIBUTIONS:\n"
        report += "-" * 35 + "\n"

        for model_type, probs in results['individual_probs'].items():
            weight = results['adaptive_weights'].get(model_type, 0)
            conf = results['individual_confidences'].get(model_type, 0)
            prediction = "Normal" if probs[0] > probs[1] else "Abnormal"

            report += f"{model_type.replace('_', ' ').title()}:\n"
            report += f"  Prediction: {prediction}\n"
            report += f"  Normal: {probs[0]:.3f}, Abnormal: {probs[1]:.3f}\n"
            report += f"  Confidence: {conf:.3f}, Ensemble Weight: {weight:.3f}\n\n"

        # Technical Details
        report += "TECHNICAL SPECIFICATIONS:\n"
        report += "-" * 30 + "\n"
        report += f"Classification Backbone: Multi-Model Ensemble\n"
        report += f"Segmentation Model: ResUNet Architecture\n"
        report += f"Input Resolution: 224x224 (Classification), 256x256 (Segmentation)\n"
        report += f"Preprocessing: Advanced Soft Masking + Attention Mechanisms\n"
        report += f"Visualization: Enhanced GradCAM with Multi-scale Analysis\n\n"

        # Analysis Quality Metrics
        report += "ANALYSIS QUALITY METRICS:\n"
        report += "-" * 30 + "\n"
        segmentation_quality = self._assess_segmentation_quality(results)
        report += f"Segmentation Quality: {segmentation_quality}\n"
        report += f"Model Agreement: {self._calculate_model_agreement(results):.1%}\n"
        report += f"Confidence Reliability: {self._assess_confidence_reliability(results)}\n\n"

        # Clinical Insights
        report += "CLINICAL INSIGHTS:\n"
        report += "-" * 20 + "\n"
        report += self._generate_clinical_insights(results)

        # Disclaimer
        report += "\nDISCLAIMER:\n"
        report += "-" * 15 + "\n"
        report += "This analysis is generated by an AI-assisted diagnostic tool.\n"
        report += "Results should be interpreted by qualified medical professionals.\n"
        report += "This tool is intended to assist, not replace, clinical judgment.\n"

        return report

    def _get_clinical_recommendation(self, results):
        """Get clinical recommendation based on results"""
        confidence = results['confidence']
        label = results['prediction_label']

        if label == 'Normal':
            if confidence > 0.9:
                return "No immediate action required"
            elif confidence > 0.7:
                return "Consider routine follow-up"
            else:
                return "Clinical correlation advised"
        else:
            if confidence > 0.9:
                return "Urgent clinical review recommended"
            elif confidence > 0.7:
                return "Clinical evaluation advised"
            else:
                return "Additional imaging may be beneficial"

    def _assess_segmentation_quality(self, results):
        """Assess quality of lung segmentation"""
        try:
            mask = results.get('segmentation_mask')
            if mask is None:
                return "Unknown"

            # Simple quality metrics
            mask_coverage = np.sum(mask) / (mask.shape[0] * mask.shape[1])

            if mask_coverage > 0.3:
                return "Good"
            elif mask_coverage > 0.15:
                return "Moderate"
            else:
                return "Poor"

        except:
            return "Unknown"

    def _calculate_model_agreement(self, results):
        """Calculate agreement between individual models"""
        try:
            individual_probs = results.get('individual_probs', {})
            if len(individual_probs) < 2:
                return 0.0

            # Calculate variance in predictions
            normal_probs = [probs[0] for probs in individual_probs.values()]
            variance = np.var(normal_probs)

            # Convert variance to agreement score (lower variance = higher agreement)
            agreement = max(0, 1 - (variance * 4))  # Scale factor of 4
            return agreement

        except:
            return 0.0

    def _assess_confidence_reliability(self, results):
        """Assess reliability of confidence score"""
        confidence = results.get('confidence', 0)
        agreement = self._calculate_model_agreement(results)

        if confidence > 0.9 and agreement > 0.8:
            return "Very High"
        elif confidence > 0.8 and agreement > 0.6:
            return "High"
        elif confidence > 0.6 and agreement > 0.4:
            return "Moderate"
        else:
            return "Low"

    def _generate_clinical_insights(self, results):
        """Generate clinical insights based on analysis"""
        insights = ""
        label = results['prediction_label']
        confidence = results['confidence']

        if label == 'Normal':
            if confidence > 0.9:
                insights += "• High confidence normal chest radiograph\n"
                insights += "• No obvious pathological findings detected\n"
                insights += "• Lung fields appear clear\n"
            elif confidence > 0.7:
                insights += "• Likely normal chest radiograph\n"
                insights += "• Minor variations within normal limits\n"
                insights += "• Consider clinical correlation if symptomatic\n"
            else:
                insights += "• Borderline normal classification\n"
                insights += "• Subtle findings may be present\n"
                insights += "• Recommend clinical review and correlation\n"
        else:
            if confidence > 0.9:
                insights += "• High confidence abnormal findings\n"
                insights += "• Significant pathological changes detected\n"
                insights += "• Immediate clinical correlation strongly recommended\n"
            elif confidence > 0.7:
                insights += "• Likely abnormal chest radiograph\n"
                insights += "• Pathological findings present\n"
                insights += "• Clinical evaluation and follow-up advised\n"
            else:
                insights += "• Possible abnormal findings\n"
                insights += "• Subtle pathological changes detected\n"
                insights += "• Additional imaging or clinical assessment recommended\n"

        return insights

    def clear_all_displays(self):
        """Clear all displays and reset to initial state"""
        # Clear preview
        self.preview_frame.reset_to_initial_state()

        # Clear processing steps
        for i in range(1, 7):
            step_image = getattr(self, f"step{i}_image")
            step_image.configure(text="", image="")

        # Clear results
        self.overlay_image_label.configure(text="Load an image and run analysis to see results", image="")
        self.export_button.configure(state="disabled")

        # Clear analysis
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "No analysis available. Load an image and run analysis to see detailed results.")

    def _export_results(self):
        """Export analysis results"""
        if hasattr(self.parent, 'export_results'):
            self.parent.export_results()

    def set_drag_drop_enabled(self, enabled):
        """Enable or disable drag and drop in preview"""
        self.preview_frame.set_drag_drop_enabled(enabled)