"""
Direct Dropdown Functionality Test for CXR v3
==============================================

Test to verify ComboBox dropdown functionality directly.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from model_management_v3.model_config import ModelConfig

class DropdownTest(ctk.CTk):
    """Test application for dropdown functionality"""

    def __init__(self):
        super().__init__()

        self.title("Dropdown Functionality Test")
        self.geometry("600x400")

        # Initialize model config
        self.model_config = ModelConfig()

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        """Setup test UI"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Dropdown Functionality Test",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Segmentation model selection
        seg_label = ctk.CTkLabel(main_frame, text="Segmentation Model:")
        seg_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.seg_dropdown = ctk.CTkComboBox(
            main_frame,
            values=["Loading models..."],
            state="disabled",
            command=self.on_seg_model_change
        )
        self.seg_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Classification model selection
        clf_label = ctk.CTkLabel(main_frame, text="Classification Model:")
        clf_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.clf_dropdown = ctk.CTkComboBox(
            main_frame,
            values=["Loading models..."],
            state="disabled",
            command=self.on_clf_model_change
        )
        self.clf_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Test buttons
        test_frame = ctk.CTkFrame(main_frame)
        test_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky="ew")

        test_button1 = ctk.CTkButton(
            test_frame,
            text="Force Enable Dropdowns",
            command=self.force_enable_dropdowns
        )
        test_button1.grid(row=0, column=0, padx=10, pady=10)

        test_button2 = ctk.CTkButton(
            test_frame,
            text="Check Dropdown States",
            command=self.check_dropdown_states
        )
        test_button2.grid(row=0, column=1, padx=10, pady=10)

        test_button3 = ctk.CTkButton(
            test_frame,
            text="Manually Set Values",
            command=self.manually_set_values
        )
        test_button3.grid(row=0, column=2, padx=10, pady=10)

        # Status display
        self.status_text = ctk.CTkTextbox(main_frame, height=150)
        self.status_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def log_status(self, message):
        """Log status message"""
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")
        print(f"TEST: {message}")

    def load_models(self):
        """Load models and update dropdowns"""
        self.log_status("Starting model loading...")

        # Get model names
        seg_models = self.model_config.get_model_names('segmentation')
        clf_models = self.model_config.get_model_names('classification')

        self.log_status(f"Found {len(seg_models)} segmentation models")
        self.log_status(f"Found {len(clf_models)} classification models")

        # Update dropdowns
        if seg_models:
            self.seg_dropdown.configure(values=seg_models)
            self.seg_dropdown.configure(state="readonly")
            self.seg_dropdown.set(seg_models[0])
            self.log_status(f"Updated segmentation dropdown: {seg_models}")
        else:
            self.log_status("No segmentation models found!")

        if clf_models:
            self.clf_dropdown.configure(values=clf_models)
            self.clf_dropdown.configure(state="readonly")
            self.clf_dropdown.set(clf_models[0])
            self.log_status(f"Updated classification dropdown: {clf_models}")
        else:
            self.log_status("No classification models found!")

    def on_seg_model_change(self, value):
        """Handle segmentation model change"""
        self.log_status(f"Segmentation model changed to: {value}")
        messagebox.showinfo("Success", f"Segmentation model changed to: {value}")

    def on_clf_model_change(self, value):
        """Handle classification model change"""
        self.log_status(f"Classification model changed to: {value}")
        messagebox.showinfo("Success", f"Classification model changed to: {value}")

    def force_enable_dropdowns(self):
        """Force enable dropdowns"""
        self.log_status("Forcing dropdown enable...")
        self.seg_dropdown.configure(state="normal")
        self.clf_dropdown.configure(state="normal")
        self.log_status("Dropdowns forced to normal state")

    def check_dropdown_states(self):
        """Check current dropdown states"""
        try:
            seg_state = self.seg_dropdown.cget("state")
            clf_state = self.clf_dropdown.cget("state")
            seg_values = self.seg_dropdown.cget("values")
            clf_values = self.clf_dropdown.cget("values")

            self.log_status(f"Segmentation dropdown state: {seg_state}")
            self.log_status(f"Classification dropdown state: {clf_state}")
            self.log_status(f"Segmentation values: {seg_values}")
            self.log_status(f"Classification values: {clf_values}")

            # Try to get current selections
            try:
                seg_current = self.seg_dropdown.get()
                clf_current = self.clf_dropdown.get()
                self.log_status(f"Current segmentation selection: {seg_current}")
                self.log_status(f"Current classification selection: {clf_current}")
            except Exception as e:
                self.log_status(f"Error getting current selections: {e}")

        except Exception as e:
            self.log_status(f"Error checking states: {e}")

    def manually_set_values(self):
        """Manually set dropdown values for testing"""
        test_seg_models = ["Test Seg Model 1", "Test Seg Model 2"]
        test_clf_models = ["Test Clf Model 1", "Test Clf Model 2"]

        self.log_status("Setting test values manually...")

        self.seg_dropdown.configure(values=test_seg_models, state="readonly")
        self.clf_dropdown.configure(values=test_clf_models, state="readonly")

        self.seg_dropdown.set(test_seg_models[0])
        self.clf_dropdown.set(test_clf_models[0])

        self.log_status("Test values set - try clicking dropdowns now!")

def main():
    """Run the dropdown test"""
    print("Starting Dropdown Functionality Test...")

    app = DropdownTest()
    app.mainloop()

if __name__ == "__main__":
    main()