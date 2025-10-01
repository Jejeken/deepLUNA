"""
Drag and Drop Component for CXR Analysis v3
===========================================

Enhanced drag and drop functionality for image loading.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import os
import cv2
from PIL import Image, ImageTk
import tkinterdnd2 as tkdnd


class DragDropFrame(ctk.CTkFrame):
    """Drag and drop frame for image loading"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.parent = parent
        self.drop_enabled = True

        # Setup UI
        self._setup_ui()

        # Initialize drag and drop after UI is ready
        self.after(100, self._setup_drag_drop)  # Delay to ensure UI is fully initialized

    def _setup_ui(self):
        """Setup the drag and drop UI"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main drop area
        self.drop_frame = ctk.CTkFrame(
            self,
            fg_color=("gray85", "gray20"),
            border_width=2,
            border_color=("gray70", "gray30"),
            corner_radius=10
        )
        self.drop_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.drop_frame.grid_columnconfigure(0, weight=1)
        self.drop_frame.grid_rowconfigure(0, weight=1)

        # Drop icon and text
        self.drop_icon_label = ctk.CTkLabel(
            self.drop_frame,
            text="📁",
            font=ctk.CTkFont(size=48)
        )
        self.drop_icon_label.grid(row=0, column=0, pady=(20, 5))

        self.drop_text_label = ctk.CTkLabel(
            self.drop_frame,
            text="Drag & Drop X-Ray Image Here\nor click to browse",
            font=ctk.CTkFont(size=14),
            text_color=("gray50", "gray60")
        )
        self.drop_text_label.grid(row=1, column=0, pady=(5, 10))

        self.supported_formats_label = ctk.CTkLabel(
            self.drop_frame,
            text="Supported: PNG, JPG, JPEG, BMP, TIFF",
            font=ctk.CTkFont(size=10),
            text_color=("gray40", "gray70")
        )
        self.supported_formats_label.grid(row=2, column=0, pady=(0, 20))

    def _setup_drag_drop(self):
        """Setup drag and drop functionality with tkinterdnd2"""
        try:
            # Bind click events for file dialog fallback
            widgets_to_bind = [self, self.drop_frame, self.drop_icon_label,
                              self.drop_text_label, self.supported_formats_label]

            successful_bindings = 0
            for widget in widgets_to_bind:
                # Bind click to open file dialog
                widget.bind('<Button-1>', self._on_click)

                # Setup drag and drop with tkinterdnd2
                try:
                    widget.drop_target_register(tkdnd.DND_FILES)
                    widget.dnd_bind('<<Drop>>', self._on_drop)
                    widget.dnd_bind('<<DragEnter>>', self._on_drag_enter)
                    widget.dnd_bind('<<DragLeave>>', self._on_drag_leave)
                    successful_bindings += 1
                except Exception as e:
                    print(f"DEBUG: Failed to setup drag drop for widget {type(widget).__name__}: {e}")

            if successful_bindings > 0:
                print(f"DEBUG: Drag and drop setup completed ({successful_bindings}/{len(widgets_to_bind)} widgets)")
            else:
                print("DEBUG: Drag and drop setup failed - using click-only mode")
                self._setup_click_only_mode()

        except Exception as e:
            print(f"DEBUG: Drag and drop setup completely failed: {e}")
            self._setup_click_only_mode()

    def _setup_click_only_mode(self):
        """Setup click-only mode when drag and drop fails"""
        self.drop_text_label.configure(
            text="Click here to browse for X-Ray Image\n(Drag & Drop not available)",
            text_color=("gray60", "gray50")
        )

    def _on_drop(self, event):
        """Handle file drop event with tkinterdnd2"""
        if not self.drop_enabled:
            return

        try:
            files = self.tk.splitlist(event.data)
            if files:
                file_path = files[0]
                self._handle_dropped_file(file_path)
        except Exception as e:
            print(f"Drop handling error: {e}")
            messagebox.showerror("Drop Error", f"Failed to handle dropped file: {str(e)}")
        finally:
            self._reset_drop_appearance()

    def _on_drag_enter(self, event):
        """Handle drag enter event"""
        if not self.drop_enabled:
            return

        self.drop_frame.configure(
            border_color=("#3B82F6", "#3B82F6"),
            fg_color=("gray90", "gray20")
        )
        self.drop_text_label.configure(text="Drop image here...")

    def _on_drag_leave(self, event):
        """Handle drag leave event"""
        self._reset_drop_appearance()


    def _on_click(self, event):
        """Handle click event as fallback"""
        if hasattr(self.parent, 'load_image'):
            self.parent.load_image()

    def _handle_dropped_file(self, file_path):
        """Process dropped file"""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File not found")
            return

        if not self._is_valid_image_file(file_path):
            messagebox.showerror("Error",
                               "Invalid image format. Supported: PNG, JPG, JPEG, BMP, TIFF")
            return

        if not self._validate_image_file(file_path):
            return

        # Notify parent about the dropped file
        if hasattr(self.parent, 'handle_image_drop'):
            self.parent.handle_image_drop(file_path)
        elif hasattr(self.parent, 'load_specific_image'):
            self.parent.load_specific_image(file_path)

    def _is_valid_image_file(self, file_path):
        """Check if file is a valid image format"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in valid_extensions

    def _validate_image_file(self, file_path):
        """Validate image file before processing"""
        try:
            # Try to read with OpenCV
            test_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if test_img is None:
                messagebox.showerror("Error", f"Cannot read image file: {os.path.basename(file_path)}")
                return False

            if test_img.size == 0:
                messagebox.showerror("Error", "Selected image file is empty")
                return False

            # Try to read with PIL as additional validation
            with Image.open(file_path) as pil_img:
                pil_img.verify()

            del test_img
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Image validation failed: {str(e)}")
            return False

    def _reset_drop_appearance(self):
        """Reset drop area appearance to default"""
        self.drop_frame.configure(
            border_color=("gray70", "gray30"),
            fg_color=("gray85", "gray20")
        )
        self.drop_text_label.configure(
            text="Drag & Drop X-Ray Image Here\nor click to browse"
        )

    def set_drop_enabled(self, enabled):
        """Enable or disable drop functionality"""
        self.drop_enabled = enabled

        if enabled:
            self.drop_text_label.configure(
                text="Drag & Drop X-Ray Image Here\nor click to browse",
                text_color=("gray50", "gray60")
            )
            self.drop_icon_label.configure(text="📁")
        else:
            self.drop_text_label.configure(
                text="Drop disabled during analysis",
                text_color=("gray30", "gray80")
            )
            self.drop_icon_label.configure(text="🚫")

    def show_processing_state(self):
        """Show processing state"""
        self.drop_icon_label.configure(text="⏳")
        self.drop_text_label.configure(text="Processing image...")

    def show_success_state(self, filename):
        """Show success state with filename"""
        self.drop_icon_label.configure(text="✅")
        self.drop_text_label.configure(text=f"Loaded: {filename}")

    def show_error_state(self, error_msg):
        """Show error state"""
        self.drop_icon_label.configure(text="❌")
        self.drop_text_label.configure(text=f"Error: {error_msg}")

    def reset_to_default(self):
        """Reset to default state"""
        self.drop_icon_label.configure(text="📁")
        self._reset_drop_appearance()
        self.set_drop_enabled(True)


