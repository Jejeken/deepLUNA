"""
Image Preview Component for CXR Analysis v3
===========================================

Enhanced image preview with better layout and display options.
"""

import customtkinter as ctk
import numpy as np
import cv2
from PIL import Image, ImageTk
import os


class ImagePreviewFrame(ctk.CTkFrame):
    """Enhanced image preview frame with drag-drop integration"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=("gray90", "gray17"), **kwargs)

        self.parent = parent
        self.current_image = None
        self.current_image_path = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the image preview UI"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header section
        self.header_frame = ctk.CTkFrame(self, height=50)
        self.header_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Image Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        # Image info label
        self.info_label = ctk.CTkLabel(
            self.header_frame,
            text="No image loaded",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.info_label.grid(row=0, column=1, padx=15, pady=10, sticky="e")

        # Main preview area
        self.preview_container = ctk.CTkFrame(self)
        self.preview_container.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.preview_container.grid_columnconfigure(0, weight=1)
        self.preview_container.grid_rowconfigure(0, weight=1)

        # Initial state - show drag and drop area
        from .drag_drop import DragDropFrame
        self.drag_drop_frame = DragDropFrame(self.preview_container)
        self.drag_drop_frame.grid(row=0, column=0, sticky="nsew")

        # Image display label (hidden initially)
        self.image_label = ctk.CTkLabel(
            self.preview_container,
            text=""
        )

    def load_image_preview(self, image_path):
        """Load and display image preview"""
        try:
            # Validate file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image file: {image_path}")

            # Store references
            self.current_image = image
            self.current_image_path = image_path

            # Update UI
            self._display_image_preview(image, image_path)
            self._update_image_info(image, image_path)

            # Hide drag-drop, show image
            self.drag_drop_frame.grid_remove()
            self.image_label.grid(row=0, column=0, sticky="nsew")

            return True

        except Exception as e:
            print(f"Error loading image preview: {e}")
            self.show_error(f"Failed to load image: {str(e)}")
            return False

    def _display_image_preview(self, image, image_path):
        """Display the image in the preview area"""
        try:
            # Calculate display size maintaining aspect ratio
            container_width = 600  # Approximate container width
            container_height = 400  # Approximate container height

            h, w = image.shape
            aspect_ratio = w / h

            if aspect_ratio > container_width / container_height:
                # Width is limiting factor
                display_width = min(container_width, w)
                display_height = int(display_width / aspect_ratio)
            else:
                # Height is limiting factor
                display_height = min(container_height, h)
                display_width = int(display_height * aspect_ratio)

            # Resize image for display
            display_image = cv2.resize(image, (display_width, display_height))

            # Convert to PIL and then to CTkImage
            pil_image = Image.fromarray(display_image)
            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(display_width, display_height)
            )

            # Update image label
            self.image_label.configure(image=ctk_image, text="")

        except Exception as e:
            print(f"Error displaying image: {e}")
            self.image_label.configure(text=f"Error displaying image: {str(e)}", image="")

    def _update_image_info(self, image, image_path):
        """Update image information display"""
        try:
            filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path)

            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"

            # Update info label
            info_text = f"{filename} | {image.shape[1]}×{image.shape[0]} | {size_str}"
            self.info_label.configure(text=info_text)

        except Exception as e:
            print(f"Error updating image info: {e}")
            self.info_label.configure(text="Error reading image info")

    def show_processing_state(self):
        """Show processing state"""
        self.drag_drop_frame.grid_remove()
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_label.configure(
            text="🔄 Processing image...",
            image="",
            font=ctk.CTkFont(size=16)
        )
        self.info_label.configure(text="Processing...")

    def show_analysis_state(self):
        """Show analysis in progress state"""
        # Keep current image but show overlay
        if self.current_image is not None:
            overlay_text = "🔬 Analysis in progress..."
            self._show_overlay_text(overlay_text)
        else:
            self.image_label.configure(
                text="🔬 Analyzing...",
                font=ctk.CTkFont(size=16)
            )
        self.info_label.configure(text="Analyzing...")

    def _show_overlay_text(self, text):
        """Show overlay text on current image"""
        if self.current_image is not None:
            try:
                # Create overlay on current image
                overlay_image = self.current_image.copy()

                # Add semi-transparent overlay
                overlay = np.zeros_like(overlay_image)
                overlay_image = cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0)

                # Convert and display
                h, w = overlay_image.shape
                display_size = min(400, min(h, w))
                display_image = cv2.resize(overlay_image, (display_size, display_size))

                pil_image = Image.fromarray(display_image)
                ctk_image = ctk.CTkImage(
                    light_image=pil_image,
                    dark_image=pil_image,
                    size=(display_size, display_size)
                )

                self.image_label.configure(image=ctk_image, text=text, compound="center")

            except Exception as e:
                print(f"Error showing overlay: {e}")
                self.image_label.configure(text=text, image="")

    def show_error(self, error_message):
        """Show error state"""
        self.drag_drop_frame.grid_remove()
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_label.configure(
            text=f"❌ {error_message}",
            image="",
            font=ctk.CTkFont(size=14),
            text_color="red"
        )
        self.info_label.configure(text="Error")

    def reset_to_initial_state(self):
        """Reset to initial drag-drop state"""
        self.current_image = None
        self.current_image_path = None

        # Hide image, show drag-drop
        self.image_label.grid_remove()
        self.drag_drop_frame.grid(row=0, column=0, sticky="nsew")
        self.drag_drop_frame.reset_to_default()

        # Reset info
        self.info_label.configure(text="No image loaded")

    def get_current_image_path(self):
        """Get current image path"""
        return self.current_image_path

    def get_current_image(self):
        """Get current image array"""
        return self.current_image

    def set_drag_drop_enabled(self, enabled):
        """Enable or disable drag and drop"""
        if hasattr(self.drag_drop_frame, 'set_drop_enabled'):
            self.drag_drop_frame.set_drop_enabled(enabled)


class ImageComparisonFrame(ctk.CTkFrame):
    """Frame for comparing original and processed images"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.parent = parent
        self._setup_ui()

    def _setup_ui(self):
        """Setup comparison UI"""
        # Configure grid
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Headers
        self.original_header = ctk.CTkLabel(
            self,
            text="Original Image",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.original_header.grid(row=0, column=0, pady=10)

        self.processed_header = ctk.CTkLabel(
            self,
            text="Processed Image",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.processed_header.grid(row=0, column=1, pady=10)

        # Image containers
        self.original_label = ctk.CTkLabel(self, text="No original image")
        self.original_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.processed_label = ctk.CTkLabel(self, text="No processed image")
        self.processed_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    def set_images(self, original_image, processed_image):
        """Set original and processed images for comparison"""
        try:
            size = 300

            # Display original
            if original_image is not None:
                orig_resized = cv2.resize(original_image, (size, size))
                orig_pil = Image.fromarray(orig_resized)
                orig_ctk = ctk.CTkImage(light_image=orig_pil, dark_image=orig_pil, size=(size, size))
                self.original_label.configure(image=orig_ctk, text="")

            # Display processed
            if processed_image is not None:
                proc_resized = cv2.resize(processed_image, (size, size))
                proc_pil = Image.fromarray(proc_resized)
                proc_ctk = ctk.CTkImage(light_image=proc_pil, dark_image=proc_pil, size=(size, size))
                self.processed_label.configure(image=proc_ctk, text="")

        except Exception as e:
            print(f"Error setting comparison images: {e}")

    def clear_images(self):
        """Clear both images"""
        self.original_label.configure(image="", text="No original image")
        self.processed_label.configure(image="", text="No processed image")