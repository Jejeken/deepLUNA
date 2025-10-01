"""
Drag and Drop Handler for CXR Analysis v3
=========================================

Enhanced drag and drop functionality with better integration and validation.
"""

import os
import threading
from typing import Callable, Optional, List, Dict, Any
from tkinter import messagebox
from .image_loader import ImageLoader
from .image_validator import ImageValidator


class DragDropHandler:
    """Enhanced drag and drop handler with validation and preprocessing"""

    def __init__(self, on_image_loaded: Optional[Callable] = None,
                 on_validation_failed: Optional[Callable] = None,
                 on_loading_started: Optional[Callable] = None,
                 on_loading_finished: Optional[Callable] = None):
        """
        Initialize drag and drop handler

        Args:
            on_image_loaded: Callback when image is successfully loaded (result_dict)
            on_validation_failed: Callback when validation fails (file_path, validation_result)
            on_loading_started: Callback when loading starts (file_path)
            on_loading_finished: Callback when loading finishes (file_path, success)
        """
        self.image_loader = ImageLoader()
        self.validator = ImageValidator()

        # Callbacks
        self.on_image_loaded = on_image_loaded
        self.on_validation_failed = on_validation_failed
        self.on_loading_started = on_loading_started
        self.on_loading_finished = on_loading_finished

        # State management
        self.enabled = True
        self.loading_in_progress = False
        self.current_loading_thread = None

        # Supported file patterns for filtering
        self.supported_patterns = [
            "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"
        ]

    def handle_file_drop(self, file_paths: List[str]):
        """Handle dropped files"""
        if not self.enabled or self.loading_in_progress:
            return

        # Filter for image files
        image_files = self._filter_image_files(file_paths)

        if not image_files:
            self._show_error("No valid image files found in dropped items")
            return

        if len(image_files) > 1:
            self._show_warning(f"Multiple images dropped. Using first image: {os.path.basename(image_files[0])}")

        # Load the first (or only) image
        self._load_image_async(image_files[0])

    def handle_single_file_drop(self, file_path: str):
        """Handle single file drop"""
        if not self.enabled or self.loading_in_progress:
            return

        if not self._is_image_file(file_path):
            self._show_error(f"Not a supported image format: {os.path.basename(file_path)}")
            return

        self._load_image_async(file_path)

    def _filter_image_files(self, file_paths: List[str]) -> List[str]:
        """Filter file paths to only include supported image files"""
        image_files = []
        for file_path in file_paths:
            if os.path.isfile(file_path) and self._is_image_file(file_path):
                image_files.append(file_path)
        return image_files

    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is a supported image format"""
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return False

        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    def _load_image_async(self, file_path: str):
        """Load image asynchronously"""
        if self.loading_in_progress:
            return

        self.loading_in_progress = True

        # Notify loading started
        if self.on_loading_started:
            self.on_loading_started(file_path)

        def load_worker():
            try:
                # Quick validation first
                if not self.validator.quick_validate(file_path):
                    self._handle_validation_failure(file_path, "Quick validation failed")
                    return

                # Full validation and loading
                load_result = self.image_loader.load_image(file_path, validate=True)

                if load_result['success']:
                    self._handle_loading_success(file_path, load_result)
                else:
                    self._handle_loading_failure(file_path, load_result['error_message'])

            except Exception as e:
                self._handle_loading_failure(file_path, f"Loading failed: {str(e)}")

            finally:
                self.loading_in_progress = False
                if self.on_loading_finished:
                    self.on_loading_finished(file_path, load_result.get('success', False))

        self.current_loading_thread = threading.Thread(target=load_worker, daemon=True)
        self.current_loading_thread.start()

    def _handle_loading_success(self, file_path: str, load_result: Dict[str, Any]):
        """Handle successful image loading"""
        try:
            # Check validation warnings
            validation = load_result.get('validation')
            if validation and validation.get('warnings'):
                warning_msg = f"Image loaded with warnings:\n"
                for warning in validation['warnings'][:3]:  # Show max 3 warnings
                    warning_msg += f"• {warning}\n"
                if len(validation['warnings']) > 3:
                    warning_msg += f"... and {len(validation['warnings']) - 3} more"

                self._show_warning(warning_msg)

            # Notify successful loading
            if self.on_image_loaded:
                self.on_image_loaded(load_result)

        except Exception as e:
            print(f"Error handling loading success: {e}")

    def _handle_loading_failure(self, file_path: str, error_message: str):
        """Handle image loading failure"""
        try:
            # Create validation-like result for consistency
            failure_result = {
                'valid': False,
                'errors': [error_message],
                'warnings': [],
                'info': {'file_path': file_path}
            }

            if self.on_validation_failed:
                self.on_validation_failed(file_path, failure_result)
            else:
                self._show_error(f"Failed to load image:\n{error_message}")

        except Exception as e:
            print(f"Error handling loading failure: {e}")

    def _handle_validation_failure(self, file_path: str, error_message: str):
        """Handle validation failure"""
        self.loading_in_progress = False

        failure_result = {
            'valid': False,
            'errors': [error_message],
            'warnings': [],
            'info': {'file_path': file_path}
        }

        if self.on_validation_failed:
            self.on_validation_failed(file_path, failure_result)
        else:
            self._show_error(f"Image validation failed:\n{error_message}")

        if self.on_loading_finished:
            self.on_loading_finished(file_path, False)

    def _show_error(self, message: str):
        """Show error message"""
        try:
            messagebox.showerror("Image Loading Error", message)
        except Exception:
            print(f"Error: {message}")

    def _show_warning(self, message: str):
        """Show warning message"""
        try:
            messagebox.showwarning("Image Loading Warning", message)
        except Exception:
            print(f"Warning: {message}")

    def set_enabled(self, enabled: bool):
        """Enable or disable drag and drop"""
        self.enabled = enabled

    def is_enabled(self) -> bool:
        """Check if drag and drop is enabled"""
        return self.enabled and not self.loading_in_progress

    def cancel_loading(self):
        """Cancel current loading operation"""
        if self.loading_in_progress and self.current_loading_thread:
            # Note: Cannot actually cancel thread, but can set flag
            self.loading_in_progress = False

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return [ext.replace('*', '') for ext in self.supported_patterns]

    def get_format_description(self) -> str:
        """Get description of supported formats"""
        formats = [ext.replace('*.', '').upper() for ext in self.supported_patterns]
        return f"Supported formats: {', '.join(formats)}"

    def validate_file_before_drop(self, file_path: str) -> Dict[str, Any]:
        """Validate file before accepting drop (for preview)"""
        if not self._is_image_file(file_path):
            return {
                'valid': False,
                'error': 'Unsupported file format',
                'can_accept': False
            }

        # Quick validation
        if not self.validator.quick_validate(file_path):
            return {
                'valid': False,
                'error': 'File appears to be corrupted or invalid',
                'can_accept': False
            }

        # Get basic file info
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            return {
                'valid': True,
                'can_accept': True,
                'file_name': file_name,
                'file_size': file_size,
                'preview_text': f"{file_name} ({file_size / 1024:.1f} KB)"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f'Cannot access file: {str(e)}',
                'can_accept': False
            }

    def get_drop_zones_config(self) -> Dict[str, Any]:
        """Get configuration for drag and drop zones"""
        return {
            'accept_files': True,
            'accept_directories': False,
            'multiple_files': True,  # Will use first image only
            'file_types': self.supported_patterns,
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'enabled': self.enabled,
            'loading': self.loading_in_progress
        }

    def create_file_filter_string(self) -> str:
        """Create file filter string for file dialogs"""
        # For tkinter filedialog
        filter_parts = []
        filter_parts.append(("Image files", " ".join(self.supported_patterns)))

        for pattern in self.supported_patterns:
            ext = pattern.replace('*', '').upper()
            filter_parts.append((f"{ext} files", pattern))

        filter_parts.append(("All files", "*.*"))

        return filter_parts

    def get_loading_status(self) -> Dict[str, Any]:
        """Get current loading status"""
        return {
            'enabled': self.enabled,
            'loading_in_progress': self.loading_in_progress,
            'has_current_image': self.image_loader.is_image_loaded(),
            'current_image_path': self.image_loader.current_image_path,
            'cache_info': self.image_loader.get_cache_info()
        }

    def clear_current_image(self):
        """Clear currently loaded image"""
        self.image_loader.clear_current_image()

    def get_current_image_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded image"""
        if self.image_loader.is_image_loaded():
            validation = self.image_loader.get_current_validation()
            if validation and validation.get('info'):
                return validation['info']
        return None