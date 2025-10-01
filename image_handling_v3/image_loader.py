"""
Image Loader for CXR Analysis v3
================================

Enhanced image loading with preprocessing and caching capabilities.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List
import threading
import queue
import time
from .image_validator import ImageValidator


class ImageLoader:
    """Enhanced image loader with validation and preprocessing"""

    def __init__(self, cache_size: int = 5):
        """Initialize image loader with caching"""
        self.validator = ImageValidator()
        self.cache_size = cache_size
        self.image_cache = {}  # Cache for loaded images
        self.cache_order = []  # LRU tracking

        # Loading state
        self.current_image_path = None
        self.current_image = None
        self.current_validation = None

        # Thread safety
        self.loading_lock = threading.Lock()

    def load_image(self, file_path: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load image with optional validation

        Returns:
            Dict containing:
            - success: bool
            - image: np.ndarray (if successful)
            - original_image: np.ndarray (unchanged)
            - validation: Dict (if validation enabled)
            - error_message: str (if failed)
            - info: Dict (image information)
        """
        result = {
            'success': False,
            'image': None,
            'original_image': None,
            'validation': None,
            'error_message': '',
            'info': {}
        }

        try:
            with self.loading_lock:
                # Check cache first
                if file_path in self.image_cache:
                    cached_data = self.image_cache[file_path]
                    result.update(cached_data)
                    result['success'] = True
                    self._update_cache_order(file_path)
                    return result

                # Validate image if requested
                if validate:
                    validation_result = self.validator.validate_image_file(file_path)
                    result['validation'] = validation_result

                    if not validation_result['valid']:
                        result['error_message'] = '; '.join(validation_result['errors'])
                        return result

                # Load image
                load_result = self._load_image_file(file_path)
                if not load_result['success']:
                    result['error_message'] = load_result['error_message']
                    return result

                # Update result
                result.update(load_result)
                result['success'] = True

                # Store current image reference
                self.current_image_path = file_path
                self.current_image = result['image']
                self.current_validation = result['validation']

                # Cache the result
                self._cache_image(file_path, result)

                return result

        except Exception as e:
            result['error_message'] = f"Failed to load image: {str(e)}"
            return result

    def _load_image_file(self, file_path: str) -> Dict[str, Any]:
        """Load image file using OpenCV and PIL"""
        result = {
            'success': False,
            'image': None,
            'original_image': None,
            'info': {},
            'error_message': ''
        }

        try:
            # Load with OpenCV for processing
            cv_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if cv_image is None:
                result['error_message'] = "Cannot read image with OpenCV"
                return result

            # Store original
            original_image = cv_image.copy()

            # Get image information
            height, width = cv_image.shape
            file_size = os.path.getsize(file_path)

            result['info'] = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': file_size,
                'dimensions': (width, height),
                'channels': 1,  # Grayscale
                'dtype': str(cv_image.dtype),
                'min_value': int(np.min(cv_image)),
                'max_value': int(np.max(cv_image)),
                'mean_value': float(np.mean(cv_image))
            }

            # Verify with PIL for additional format support
            try:
                with Image.open(file_path) as pil_image:
                    result['info']['pil_format'] = pil_image.format
                    result['info']['pil_mode'] = pil_image.mode
            except Exception as e:
                result['info']['pil_error'] = str(e)

            result['image'] = cv_image
            result['original_image'] = original_image
            result['success'] = True

        except Exception as e:
            result['error_message'] = f"Error loading image file: {str(e)}"

        return result

    def _cache_image(self, file_path: str, image_data: Dict[str, Any]):
        """Cache loaded image data"""
        try:
            # Remove oldest if cache is full
            if len(self.image_cache) >= self.cache_size:
                oldest_path = self.cache_order[0]
                del self.image_cache[oldest_path]
                self.cache_order.remove(oldest_path)

            # Create cache entry (copy arrays to avoid reference issues)
            cache_entry = {
                'image': image_data['image'].copy() if image_data['image'] is not None else None,
                'original_image': image_data['original_image'].copy() if image_data['original_image'] is not None else None,
                'validation': image_data['validation'],
                'info': image_data['info'].copy()
            }

            self.image_cache[file_path] = cache_entry
            self.cache_order.append(file_path)

        except Exception as e:
            print(f"Error caching image: {e}")

    def _update_cache_order(self, file_path: str):
        """Update cache order for LRU"""
        if file_path in self.cache_order:
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Preprocess image for analysis"""
        try:
            processed = image.copy()

            # Resize if target size specified
            if target_size:
                processed = cv2.resize(processed, target_size)

            # Normalize to 0-1 range
            if processed.dtype != np.float32:
                processed = processed.astype(np.float32) / 255.0

            return processed

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return image

    def load_and_preprocess(self, file_path: str, target_size: Tuple[int, int] = None,
                          validate: bool = True) -> Dict[str, Any]:
        """Load and preprocess image in one step"""
        load_result = self.load_image(file_path, validate=validate)

        if load_result['success'] and target_size:
            processed_image = self.preprocess_image(load_result['image'], target_size)
            load_result['processed_image'] = processed_image

        return load_result

    def get_current_image(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Get currently loaded image and path"""
        return self.current_image, self.current_image_path

    def get_current_validation(self) -> Optional[Dict[str, Any]]:
        """Get validation result for current image"""
        return self.current_validation

    def is_image_loaded(self) -> bool:
        """Check if an image is currently loaded"""
        return self.current_image is not None and self.current_image_path is not None

    def clear_current_image(self):
        """Clear current image reference"""
        self.current_image = None
        self.current_image_path = None
        self.current_validation = None

    def clear_cache(self):
        """Clear all cached images"""
        with self.loading_lock:
            self.image_cache.clear()
            self.cache_order.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'cache_size': len(self.image_cache),
            'max_cache_size': self.cache_size,
            'cached_images': list(self.cache_order),
            'current_image': self.current_image_path
        }

    def batch_load_images(self, file_paths: List[str], validate: bool = True) -> Dict[str, Dict[str, Any]]:
        """Load multiple images"""
        results = {}
        for file_path in file_paths:
            results[file_path] = self.load_image(file_path, validate=validate)
        return results

    def async_load_image(self, file_path: str, callback, validate: bool = True):
        """Load image asynchronously and call callback when done"""
        def load_worker():
            result = self.load_image(file_path, validate=validate)
            callback(result)

        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()

    def create_thumbnail(self, image: np.ndarray, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """Create thumbnail of image"""
        try:
            # Calculate aspect ratio preserving resize
            h, w = image.shape
            target_w, target_h = size

            aspect_ratio = w / h
            if aspect_ratio > target_w / target_h:
                # Width is limiting
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                # Height is limiting
                new_h = target_h
                new_w = int(target_h * aspect_ratio)

            thumbnail = cv2.resize(image, (new_w, new_h))
            return thumbnail

        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return cv2.resize(image, size)  # Fallback to direct resize

    def export_image(self, image: np.ndarray, file_path: str, quality: int = 95) -> bool:
        """Export image to file"""
        try:
            # Ensure image is in correct format
            if image.dtype == np.float32 or image.dtype == np.float64:
                image_export = (image * 255).astype(np.uint8)
            else:
                image_export = image

            # Determine format and save
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in ['.jpg', '.jpeg']:
                cv2.imwrite(file_path, image_export, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif file_ext == '.png':
                cv2.imwrite(file_path, image_export, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(file_path, image_export)

            return True

        except Exception as e:
            print(f"Error exporting image: {e}")
            return False

    def get_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive image statistics"""
        try:
            stats = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'median': float(np.median(image))
            }

            # Histogram information
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            stats['histogram_peak'] = int(np.argmax(hist))
            stats['histogram_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))

            return stats

        except Exception as e:
            return {'error': str(e)}

    def validate_current_image(self) -> Optional[Dict[str, Any]]:
        """Validate currently loaded image"""
        if self.current_image_path:
            return self.validator.validate_image_file(self.current_image_path)
        return None