"""
Image Validator for CXR Analysis v3
===================================

Comprehensive image validation with medical imaging specific checks.
"""

import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import Dict, List, Tuple, Optional, Any
import mimetypes


class ImageValidator:
    """Enhanced image validator with comprehensive checks"""

    def __init__(self):
        """Initialize image validator with configuration"""
        # Supported formats
        self.supported_extensions = {
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'
        }

        # Size constraints
        self.min_size = (64, 64)  # Minimum image dimensions
        self.max_size = (8192, 8192)  # Maximum image dimensions
        self.max_file_size = 100 * 1024 * 1024  # 100MB max file size

        # Medical imaging preferences
        self.preferred_formats = {'.png', '.tiff', '.tif'}  # Lossless formats preferred
        self.medical_size_ranges = {
            'chest_xray': [(256, 256), (2048, 2048)],  # Typical chest X-ray sizes
            'general': [(128, 128), (4096, 4096)]
        }

    def validate_image_file(self, file_path: str) -> Dict[str, Any]:
        """
        Comprehensive image file validation

        Returns:
            Dict with validation results including:
            - valid: bool
            - errors: List[str]
            - warnings: List[str]
            - info: Dict[str, Any]
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        try:
            # Basic file checks
            file_checks = self._validate_file_basic(file_path)
            result['errors'].extend(file_checks['errors'])
            result['warnings'].extend(file_checks['warnings'])
            result['info'].update(file_checks['info'])

            if file_checks['errors']:
                result['valid'] = False
                return result

            # Image format checks
            format_checks = self._validate_image_format(file_path)
            result['errors'].extend(format_checks['errors'])
            result['warnings'].extend(format_checks['warnings'])
            result['info'].update(format_checks['info'])

            if format_checks['errors']:
                result['valid'] = False
                return result

            # Image content checks
            content_checks = self._validate_image_content(file_path)
            result['errors'].extend(content_checks['errors'])
            result['warnings'].extend(content_checks['warnings'])
            result['info'].update(content_checks['info'])

            if content_checks['errors']:
                result['valid'] = False
                return result

            # Medical imaging specific checks
            medical_checks = self._validate_medical_imaging(file_path, result['info'])
            result['warnings'].extend(medical_checks['warnings'])
            result['info'].update(medical_checks['info'])

        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation failed: {str(e)}")

        return result

    def _validate_file_basic(self, file_path: str) -> Dict[str, Any]:
        """Basic file validation"""
        result = {'errors': [], 'warnings': [], 'info': {}}

        # Check file existence
        if not os.path.exists(file_path):
            result['errors'].append("File does not exist")
            return result

        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            result['errors'].append("Path is not a file")
            return result

        # Get file info
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        file_ext = os.path.splitext(file_path)[1].lower()

        result['info'].update({
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': file_size,
            'file_extension': file_ext
        })

        # Check file size
        if file_size == 0:
            result['errors'].append("File is empty")
            return result

        if file_size > self.max_file_size:
            result['errors'].append(f"File too large: {file_size / (1024*1024):.1f}MB (max: {self.max_file_size / (1024*1024):.1f}MB)")

        # Check file extension
        if file_ext not in self.supported_extensions:
            result['errors'].append(f"Unsupported file format: {file_ext}")
            return result

        # Mime type validation
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and not mime_type.startswith('image/'):
            result['warnings'].append(f"Unexpected MIME type: {mime_type}")

        result['info']['mime_type'] = mime_type

        return result

    def _validate_image_format(self, file_path: str) -> Dict[str, Any]:
        """Validate image format and readability"""
        result = {'errors': [], 'warnings': [], 'info': {}}

        try:
            # Try reading with PIL
            with Image.open(file_path) as pil_img:
                result['info'].update({
                    'pil_format': pil_img.format,
                    'pil_mode': pil_img.mode,
                    'pil_size': pil_img.size
                })

                # Check for corruption
                try:
                    pil_img.verify()
                except Exception as e:
                    result['errors'].append(f"Image appears corrupted (PIL): {str(e)}")
                    return result

        except Exception as e:
            result['errors'].append(f"Cannot open image with PIL: {str(e)}")
            return result

        # Try reading with OpenCV
        try:
            cv_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                result['errors'].append("Cannot open image with OpenCV")
                return result

            result['info'].update({
                'cv_shape': cv_img.shape,
                'cv_dtype': str(cv_img.dtype)
            })

            # Check for empty image
            if cv_img.size == 0:
                result['errors'].append("Image has no pixels")
                return result

        except Exception as e:
            result['errors'].append(f"Cannot read image with OpenCV: {str(e)}")

        return result

    def _validate_image_content(self, file_path: str) -> Dict[str, Any]:
        """Validate image content and properties"""
        result = {'errors': [], 'warnings': [], 'info': {}}

        try:
            # Read image for analysis
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                result['errors'].append("Cannot read image for content analysis")
                return result

            height, width = image.shape
            result['info'].update({
                'dimensions': (width, height),
                'aspect_ratio': width / height if height > 0 else 0
            })

            # Size validation
            if width < self.min_size[0] or height < self.min_size[1]:
                result['errors'].append(f"Image too small: {width}x{height} (min: {self.min_size[0]}x{self.min_size[1]})")

            if width > self.max_size[0] or height > self.max_size[1]:
                result['errors'].append(f"Image too large: {width}x{height} (max: {self.max_size[0]}x{self.max_size[1]})")

            # Content analysis
            content_analysis = self._analyze_image_content(image)
            result['info'].update(content_analysis)

            # Check for mostly empty/black image
            if content_analysis['mean_intensity'] < 10:
                result['warnings'].append("Image appears very dark/mostly black")

            # Check for overexposed image
            if content_analysis['mean_intensity'] > 245:
                result['warnings'].append("Image appears overexposed/mostly white")

            # Check for low contrast
            if content_analysis['std_intensity'] < 20:
                result['warnings'].append("Image has very low contrast")

        except Exception as e:
            result['warnings'].append(f"Content analysis failed: {str(e)}")

        return result

    def _validate_medical_imaging(self, file_path: str, existing_info: Dict) -> Dict[str, Any]:
        """Medical imaging specific validation"""
        result = {'warnings': [], 'info': {}}

        try:
            # Check file format preferences for medical imaging
            file_ext = existing_info.get('file_extension', '').lower()
            if file_ext not in self.preferred_formats:
                result['warnings'].append(f"Format {file_ext} may have compression artifacts. Consider using PNG or TIFF for medical images")

            # Check image dimensions for typical chest X-ray sizes
            dimensions = existing_info.get('dimensions')
            if dimensions:
                width, height = dimensions
                aspect_ratio = width / height

                # Typical chest X-ray aspect ratios
                if not (0.7 <= aspect_ratio <= 1.5):
                    result['warnings'].append(f"Unusual aspect ratio for chest X-ray: {aspect_ratio:.2f}")

                # Check if dimensions are typical for chest X-rays
                is_typical_size = False
                for min_size, max_size in self.medical_size_ranges['chest_xray']:
                    if (min_size[0] <= width <= max_size[0] and
                        min_size[1] <= height <= max_size[1]):
                        is_typical_size = True
                        break

                if not is_typical_size:
                    result['warnings'].append(f"Unusual dimensions for chest X-ray: {width}x{height}")

            # Check for EXIF data (may contain patient info)
            try:
                with Image.open(file_path) as pil_img:
                    exif_data = pil_img._getexif()
                    if exif_data:
                        result['warnings'].append("Image contains EXIF data - ensure no patient information is included")
                        result['info']['has_exif'] = True
                    else:
                        result['info']['has_exif'] = False
            except:
                result['info']['has_exif'] = False

        except Exception as e:
            result['warnings'].append(f"Medical imaging validation incomplete: {str(e)}")

        return result

    def _analyze_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image content for quality metrics"""
        analysis = {}

        try:
            # Basic statistics
            analysis['mean_intensity'] = float(np.mean(image))
            analysis['std_intensity'] = float(np.std(image))
            analysis['min_intensity'] = int(np.min(image))
            analysis['max_intensity'] = int(np.max(image))

            # Histogram analysis
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            analysis['histogram_peak'] = int(np.argmax(hist))

            # Edge detection for content assessment
            edges = cv2.Canny(image, 50, 150)
            analysis['edge_density'] = float(np.sum(edges > 0) / edges.size)

            # Gradient magnitude for sharpness assessment
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            analysis['average_gradient'] = float(np.mean(gradient_magnitude))

            # Noise estimation (using high-frequency content)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_response = cv2.filter2D(image, cv2.CV_32F, kernel)
            analysis['noise_estimate'] = float(np.std(noise_response))

        except Exception as e:
            analysis['analysis_error'] = str(e)

        return analysis

    def quick_validate(self, file_path: str) -> bool:
        """Quick validation for basic file checks"""
        try:
            # Check file existence and extension
            if not os.path.exists(file_path):
                return False

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_extensions:
                return False

            # Quick OpenCV read test
            test_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if test_img is None or test_img.size == 0:
                return False

            return True

        except:
            return False

    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """Get human-readable validation summary"""
        if not validation_result['valid']:
            errors = '; '.join(validation_result['errors'])
            return f"❌ Invalid: {errors}"

        warnings_count = len(validation_result['warnings'])
        if warnings_count == 0:
            return "✅ Valid image file"
        else:
            return f"⚠️ Valid with {warnings_count} warning(s)"

    def get_image_info_string(self, validation_result: Dict[str, Any]) -> str:
        """Get formatted image information string"""
        if not validation_result['valid']:
            return "Invalid image file"

        info = validation_result['info']
        dimensions = info.get('dimensions', (0, 0))
        file_size = info.get('file_size', 0)
        file_ext = info.get('file_extension', 'unknown')

        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"

        return f"{dimensions[0]}×{dimensions[1]} • {file_ext.upper()} • {size_str}"

    def batch_validate(self, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """Validate multiple image files"""
        results = {}
        for file_path in file_paths:
            results[file_path] = self.validate_image_file(file_path)
        return results