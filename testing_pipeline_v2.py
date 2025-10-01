"""
Comprehensive Testing Script for CXR Pipeline with Excel Report
================================================================

Tests the enhanced pipeline from app_CXR_v2.py on 704 images from Kaggle dataset.
Generates performance metrics, confusion matrix, AUC-ROC curve, and detailed Excel report with embedded images.
Excel columns: ID, Ground Truth Label, Prediction Label, Mask GT, Mask Prediction, Original X-Ray Image

Dataset: https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation/data

Requirements:
- app_CXR_v2.py in same directory
- Dataset extracted with proper structure
- Required model files (best_resunet_lung_segmentation.h5, optimized_classifier_efficientB2.pth)
- openpyxl and Pillow for Excel generation
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc
import time
from pathlib import Path
import json

# Excel and image handling
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill
from PIL import Image
import io

warnings.filterwarnings('ignore')

# Import necessary components from app_CXR_v2.py
try:
    from app_CXR_v2 import (
        OptimizedMultiInputClassifier,
        AdvancedXRayTransforms,
        apply_advanced_soft_mask,
        apply_attention_mask,
        dice_coefficient, dice_loss, iou_coefficient, iou_loss,
        combined_loss, precision, recall, specificity,
        cleanup_gpu_memory
    )
    print("✓ Successfully imported components from app_CXR_v2.py")
except ImportError as e:
    print(f"✗ Error importing from app_CXR_v2.py: {e}")
    print("Please ensure app_CXR_v2.py is in the same directory")
    sys.exit(1)

class CXRPipelineTester:
    """Comprehensive tester for CXR analysis pipeline with Excel reporting"""
    
    def __init__(self, dataset_path, metadata_path, results_dir="test_results"):
        """
        Initialize the pipeline tester
        
        Args:
            dataset_path: Path to Chest-X-Ray dataset folder
            metadata_path: Path to MetaData.csv file
            results_dir: Directory to save test results
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = Path(metadata_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create temp directory for images
        self.temp_dir = self.results_dir / "temp_images"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Model components
        self.sgm_model = None
        self.clf_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test results storage
        self.test_results = {
            'predictions': [],
            'ground_truth': [],
            'probabilities': [],
            'image_ids': [],
            'processing_times': [],
            'errors': []
        }
        
        # Excel data storage
        self.excel_data = []
        
        print(f"Initialized CXR Pipeline Tester")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Device: {self.device}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def load_models(self):
        """Load both segmentation and classification models"""
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)
        
        # Custom objects for segmentation model
        custom_objects = {
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient,
            'iou_coefficient': iou_coefficient,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }
        
        # Define possible model paths
        seg_model_paths = [
            'best_resunet_lung_segmentation.h5',
            'models/best_resunet_lung_segmentation.h5',
            'J:/Portofolio Diri/BRIN/aplikasi/models/segmentation/best_resunet_lung_segmentation.h5'
        ]
        
        clf_model_paths = [
            'optimized_classifier_efficientB2.pth',
            'models/optimized_classifier_efficientB2.pth',
            'J:/Portofolio Diri/BRIN/aplikasi/models/clf/optimized_classifier_efficientB2.pth'
        ]
        
        # Load segmentation model
        seg_loaded = False
        for path in seg_model_paths:
            if os.path.exists(path):
                try:
                    print(f"Loading segmentation model from: {path}")
                    self.sgm_model = keras.models.load_model(path, custom_objects=custom_objects)
                    print("✓ Segmentation model loaded successfully")
                    seg_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load from {path}: {e}")
                    continue
        
        if not seg_loaded:
            raise FileNotFoundError("ResUNet segmentation model not found in any expected location")
        
        # Load classification model
        clf_loaded = False
        for path in clf_model_paths:
            if os.path.exists(path):
                try:
                    print(f"Loading classification model from: {path}")
                    self.clf_model = OptimizedMultiInputClassifier(dropout=0.3).to(self.device)
                    
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.clf_model.load_state_dict(checkpoint['model_state_dict'])
                        self.model_config = checkpoint.get('model_config', {})
                        self.performance_metrics = checkpoint.get('performance_metrics', {})
                    else:
                        self.clf_model.load_state_dict(checkpoint)
                    
                    self.clf_model.eval()
                    print("✓ Classification model loaded successfully")
                    clf_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load from {path}: {e}")
                    continue
        
        if not clf_loaded:
            raise FileNotFoundError("OptimizedMultiInputClassifier model not found in any expected location")
        
        print("✓ All models loaded successfully")
    
    def load_metadata(self):
        """Load and validate metadata"""
        print("\n" + "="*60)
        print("LOADING METADATA")
        print("="*60)
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"MetaData.csv not found at: {self.metadata_path}")
        
        self.metadata = pd.read_csv(self.metadata_path)
        print(f"✓ Metadata loaded: {len(self.metadata)} records")
        
        # Validate required columns
        required_cols = ['id', 'ptb']
        for col in required_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"Required column '{col}' not found in metadata")
        
        # Display label distribution
        label_dist = self.metadata['ptb'].value_counts().sort_index()
        print(f"Label distribution:")
        print(f"  Normal (0): {label_dist.get(0, 0)} images")
        print(f"  TB (1): {label_dist.get(1, 0)} images")
        print(f"  Total: {len(self.metadata)} images")
        
        # Validate image paths
        image_dir = self.dataset_path / "Chest-X-Ray" / "Chest-X-Ray" / "image"
        mask_dir = self.dataset_path / "Chest-X-Ray" / "Chest-X-Ray" / "mask"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Check how many metadata entries have corresponding images
        valid_images = 0
        for _, row in self.metadata.iterrows():
            image_path = image_dir / f"{row['id']}.png"
            if image_path.exists():
                valid_images += 1
        
        print(f"✓ Found {valid_images} valid images out of {len(self.metadata)} metadata entries")
        
        if valid_images == 0:
            raise ValueError("No valid images found. Check dataset structure.")
        
        return image_dir, mask_dir
    
    def _save_image_for_excel(self, image_id, image_array, image_type):
        """
        Save image to temp directory for Excel embedding
        
        Args:
            image_id: Image identifier
            image_array: Image as numpy array
            image_type: Type of image (original, mask_gt, mask_pred, gradcam)
            
        Returns:
            Path to saved image
        """
        filename = f"{image_id}_{image_type}.png"
        filepath = self.temp_dir / filename
        
        # Ensure image is in correct format
        if len(image_array.shape) == 2:
            # Grayscale to BGR
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.shape[2] == 1:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(str(filepath), image_array)
        return str(filepath)
    
    def _prepare_image_for_excel(self, image_path, target_width=200):
        """
        Prepare and resize image for Excel embedding
        
        Args:
            image_path: Path to image file
            target_width: Target width in pixels
            
        Returns:
            openpyxl Image object or None
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to fit Excel cell
            aspect_ratio = img.shape[0] / img.shape[1]
            target_height = int(target_width * aspect_ratio)
            img_resized = cv2.resize(img, (target_width, target_height))
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img_resized)
            
            # Save to BytesIO buffer
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Create openpyxl Image
            xl_img = XLImage(img_buffer)
            
            return xl_img
            
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            return None
    
    def inference_single_image(self, image_path, mask_gt_path=None):
        """
        Run inference on a single image with full outputs for Excel
        
        Args:
            image_path: Path to the image file
            mask_gt_path: Path to ground truth mask (optional)
            
        Returns:
            dict: Results including prediction, mask, and image paths (no GradCAM)
        """
        start_time = time.time()
        
        try:
            image_id = Path(image_path).stem
            
            # Step 1: Load image
            original_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Load ground truth mask if available
            mask_gt = None
            if mask_gt_path and os.path.exists(mask_gt_path):
                mask_gt = cv2.imread(str(mask_gt_path), cv2.IMREAD_GRAYSCALE)
            
            # Step 2: Segmentation preprocessing
            seg_input = cv2.resize(original_img, (256, 256))
            seg_input = seg_input.astype(np.float32) / 255.0
            seg_input = np.expand_dims(seg_input, axis=(0, -1))
            
            # Step 3: Run segmentation
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                mask_pred_raw = self.sgm_model.predict(seg_input, verbose=0)
            mask_pred = (mask_pred_raw[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            
            # Fallback if mask is empty
            if np.sum(mask_pred) == 0:
                mask_pred = np.zeros((256, 256), dtype=np.uint8)
                mask_pred[64:192, 64:192] = 255
            
            # Step 4: Create masked versions
            original_resized = cv2.resize(original_img, (256, 256))
            segmented_soft = apply_advanced_soft_mask(original_resized, mask_pred // 255, alpha=0.2)
            segmented_attention = apply_attention_mask(original_resized, mask_pred // 255)
            
            # Step 5: Prepare classification inputs
            clf_transform = AdvancedXRayTransforms(image_size=224, is_training=False)
            
            original_tensor = clf_transform(original_resized).unsqueeze(0).to(self.device)
            soft_tensor = clf_transform(segmented_soft).unsqueeze(0).to(self.device)
            att_tensor = clf_transform(segmented_attention).unsqueeze(0).to(self.device)
            
            # Step 6: Run classification
            self.clf_model.eval()
            
            with torch.no_grad():
                # Original input (type = 0)
                output_orig = self.clf_model(original_tensor, torch.tensor([0]).to(self.device))
                probs_orig = F.softmax(output_orig, dim=1)
                
                # Soft masked input (type = 1)
                output_soft = self.clf_model(soft_tensor, torch.tensor([1]).to(self.device))
                probs_soft = F.softmax(output_soft, dim=1)
                
                # Attention masked input (type = 1)
                output_att = self.clf_model(att_tensor, torch.tensor([1]).to(self.device))
                probs_att = F.softmax(output_att, dim=1)
                
                # Adaptive ensemble weighting
                conf_orig = probs_orig.max().item()
                conf_soft = probs_soft.max().item()
                conf_att = probs_att.max().item()
                
                total_conf = conf_orig + conf_soft + conf_att
                if total_conf == 0:
                    w_orig = w_soft = w_att = 1.0 / 3.0
                else:
                    w_orig = conf_orig / total_conf
                    w_soft = conf_soft / total_conf
                    w_att = conf_att / total_conf
                
                # Ensemble prediction
                ensemble_probs = w_orig * probs_orig + w_soft * probs_soft + w_att * probs_att
                pred_class = ensemble_probs.argmax(1).item()
                confidence = ensemble_probs.max(1)[0].item()
                
                # Convert to numpy for storage
                ensemble_probs_np = ensemble_probs.cpu().numpy()[0]
            
            # Step 7: Save all images for Excel (no GradCAM)
            original_path = self._save_image_for_excel(image_id, cv2.resize(original_img, (256, 256)), 'original')
            mask_pred_path = self._save_image_for_excel(image_id, mask_pred, 'mask_pred')
            
            mask_gt_path_saved = None
            if mask_gt is not None:
                mask_gt_resized = cv2.resize(mask_gt, (256, 256))
                mask_gt_path_saved = self._save_image_for_excel(image_id, mask_gt_resized, 'mask_gt')
            
            # Clean up tensors
            del original_tensor, soft_tensor, att_tensor
            del probs_orig, probs_soft, probs_att, ensemble_probs
            cleanup_gpu_memory()
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': pred_class,
                'probabilities': ensemble_probs_np,
                'confidence': confidence,
                'processing_time': processing_time,
                'error': None,
                'original_path': original_path,
                'mask_gt_path': mask_gt_path_saved,
                'mask_pred_path': mask_pred_path
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'prediction': None,
                'probabilities': None,
                'confidence': None,
                'processing_time': processing_time,
                'error': str(e),
                'original_path': None,
                'mask_gt_path': None,
                'mask_pred_path': None
            }
    
    def generate_excel_report(self):
        """Generate comprehensive Excel report with embedded images"""
        print("\n" + "="*60)
        print("GENERATING EXCEL REPORT WITH EMBEDDED IMAGES")
        print("="*60)
        
        # Create workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "CXR Pipeline Results"
        
        # Define headers (removed GradCAM)
        headers = [
            'ID Gambar',
            'Label Ground Truth',
            'Label Prediksi',
            'Mask Ground Truth',
            'Mask Prediksi',
            'Original X-Ray'
        ]
        
        # Style headers
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=12)
        
        for col_num, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num)
            cell.value = header
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Set column widths (removed GradCAM column)
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 30
        ws.column_dimensions['E'].width = 30
        ws.column_dimensions['F'].width = 30
        
        # Process each result
        for idx, result in enumerate(tqdm(self.excel_data, desc="Writing to Excel")):
            row_num = idx + 2
            
            # Set row height for images
            ws.row_dimensions[row_num].height = 150
            
            # Column A: ID Gambar
            ws.cell(row=row_num, column=1, value=result['image_id'])
            ws.cell(row=row_num, column=1).alignment = Alignment(horizontal='center', vertical='center')
            
            # Column B: Label Ground Truth
            gt_label = "TB" if result['ground_truth'] == 1 else "Normal"
            cell_b = ws.cell(row=row_num, column=2, value=gt_label)
            cell_b.alignment = Alignment(horizontal='center', vertical='center')
            cell_b.font = Font(bold=True, size=11)
            
            # Column C: Label Prediksi
            if result['prediction'] is not None:
                pred_label = "TB" if result['prediction'] == 1 else "Normal"
                cell_c = ws.cell(row=row_num, column=3, value=pred_label)
                cell_c.alignment = Alignment(horizontal='center', vertical='center')
                cell_c.font = Font(bold=True, size=11)
                
                # Color code: Green if correct, Red if wrong
                if result['prediction'] == result['ground_truth']:
                    cell_c.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                else:
                    cell_c.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            else:
                cell_c = ws.cell(row=row_num, column=3, value="ERROR")
                cell_c.alignment = Alignment(horizontal='center', vertical='center')
                cell_c.fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
            
            # Column D: Mask Ground Truth
            if result.get('mask_gt_path') and os.path.exists(result['mask_gt_path']):
                img_gt = self._prepare_image_for_excel(result['mask_gt_path'])
                if img_gt:
                    ws.add_image(img_gt, f'D{row_num}')
            
            # Column E: Mask Prediksi
            if result.get('mask_pred_path') and os.path.exists(result['mask_pred_path']):
                img_pred = self._prepare_image_for_excel(result['mask_pred_path'])
                if img_pred:
                    ws.add_image(img_pred, f'E{row_num}')
            
            # Column F: Original X-Ray
            if result.get('original_path') and os.path.exists(result['original_path']):
                img_orig = self._prepare_image_for_excel(result['original_path'])
                if img_orig:
                    ws.add_image(img_orig, f'F{row_num}')
        
        # Save workbook
        excel_path = self.results_dir / "comprehensive_results_with_images.xlsx"
        wb.save(excel_path)
        print(f"✓ Excel report saved: {excel_path}")
        print(f"  Total rows: {len(self.excel_data)}")
        print(f"  File size: {os.path.getsize(excel_path) / (1024*1024):.2f} MB")
        
        return excel_path
    
    def run_batch_testing(self, max_images=None):
        """
        Run batch testing on the entire dataset
        
        Args:
            max_images: Maximum number of images to process (None for all)
        """
        print("\n" + "="*60)
        print("RUNNING BATCH TESTING")
        print("="*60)
        
        image_dir, mask_dir = self.load_metadata()
        
        # Filter metadata to only include images that exist
        valid_metadata = []
        for _, row in self.metadata.iterrows():
            image_path = image_dir / f"{row['id']}.png"
            if image_path.exists():
                valid_metadata.append(row)
        
        if max_images:
            valid_metadata = valid_metadata[:max_images]
            print(f"Testing limited to {max_images} images")
        
        print(f"Processing {len(valid_metadata)} images...")
        
        # Process images with progress bar
        for i, row in enumerate(tqdm(valid_metadata, desc="Processing images")):
            image_path = image_dir / f"{row['id']}.png"
            mask_path = mask_dir / f"{row['id']}.png"
            ground_truth = row['ptb']
            
            # Run inference
            result = self.inference_single_image(image_path, mask_path)
            
            # Store results for metrics
            self.test_results['image_ids'].append(row['id'])
            self.test_results['ground_truth'].append(ground_truth)
            self.test_results['predictions'].append(result['prediction'])
            self.test_results['probabilities'].append(result['probabilities'])
            self.test_results['processing_times'].append(result['processing_time'])
            
            if result['error']:
                self.test_results['errors'].append({
                    'image_id': row['id'],
                    'error': result['error']
                })
            
            # Store data for Excel report (no GradCAM)
            self.excel_data.append({
                'image_id': row['id'],
                'ground_truth': ground_truth,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'original_path': result['original_path'],
                'mask_gt_path': result['mask_gt_path'],
                'mask_pred_path': result['mask_pred_path']
            })
            
            # Periodic memory cleanup
            if (i + 1) % 50 == 0:
                cleanup_gpu_memory()
                print(f"\nProcessed {i + 1}/{len(valid_metadata)} images")
        
        print(f"\n✓ Batch testing completed")
        print(f"  Total images processed: {len(valid_metadata)}")
        print(f"  Errors encountered: {len(self.test_results['errors'])}")
        print(f"  Average processing time: {np.mean(self.test_results['processing_times']):.3f}s per image")
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("\n" + "="*60)
        print("CALCULATING METRICS")
        print("="*60)
        
        # Filter out failed predictions
        valid_indices = [i for i, pred in enumerate(self.test_results['predictions']) if pred is not None]
        
        if len(valid_indices) == 0:
            raise ValueError("No valid predictions to evaluate")
        
        y_true = [self.test_results['ground_truth'][i] for i in valid_indices]
        y_pred = [self.test_results['predictions'][i] for i in valid_indices]
        y_proba = np.array([self.test_results['probabilities'][i] for i in valid_indices])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Class-specific metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # AUC-ROC
        if y_proba.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        
        self.metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'auc_roc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=['Normal', 'TB'])
        }
        
        # Print metrics
        print(f"Performance Metrics (n={len(valid_indices)}):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  AUC-ROC:   {roc_auc:.4f}")
        
        print(f"\nPer-Class Metrics:")
        class_names = ['Normal', 'TB']
        for i, name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"  {name}:")
                print(f"    Precision: {precision_per_class[i]:.4f}")
                print(f"    Recall:    {recall_per_class[i]:.4f}")
                print(f"    F1-Score:  {f1_per_class[i]:.4f}")
        
        return self.metrics
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'TB'], 
                   yticklabels=['Normal', 'TB'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        if self.metrics['fpr'] is not None and self.metrics['tpr'] is not None:
            plt.plot(self.metrics['fpr'], self.metrics['tpr'], 
                    color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {self.metrics["auc_roc"]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
        else:
            plt.text(0.5, 0.5, 'ROC Curve\nNot Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
            plt.title('ROC Curve', fontsize=14, fontweight='bold')
        
        # 3. Metrics Bar Plot
        ax3 = plt.subplot(2, 3, 3)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            self.metrics['accuracy'],
            self.metrics['precision_weighted'],
            self.metrics['recall_weighted'],
            self.metrics['f1_weighted']
        ]
        bars = plt.bar(metrics_names, metrics_values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        plt.ylim(0, 1)
        plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Per-Class Metrics
        ax4 = plt.subplot(2, 3, 4)
        class_names = ['Normal', 'TB']
        x = np.arange(len(class_names))
        width = 0.25
        
        if len(self.metrics['precision_per_class']) >= 2:
            plt.bar(x - width, self.metrics['precision_per_class'], width, 
                   label='Precision', alpha=0.8)
            plt.bar(x, self.metrics['recall_per_class'], width, 
                   label='Recall', alpha=0.8)
            plt.bar(x + width, self.metrics['f1_per_class'], width, 
                   label='F1-Score', alpha=0.8)
        
        plt.ylabel('Score')
        plt.title('Per-Class Performance', fontsize=14, fontweight='bold')
        plt.xticks(x, class_names)
        plt.legend()
        plt.ylim(0, 1)
        
        # 5. Processing Time Distribution
        ax5 = plt.subplot(2, 3, 5)
        valid_times = [t for t in self.test_results['processing_times'] if t is not None]
        plt.hist(valid_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(valid_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_times):.3f}s')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Processing Time Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 6. Error Analysis
        ax6 = plt.subplot(2, 3, 6)
        total_images = len(self.test_results['image_ids'])
        successful_images = len([p for p in self.test_results['predictions'] if p is not None])
        failed_images = total_images - successful_images
        
        sizes = [successful_images, failed_images]
        labels = ['Successful', 'Failed']
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Processing Success Rate\n(Total: {total_images} images)', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot (convert Path to string to avoid path issues)
        save_path = str(self.results_dir / 'performance_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to: {save_path}")
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("GENERATING REPORT")
        print("="*60)
        
        report = {
            'test_summary': {
                'total_images': len(self.test_results['image_ids']),
                'successful_predictions': len([p for p in self.test_results['predictions'] if p is not None]),
                'failed_predictions': len(self.test_results['errors']),
                'average_processing_time': float(np.mean([t for t in self.test_results['processing_times'] if t is not None])),
                'total_processing_time': float(np.sum([t for t in self.test_results['processing_times'] if t is not None]))
            },
            'performance_metrics': {
                'accuracy': float(self.metrics['accuracy']),
                'precision_weighted': float(self.metrics['precision_weighted']),
                'recall_weighted': float(self.metrics['recall_weighted']),
                'f1_weighted': float(self.metrics['f1_weighted']),
                'auc_roc': float(self.metrics['auc_roc']) if self.metrics['auc_roc'] else None,
                'precision_per_class': [float(x) for x in self.metrics['precision_per_class']],
                'recall_per_class': [float(x) for x in self.metrics['recall_per_class']],
                'f1_per_class': [float(x) for x in self.metrics['f1_per_class']]
            },
            'confusion_matrix': self.metrics['confusion_matrix'].tolist(),
            'classification_report': self.metrics['classification_report'],
            'errors': self.test_results['errors'],
            'test_config': {
                'device': str(self.device),
                'dataset_path': str(self.dataset_path),
                'metadata_path': str(self.metadata_path),
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save detailed results
        results_file = self.results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV with detailed predictions
        detailed_results = pd.DataFrame({
            'image_id': self.test_results['image_ids'],
            'ground_truth': self.test_results['ground_truth'],
            'prediction': self.test_results['predictions'],
            'confidence': [p[1] if p is not None else None for p in self.test_results['probabilities']],
            'prob_normal': [p[0] if p is not None else None for p in self.test_results['probabilities']],
            'prob_tb': [p[1] if p is not None else None for p in self.test_results['probabilities']],
            'processing_time': self.test_results['processing_times']
        })
        
        csv_file = self.results_dir / 'detailed_results.csv'
        detailed_results.to_csv(csv_file, index=False)
        
        # Generate text report
        text_report = f"""
CXR Pipeline Testing Report
===========================

Test Configuration:
  - Dataset: {self.dataset_path}
  - Device: {self.device}
  - Test Date: {report['test_config']['test_timestamp']}

Test Summary:
  - Total Images: {report['test_summary']['total_images']}
  - Successful Predictions: {report['test_summary']['successful_predictions']}
  - Failed Predictions: {report['test_summary']['failed_predictions']}
  - Success Rate: {report['test_summary']['successful_predictions']/report['test_summary']['total_images']*100:.1f}%
  - Average Processing Time: {report['test_summary']['average_processing_time']:.3f} seconds
  - Total Processing Time: {report['test_summary']['total_processing_time']:.1f} seconds

Performance Metrics:
  - Accuracy:  {report['performance_metrics']['accuracy']:.4f}
  - Precision: {report['performance_metrics']['precision_weighted']:.4f}
  - Recall:    {report['performance_metrics']['recall_weighted']:.4f}
  - F1-Score:  {report['performance_metrics']['f1_weighted']:.4f}
  - AUC-ROC:   {report['performance_metrics']['auc_roc']:.4f if report['performance_metrics']['auc_roc'] else 'N/A'}

Per-Class Performance:
  Normal:
    - Precision: {report['performance_metrics']['precision_per_class'][0]:.4f}
    - Recall:    {report['performance_metrics']['recall_per_class'][0]:.4f}
    - F1-Score:  {report['performance_metrics']['f1_per_class'][0]:.4f}
  
  TB:
    - Precision: {report['performance_metrics']['precision_per_class'][1]:.4f}
    - Recall:    {report['performance_metrics']['recall_per_class'][1]:.4f}
    - F1-Score:  {report['performance_metrics']['f1_per_class'][1]:.4f}

Confusion Matrix:
  Predicted    Normal    TB
  Normal       {report['confusion_matrix'][0][0]:6d}  {report['confusion_matrix'][0][1]:4d}
  TB           {report['confusion_matrix'][1][0]:6d}  {report['confusion_matrix'][1][1]:4d}

Classification Report:
{report['classification_report']}

Files Generated:
  - test_results.json: Complete test results in JSON format
  - detailed_results.csv: Per-image predictions and probabilities
  - performance_analysis.png: Visualization charts
  - comprehensive_results_with_images.xlsx: Excel report with embedded images
    (ID, GT Label, Pred Label, Mask GT, Mask Pred, Original X-Ray)
  - test_report.txt: This text report
"""
        
        txt_file = self.results_dir / 'test_report.txt'
        with open(txt_file, 'w') as f:
            f.write(text_report)
        
        print(f"✓ Reports generated:")
        print(f"  - {results_file}")
        print(f"  - {csv_file}")
        print(f"  - {txt_file}")
        
        return report

def main():
    """Main function to run the complete testing pipeline"""
    print("CXR Pipeline Comprehensive Testing with Excel Report")
    print("=" * 60)
    
    # Configuration - Update these paths according to your setup
    DATASET_PATH = "J:/Portofolio Diri/BRIN/aplikasi/Dataset/Testing"
    METADATA_PATH = "J:/Portofolio Diri/BRIN/aplikasi/Dataset/Testing/MetaData.csv"
    RESULTS_DIR = "test_results"
    MAX_IMAGES = None  # Set to a number to limit testing, None for all images
    
    try:
        # Initialize tester
        tester = CXRPipelineTester(
            dataset_path=DATASET_PATH,
            metadata_path=METADATA_PATH,
            results_dir=RESULTS_DIR
        )
        
        # Load models
        tester.load_models()
        
        # Run batch testing
        tester.run_batch_testing(max_images=MAX_IMAGES)
        
        # Calculate metrics
        tester.calculate_metrics()
        
         # Generate Excel report with images
        tester.generate_excel_report()

        # Create visualizations
        tester.create_visualizations()
       
       
        
        # Generate comprehensive report
        tester.generate_report()
        
        
        
        print("\n" + "="*60)
        print("TESTING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {RESULTS_DIR}/")
        print("\nGenerated files:")
        print("  1. comprehensive_results_with_images.xlsx - Excel with embedded images")
        print("     (Columns: ID, GT Label, Pred Label, Mask GT, Mask Pred, Original Image)")
        print("  2. performance_analysis.png - Performance visualizations")
        print("  3. test_results.json - Complete results in JSON")
        print("  4. detailed_results.csv - Detailed predictions CSV")
        print("  5. test_report.txt - Text summary report")
        print("\nCheck the generated files for detailed analysis.")
        
    except Exception as e:
        print(f"\n✗ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()