"""
TB Detection Application - Final Fixed Version
==============================================
Aplikasi deteksi TB dari X-ray dengan model ResNet101V2
Berdasarkan script training yang diberikan
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet101V2
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

# Disable GPU jika bermasalah
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncomment untuk force CPU

class TBDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TB Detection - Computer Aided Diagnosis System")
        self.root.geometry("1100x750")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.current_image_path = None
        self.current_image = None
        self.processed_image = None
        
        # Model configuration sesuai training script
        self.model_path = r"D:\SEM 6\Big Data dalam Kesehatan\final-project\gambar\final_xray_classification_model.keras"
        self.fixed_model_path = "fixed_tb_detection_model.keras"  # Jika sudah di-fix
        self.img_size = (400, 400)  # Sesuai training script
        
        # Class labels sesuai training script (dengan typo TURBERCULOSIS)
        self.class_labels = ['COVID', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
        
        # Label to int mapping (untuk consistency)
        self.label_to_int = {label: idx for idx, label in enumerate(sorted(self.class_labels))}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
        
        self.setup_ui()
        self.load_or_create_model()
    
    def setup_ui(self):
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(header_frame, text="🏥 TB Detection System - ResNet101V2", 
                               font=("Arial", 18, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="Computer-Aided Diagnosis for Chest X-Ray Analysis", 
                                  font=("Arial", 10))
        subtitle_label.pack()
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_container, text="Control Panel", padding="15")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 1. Image Upload Section
        upload_frame = ttk.LabelFrame(left_panel, text="1. Select X-Ray Image", padding="10")
        upload_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.upload_btn = ttk.Button(upload_frame, text="📂 Browse Image", 
                                    command=self.browse_image, width=25)
        self.upload_btn.grid(row=0, column=0, pady=5)
        
        self.image_info_label = ttk.Label(upload_frame, text="No image loaded", 
                                         foreground="gray")
        self.image_info_label.grid(row=1, column=0)
        
        # 2. Model Status Section
        model_frame = ttk.LabelFrame(left_panel, text="2. Model Status", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.model_status_label = ttk.Label(model_frame, text="Loading model...", 
                                           font=("Arial", 10, "bold"))
        self.model_status_label.grid(row=0, column=0, pady=5)
        
        model_info_label = ttk.Label(model_frame, text="ResNet101V2 - 4 Classes", 
                                    foreground="gray")
        model_info_label.grid(row=1, column=0)
        
        ttk.Button(model_frame, text="🔄 Reload Model", 
                  command=self.reload_model, width=25).grid(row=2, column=0, pady=5)
        
        # 3. Prediction Section
        predict_frame = ttk.LabelFrame(left_panel, text="3. Generate Prediction", padding="10")
        predict_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.predict_btn = ttk.Button(predict_frame, text="🔍 Analyze X-Ray", 
                                     command=self.predict, state="disabled", width=25)
        self.predict_btn.grid(row=0, column=0, pady=5)
        
        # 4. Results Section
        results_frame = ttk.LabelFrame(left_panel, text="Results & Interpretation", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Results text with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.result_text = tk.Text(text_frame, height=12, width=40, wrap=tk.WORD,
                                  font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", 
                                 command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(left_panel, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(3, weight=1)
        
        # Right panel - Image Display
        right_panel = ttk.LabelFrame(main_container, text="X-Ray Image Viewer", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for image display
        self.canvas = tk.Canvas(right_panel, bg="black", width=600, height=600)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Initial canvas text
        self.canvas.create_text(300, 300, text="No image loaded", 
                               fill="white", font=("Arial", 14), tags="placeholder")
    
    def create_model_architecture(self):
        """Create exact model architecture as in training script"""
        try:
            print("Creating ResNet101V2 model architecture...")
            
            # Base model
            base_model = ResNet101V2(
                weights='imagenet',
                include_top=False,
                input_shape=(400, 400, 3)
            )
            
            # Set base model as trainable (since we'll load trained weights)
            base_model.trainable = True
            
            # Build exact same architecture as training
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(4, activation='softmax', name='predictions')
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Build model
            dummy_input = tf.ones((1, 400, 400, 3))
            _ = model(dummy_input)
            
            print("Model architecture created successfully!")
            return model
            
        except Exception as e:
            print(f"Failed to create model architecture: {e}")
            return None
    
    def load_or_create_model(self):
        """Load model with multiple fallback methods"""
        self.model_status_label.config(text="Loading model...", foreground="orange")
        self.root.update()
        
        # Check if fixed model exists
        if os.path.exists(self.fixed_model_path):
            print(f"Found fixed model at: {self.fixed_model_path}")
            try:
                self.model = tf.keras.models.load_model(self.fixed_model_path, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.model_loaded_success("Fixed model loaded")
                return
            except Exception as e:
                print(f"Failed to load fixed model: {e}")
        
        # Try loading original model
        if os.path.exists(self.model_path):
            print(f"Trying to load original model from: {self.model_path}")
            
            # Method 1: Load with compile=False
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.model_loaded_success("Original model loaded")
                return
            except Exception as e:
                print(f"Method 1 failed: {e}")
                
                # Method 2: Create architecture and load weights
                try:
                    print("Creating architecture and loading weights...")
                    self.model = self.create_model_architecture()
                    if self.model:
                        self.model.load_weights(self.model_path)
                        self.model_loaded_success("Model recreated with weights")
                        return
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    
                    # Method 3: Create fresh model (no weights)
                    try:
                        print("Creating fresh model architecture...")
                        self.model = self.create_model_architecture()
                        if self.model:
                            self.model_loaded_success("Fresh model created (no trained weights)")
                            messagebox.showwarning("Warning", 
                                "Model architecture created but trained weights could not be loaded.\n"
                                "Predictions may not be accurate.\n\n"
                                "Please run the model fix script first.")
                            return
                    except Exception as e3:
                        print(f"Method 3 failed: {e3}")
        
        # If all methods failed
        self.model_status_label.config(text="❌ Model loading failed", foreground="red")
        self.show_model_error()
    
    def model_loaded_success(self, method=""):
        """Handle successful model loading"""
        self.model_status_label.config(text=f"✅ Model ready", foreground="green")
        self.status_var.set(f"Model loaded: {method}")
        
        # Display model info
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "="*45 + "\n")
        self.result_text.insert(tk.END, "MODEL INFORMATION\n")
        self.result_text.insert(tk.END, "="*45 + "\n\n")
        
        self.result_text.insert(tk.END, f"Architecture: ResNet101V2\n")
        self.result_text.insert(tk.END, f"Input shape: (400, 400, 3)\n")
        self.result_text.insert(tk.END, f"Loading method: {method}\n\n")
        
        self.result_text.insert(tk.END, "Classes (4 categories):\n")
        self.result_text.insert(tk.END, "-"*25 + "\n")
        for i, label in enumerate(self.class_labels):
            self.result_text.insert(tk.END, f"{i}: {label}\n")
        
        self.result_text.insert(tk.END, "\n" + "="*45 + "\n")
        self.result_text.insert(tk.END, "Ready for prediction!\n")
        self.result_text.insert(tk.END, "Upload a chest X-ray image to analyze.\n")
    
    def show_model_error(self):
        """Show error message and solution"""
        error_msg = "Model Loading Failed!\n\n"
        error_msg += f"Could not load model from:\n{self.model_path}\n\n"
        error_msg += "To fix this issue:\n\n"
        error_msg += "1. Run the 'recreate_model_architecture.py' script\n"
        error_msg += "2. This will create 'fixed_tb_detection_model.keras'\n"
        error_msg += "3. Restart this application\n\n"
        error_msg += "The fix script will recreate the model architecture\n"
        error_msg += "and properly load the trained weights."
        
        messagebox.showerror("Model Loading Error", error_msg)
        
        # Show instructions in result text
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "❌ MODEL LOADING ERROR\n")
        self.result_text.insert(tk.END, "="*45 + "\n\n")
        self.result_text.insert(tk.END, "Steps to fix:\n\n")
        self.result_text.insert(tk.END, "1. Save and run this script:\n")
        self.result_text.insert(tk.END, "   recreate_model_architecture.py\n\n")
        self.result_text.insert(tk.END, "2. It will create:\n")
        self.result_text.insert(tk.END, "   fixed_tb_detection_model.keras\n\n")
        self.result_text.insert(tk.END, "3. Restart this application\n\n")
        self.result_text.insert(tk.END, "The model uses ResNet101V2 architecture\n")
        self.result_text.insert(tk.END, "with 4 classes: COVID, NORMAL,\n")
        self.result_text.insert(tk.END, "PNEUMONIA, TUBERCULOSIS\n")
    
    def reload_model(self):
        """Reload the model"""
        self.model = None
        self.model_status_label.config(text="Reloading...", foreground="orange")
        self.root.update()
        self.load_or_create_model()
    
    def browse_image(self):
        """Browse and load an image file"""
        file_types = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Chest X-Ray Image",
            filetypes=file_types
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display the selected image"""
        try:
            self.current_image_path = file_path
            
            # Load image using PIL for display
            self.current_image = Image.open(file_path)
            
            # Also load with OpenCV for preprocessing (as in training)
            self.cv_image = cv2.imread(file_path)
            
            # Display image
            self.display_image()
            
            # Update info
            filename = os.path.basename(file_path)
            self.image_info_label.config(text=f"✓ {filename[:30]}...", foreground="green")
            
            # Enable predict button if model is loaded
            if self.model is not None:
                self.predict_btn.config(state="normal")
                self.status_var.set(f"Image loaded: {filename}")
            else:
                self.predict_btn.config(state="disabled")
                self.status_var.set("Image loaded - Model not ready")
            
            # Clear previous results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "IMAGE INFORMATION\n")
            self.result_text.insert(tk.END, "="*45 + "\n\n")
            self.result_text.insert(tk.END, f"File: {filename}\n")
            self.result_text.insert(tk.END, f"Size: {self.current_image.size}\n")
            self.result_text.insert(tk.END, f"Mode: {self.current_image.mode}\n\n")
            
            if self.model:
                self.result_text.insert(tk.END, "Ready for analysis!\n")
                self.result_text.insert(tk.END, "Click 'Analyze X-Ray' to detect abnormalities.\n")
            else:
                self.result_text.insert(tk.END, "⚠️ Model not loaded!\n")
                self.result_text.insert(tk.END, "Please reload the model first.\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def display_image(self):
        """Display the current image on canvas"""
        if self.current_image:
            # Clear canvas
            self.canvas.delete("all")
            
            # Get canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 600
            
            img_width, img_height = self.current_image.size
            
            # Calculate scaling
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0) * 0.95  # 95% to add padding
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            display_image = self.current_image.resize((new_width, new_height), 
                                                     Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Center the image
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
    
    def preprocess_image(self):
        """Preprocess image exactly as in training script"""
        if self.cv_image is None:
            return None
        
        try:
            # Follow exact preprocessing from training
            # Convert BGR to RGB
            image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            self.processed_image = image_batch
            return image_batch
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess image: {str(e)}")
            return None
    
    def predict(self):
        """Perform prediction on the current image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded! Please reload model.")
            return
        
        if self.current_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        
        try:
            self.status_var.set("Analyzing X-ray...")
            self.root.update()
            
            # Preprocess image
            processed_img = self.preprocess_image()
            if processed_img is None:
                return
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_labels[predicted_class_idx]
            
            # Display results
            self.display_results(predictions[0], predicted_class, confidence)
            
            self.status_var.set("Analysis completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
    
    def display_results(self, predictions, predicted_class, confidence):
        """Display prediction results with clinical interpretation"""
        self.result_text.delete(1.0, tk.END)
        
        # Header
        self.result_text.insert(tk.END, "="*45 + "\n")
        self.result_text.insert(tk.END, "ANALYSIS RESULTS\n")
        self.result_text.insert(tk.END, "="*45 + "\n\n")
        
        # Main prediction
        self.result_text.insert(tk.END, f"PRIMARY FINDING: {predicted_class}\n")
        self.result_text.insert(tk.END, f"Confidence: {confidence:.1%}\n\n")
        
        # Confidence bar visualization
        bar_length = 30
        filled = int(confidence * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        self.result_text.insert(tk.END, f"[{bar}] {confidence:.1%}\n\n")
        
        # All probabilities
        self.result_text.insert(tk.END, "Probability Distribution:\n")
        self.result_text.insert(tk.END, "-"*45 + "\n")
        
        # Sort by probability
        prob_indices = np.argsort(predictions)[::-1]
        for idx in prob_indices:
            label = self.class_labels[idx]
            prob = predictions[idx]
            
            # Visual indicator
            if idx == predicted_class_idx:
                marker = ">>>"
                self.result_text.insert(tk.END, f"{marker} ", "highlight")
            else:
                marker = "   "
                self.result_text.insert(tk.END, f"{marker} ")
            
            # Class and probability
            self.result_text.insert(tk.END, f"{label:15} {prob:6.2%}\n")
        
        # Clinical Interpretation
        self.result_text.insert(tk.END, "\n" + "="*45 + "\n")
        self.result_text.insert(tk.END, "CLINICAL INTERPRETATION\n")
        self.result_text.insert(tk.END, "="*45 + "\n\n")
        
        if predicted_class == "NORMAL":
            self.result_text.insert(tk.END, "✅ No significant abnormalities detected\n\n")
            self.result_text.insert(tk.END, "Finding: Clear lung fields\n")
            self.result_text.insert(tk.END, "Recommendation:\n")
            self.result_text.insert(tk.END, "• No immediate action required\n")
            self.result_text.insert(tk.END, "• Continue routine screening\n")
            
        elif predicted_class == "TURBERCULOSIS" or predicted_class == "TUBERCULOSIS":
            self.result_text.insert(tk.END, "⚠️ Tuberculosis indicators detected\n\n")
            self.result_text.insert(tk.END, "Possible findings:\n")
            self.result_text.insert(tk.END, "• Upper lobe infiltrates\n")
            self.result_text.insert(tk.END, "• Cavitary lesions\n")
            self.result_text.insert(tk.END, "• Hilar lymphadenopathy\n\n")
            self.result_text.insert(tk.END, "Recommended actions:\n")
            self.result_text.insert(tk.END, "• Sputum AFB examination\n")
            self.result_text.insert(tk.END, "• GeneXpert MTB/RIF test\n")
            self.result_text.insert(tk.END, "• Clinical correlation required\n")
            
        elif predicted_class == "PNEUMONIA":
            self.result_text.insert(tk.END, "⚠️ Pneumonia patterns detected\n\n")
            self.result_text.insert(tk.END, "Possible findings:\n")
            self.result_text.insert(tk.END, "• Consolidation areas\n")
            self.result_text.insert(tk.END, "• Air bronchograms\n")
            self.result_text.insert(tk.END, "• Pleural effusion\n\n")
            self.result_text.insert(tk.END, "Recommended actions:\n")
            self.result_text.insert(tk.END, "• Complete blood count\n")
            self.result_text.insert(tk.END, "• C-reactive protein test\n")
            self.result_text.insert(tk.END, "• Consider antibiotic therapy\n")
            
        elif predicted_class == "COVID":
            self.result_text.insert(tk.END, "⚠️ COVID-19 patterns detected\n\n")
            self.result_text.insert(tk.END, "Possible findings:\n")
            self.result_text.insert(tk.END, "• Ground-glass opacities\n")
            self.result_text.insert(tk.END, "• Bilateral peripheral distribution\n")
            self.result_text.insert(tk.END, "• Crazy-paving pattern\n\n")
            self.result_text.insert(tk.END, "Recommended actions:\n")
            self.result_text.insert(tk.END, "• RT-PCR confirmation\n")
            self.result_text.insert(tk.END, "• Oxygen saturation monitoring\n")
            self.result_text.insert(tk.END, "• Isolation protocols\n")
        
        # Disclaimer
        self.result_text.insert(tk.END, "\n" + "="*45 + "\n")
        self.result_text.insert(tk.END, "⚠️ IMPORTANT DISCLAIMER\n")
        self.result_text.insert(tk.END, "="*45 + "\n")
        self.result_text.insert(tk.END, "This is a computer-aided diagnostic tool.\n")
        self.result_text.insert(tk.END, "Results should be confirmed by a qualified\n")
        self.result_text.insert(tk.END, "radiologist or physician before treatment.\n")
        
        # Add tag for highlighting
        self.result_text.tag_config("highlight", foreground="blue", font=("Consolas", 9, "bold"))

def main():
    """Main function with error handling"""
    try:
        # Print system info
        print("="*60)
        print("TB DETECTION SYSTEM")
        print("="*60)
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"Keras Version: {tf.keras.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU Available: {len(gpus)} device(s)")
        else:
            print("GPU: Not available (using CPU)")
        print("="*60)
        
        # Create and run app
        root = tk.Tk()
        app = TBDetectionApp(root)
        root.mainloop()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show error dialog
        try:
            messagebox.showerror("Fatal Error", 
                f"Application failed to start:\n{str(e)}\n\n"
                "Please check:\n"
                "1. TensorFlow is installed correctly\n"
                "2. All required packages are installed\n"
                "3. Run the model fix script first")
        except:
            print("Could not show error dialog")

if __name__ == "__main__":
    main()