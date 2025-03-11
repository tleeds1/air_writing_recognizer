import cv2
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class MouseDrawingRecognizer:
    def __init__(self, model_path='air_writing_model.h5', metadata_path='model_metadata.joblib'):
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model or metadata file not found")
            
        self.model = load_model(model_path)
        metadata = joblib.load(metadata_path)
        self.labels = metadata['labels']
        self.scaler = metadata['scaler']
        self.index_to_label = {idx: label for label, idx in self.labels.items()}
        
        self.max_length = 100
        self.trajectory = []
        self.drawing = False
        self.recording = False
        self.prediction = ""
        self.debug_folder = "debug_output"
        self.preprocessing_method = 4
        
        os.makedirs(self.debug_folder, exist_ok=True)
        
        self.canvas_height, self.canvas_width = 480, 640
        self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        
        self.update_instruction_text()
        
        try:
            self.custom_mean = self.scaler.mean_.copy()
            self.custom_scale = self.scaler.scale_.copy()
        except Exception:
            self.custom_mean = np.zeros(4)
            self.custom_scale = np.ones(4)
    
    def update_instruction_text(self):
        instruction_color = (100, 100, 100)
        cv2.rectangle(self.canvas, (0, self.canvas_height-70), 
                    (self.canvas_width, self.canvas_height), (0, 0, 0), -1)
        
        if self.recording:
            status = "RECORDING - Draw your word/phrase"
            status_color = (0, 0, 255)
            cv2.putText(self.canvas, status, 
                    (self.canvas_width//2 - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(self.canvas, "Press 'S' again to finish and detect", 
                    (self.canvas_width//2 - 180, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, instruction_color, 1)
        else:
            cv2.putText(self.canvas, "Press 'S' to start recording", 
                    (self.canvas_width//2 - 150, self.canvas_height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, instruction_color, 1)
            
            method_names = ["Basic", "Normalized", "Stroke-aware", "Bypass Scaler", "Trainer-Aligned"]
            current_method = method_names[self.preprocessing_method]
            
            cv2.putText(self.canvas, f"Press 'M' to change preprocessing (current: {current_method})", 
                    (20, self.canvas_height-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, instruction_color, 1)
            cv2.putText(self.canvas, "Press 'R' to reset, 'D' to save debug data, 'Q' to quit", 
                    (20, self.canvas_height-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, instruction_color, 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        if not self.recording:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            if self.trajectory and self.trajectory[-1] != (-1, -1):
                self.trajectory.append((-1, -1))
            self.trajectory.append((x, y))
            cv2.circle(self.canvas, (x, y), 5, (0, 255, 0), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.trajectory.append((x, y))
                if len(self.trajectory) >= 2 and self.trajectory[-2] != (-1, -1):
                    cv2.line(self.canvas, self.trajectory[-2], self.trajectory[-1], (0, 255, 0), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.trajectory:
                self.trajectory.append((x, y))
                cv2.circle(self.canvas, (x, y), 5, (0, 255, 0), -1)

    def preprocess_aligned_with_trainer(self, trajectory):
        coords = np.array([pt for pt in trajectory if pt != (-1, -1)], dtype=np.float32)
        
        if len(coords) < 5:
            return None

        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        
        width = max(max_x - min_x, 1e-5)
        height = max(max_y - min_y, 1e-5)
        
        coords_norm = coords.copy()
        coords_norm[:, 0] = (coords[:, 0] - min_x) / width
        coords_norm[:, 1] = (coords[:, 1] - min_y) / height
        
        if len(coords_norm) > self.max_length:
            indices = np.linspace(0, len(coords_norm)-1, self.max_length).astype(int)
            coords_norm = coords_norm[indices]
        else:
            padding = np.zeros((self.max_length - len(coords_norm), 2))
            coords_norm = np.vstack([coords_norm, padding])
        
        features = np.zeros((1, self.max_length, 4))
        features[0, :, 0:2] = coords_norm
        
        velocity = np.zeros((self.max_length, 2))
        velocity[1:] = np.diff(coords_norm, axis=0)
        features[0, :, 2:4] = velocity
        
        try:
            original_shape = features.shape
            features_2d = features.reshape(-1, features.shape[-1])
            scaled_features = self.scaler.transform(features_2d).reshape(original_shape)
            return scaled_features
        except Exception:
            return self.preprocess_bypass_scaler(trajectory)

    def preprocess_bypass_scaler(self, trajectory):
        coords = np.array([pt for pt in trajectory if pt != (-1, -1)], dtype=np.float32)
        
        if len(coords) < 5:
            return None
            
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        width = max(max_x - min_x, 1e-5)
        height = max(max_y - min_y, 1e-5)
        
        coords_norm = coords.copy()
        coords_norm[:, 0] = (coords[:, 0] - min_x) / width 
        coords_norm[:, 1] = (coords[:, 1] - min_y) / height
        
        if len(coords_norm) > self.max_length:
            indices = np.linspace(0, len(coords_norm)-1, self.max_length).astype(int)
            coords_norm = coords_norm[indices]
        else:
            padding = np.zeros((self.max_length - len(coords_norm), 2))
            coords_norm = np.vstack([coords_norm, padding])
            
        features = np.zeros((1, self.max_length, 4))
        features[0, :, 0:2] = coords_norm
        
        velocity = np.zeros((self.max_length, 2))
        velocity[1:] = np.diff(coords_norm, axis=0)
        features[0, :, 2:4] = velocity
        
        features_scaled = np.zeros_like(features)
        
        for i in range(4):
            mean_val = self.custom_mean[i]
            scale_val = self.custom_scale[i]
            features_scaled[0, :, i] = (features[0, :, i] - mean_val) / scale_val
            
        return features_scaled

    def preprocess_trajectory(self, trajectory):
        if not trajectory or len(trajectory) < 5:
            return None
            
        self.original_trajectory = trajectory.copy()
        
        if self.preprocessing_method == 4 or self.preprocessing_method < 3:
            processed = self.preprocess_aligned_with_trainer(trajectory)
            method_name = "trainer_aligned"
        else:
            processed = self.preprocess_bypass_scaler(trajectory)
            method_name = "bypass_scaler"
            
        if processed is not None:
            self.processed_data = processed
            try:
                self.visualize_preprocessing(method_name)
            except Exception:
                pass
            
        return processed

    def visualize_preprocessing(self, method_name):
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        plt.title("Original Drawing")
        
        strokes = []
        current_stroke = []
        
        for pt in self.original_trajectory:
            if pt == (-1, -1):
                if current_stroke:
                    strokes.append(current_stroke)
                    current_stroke = []
            else:
                current_stroke.append(pt)
                
        if current_stroke:
            strokes.append(current_stroke)
            
        for i, stroke in enumerate(strokes):
            if stroke:
                stroke_array = np.array(stroke)
                plt.plot(stroke_array[:, 0], stroke_array[:, 1], '-', label=f'Stroke {i+1}' if i < 5 else None)
                plt.scatter(stroke_array[0, 0], stroke_array[0, 1], c='g', s=50)
                plt.scatter(stroke_array[-1, 0], stroke_array[-1, 1], c='r', s=50)
                
        plt.gca().invert_yaxis()
        if len(strokes) <= 5:
            plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.title(f"Processed Coordinates ({method_name})")
        
        coords = self.processed_data[0, :, 0:2]
        plt.plot(coords[:, 0], coords[:, 1], 'b-')
        
        non_zero = np.where((coords[:, 0] != 0) | (coords[:, 1] != 0))[0]
        if len(non_zero) > 0:
            plt.scatter(coords[0, 0], coords[0, 1], c='g', s=50, label='Start')
            last_idx = non_zero[-1]
            plt.scatter(coords[last_idx, 0], coords[last_idx, 1], c='r', s=50, label='End')
            
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.title("Features After Processing (first 20 points)")
        
        features = self.processed_data[0, :20, :]
        plt.plot(range(20), features[:, 0], 'r-', label='x')
        plt.plot(range(20), features[:, 1], 'g-', label='y')
        plt.plot(range(20), features[:, 2], 'b-', label='vx')
        plt.plot(range(20), features[:, 3], 'y-', label='vy')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.title("Feature Value Distribution")
        
        all_values = self.processed_data.flatten()
        plt.hist(all_values, bins=50)
        plt.xlabel("Feature Values")
        plt.ylabel("Frequency")
        
        stats_text = (
            f"Min: {np.min(all_values):.4f}\n"
            f"Max: {np.max(all_values):.4f}\n"
            f"Mean: {np.mean(all_values):.4f}\n"
            f"Std: {np.std(all_values):.4f}\n"
            f"% Zeros: {100 * np.sum(all_values == 0) / len(all_values):.1f}%"
        )
        plt.annotate(stats_text, xy=(0.7, 0.8), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        filename = os.path.join(self.debug_folder, f"preprocessing_{method_name}.png")
        plt.savefig(filename)
        plt.close()

    def create_comparison_visualization(self):
        method_names = ["Basic", "Normalized", "Stroke-Aware", "Bypass Scaler", "Trainer-Aligned"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        current_method = self.preprocessing_method
        
        for i, method in enumerate(range(5)):
            if i >= len(axes):
                break
                
            self.preprocessing_method = method
            processed = self.preprocess_trajectory(self.trajectory)
            
            if processed is not None:
                axes[i].set_title(f"{method_names[i]} Method")
                
                coords = processed[0, :, 0:2]
                axes[i].plot(coords[:, 0], coords[:, 1], 'b-')
                
                non_zero = np.where((coords[:, 0] != 0) | (coords[:, 1] != 0))[0]
                if len(non_zero) > 0:
                    axes[i].scatter(coords[0, 0], coords[0, 1], c='g', s=50, label='Start')
                    last_idx = non_zero[-1]
                    axes[i].scatter(coords[last_idx, 0], coords[last_idx, 1], c='r', s=50, label='End')
                    
                axes[i].legend()
                
                try:
                    predictions = self.model.predict(processed, verbose=0)
                    top_indices = np.argsort(predictions[0])[-3:][::-1]
                    
                    results = []
                    for idx in top_indices:
                        label = self.index_to_label.get(idx, f"unknown_{idx}")
                        confidence = predictions[0][idx] * 100
                        results.append(f"{label}: {confidence:.1f}%")
                            
                    if results:
                        axes[i].set_xlabel("\n".join(results[:3]))
                except Exception as e:
                    axes[i].set_xlabel(f"Error: {str(e)}")
        
        self.preprocessing_method = current_method
        
        for i in range(5, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(self.debug_folder, "method_comparison.png")
        plt.savefig(filename)
        plt.close()

    def predict_drawing(self):
        if not self.trajectory or len(self.trajectory) < 5:
            return "Drawing too short"
            
        processed = self.preprocess_trajectory(self.trajectory)
        if processed is None:
            return "No valid drawing"
        
        try:
            predictions = self.model.predict(processed, verbose=0)
            
            top10_indices = np.argsort(predictions[0])[-10:][::-1]
            
            max_confidence = np.max(predictions[0]) * 100
            if max_confidence < 20:
                return "Low confidence in predictions"
                
            top_indices = top10_indices[:3]
            
            results = []
            for idx in top_indices:
                label = self.index_to_label.get(idx, f"unknown_{idx}")
                confidence = predictions[0][idx] * 100
                results.append(f"{label} ({confidence:.1f}%)")
            
            if not results:
                return "Error: Couldn't map prediction to label"
                
            return ", ".join(results[:2])
            
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        window_name = "Air Writing Recognition"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.preprocessing_method = 4
        
        while True:
            display = self.canvas.copy()
            
            if self.prediction:
                cv2.putText(display, "Prediction: " + self.prediction, 
                            (10, self.canvas_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 255), 2)
            
            method_names = ["Basic", "Normalized", "Stroke-aware", "Bypass Scaler", "Trainer-Aligned"]
            method_name = method_names[self.preprocessing_method]
            cv2.putText(display, f"Method: {method_name}", 
                        (self.canvas_width - 170, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if not self.recording:
                    self.recording = True
                    self.trajectory = []
                    self.prediction = ""
                    self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                    self.update_instruction_text()
                else:
                    self.recording = False
                    if self.trajectory:
                        self.prediction = self.predict_drawing()
                    self.update_instruction_text()
            elif key == ord('r'):
                self.recording = False
                self.drawing = False
                self.trajectory = []
                self.prediction = ""
                self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                self.update_instruction_text()
            elif key == ord('m'):
                self.preprocessing_method = (self.preprocessing_method + 1) % 5
                self.update_instruction_text()
            elif key == ord('d'):
                self.save_debug_data()
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def save_debug_data(self):
        if not self.trajectory:
            return
            
        np.save(os.path.join(self.debug_folder, "raw_trajectory.npy"), np.array(self.trajectory))
        
        current_method = self.preprocessing_method
        
        for method in range(5):
            self.preprocessing_method = method
            try:
                self.preprocess_trajectory(self.trajectory)
            except Exception:
                pass
        
        self.preprocessing_method = current_method
        
        try:
            self.create_comparison_visualization()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        recognizer = MouseDrawingRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")