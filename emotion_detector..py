import cv2
import numpy as np
from fer import FER
import tkinter as tk
from PIL import Image, ImageTk
import time

# Initialize the emotion detector (FER model)
detector = FER(mtcnn=False)  # Use Haar Cascade for better detection with glasses

# Dictionary to map emotions to emojis
emotion_emoji = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😣',
    'neutral': '😐',
    'surprise': '😮',
    'disgust': '😣',
    'fear': '😨'
}

# List to store recent emotions for tracking
recent_emotions = []

def is_blurry(image):
    """Check if the image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Laplacian variance: {variance}")  # Debug
    return variance < 5  # Adjusted threshold based on observed values (12–17)

def process_frame(frame):
    """Process a single frame for emotion detection."""
    # Preprocess: Increase brightness and contrast to reduce glasses reflections
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=50)
    
    # Check if the frame is blurry
    if is_blurry(frame):
        return frame, "Warning: Image is blurry, please provide a clearer image."

    # Detect emotions
    try:
        result = detector.detect_emotions(frame)
        print(f"Detection result: {result}")  # Debug
    except Exception as e:
        print(f"Error in face detection: {e}")
        return frame, f"Error: {str(e)}"

    if result:
        # Get the first detected face
        face = result[0]
        emotions = face['emotions']
        # Find the emotion with the highest probability
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        # Get bounding box of the face
        x, y, w, h = face['box']
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display emotion and emoji
        label = f"{dominant_emotion} {emotion_emoji[dominant_emotion]} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Track recent emotions (store last 5)
        recent_emotions.append(dominant_emotion)
        if len(recent_emotions) > 5:
            recent_emotions.pop(0)
        
        # Display recent emotion trend
        trend = f"Recent: {', '.join(recent_emotions)}"
        cv2.putText(frame, trend, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, None
    else:
        return frame, "No face detected."

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")
        self.root.geometry("800x600")  # Set window size

        # Video capture
        self.cap = None
        self.is_running = False

        # GUI elements
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Status: Stopped", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.start_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=5)

        # Update video feed
        self.update_video()

    def start_webcam(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                self.status_label.config(text="Error: Could not open webcam.")
                return
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")

    def stop_webcam(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")
            self.video_label.config(image="")

    def exit_app(self):
        self.stop_webcam()
        self.root.quit()
        self.root.destroy()

    def update_video(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Process the frame
                processed_frame, message = process_frame(frame)
                
                # Display message in status label
                if message:
                    self.status_label.config(text=f"Status: {message}")
                else:
                    self.status_label.config(text="Status: Face detected")

                # Convert OpenCV BGR image to RGB for Tkinter
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Resize for display (optional, adjust to fit GUI)
                frame_rgb = cv2.resize(frame_rgb, (640, 360))
                # Convert to PIL Image
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                # Update video label
                self.video_label.config(image=photo)
                self.video_label.image = photo  # Keep reference to avoid garbage collection

        # Schedule next update
        self.root.after(10, self.update_video)

if __name__ == "__main__":
    # Fix moviepy issue by modifying fer/classes.py
    # Ensure C:\Users\ASUS\Desktop\emotion_detection_project\venv\Lib\site-packages\fer\classes.py has:
    # try:
    #     from moviepy.editor import *
    # except ImportError:
    #     print("Warning: moviepy.editor not found, video processing may be unavailable.")

    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()