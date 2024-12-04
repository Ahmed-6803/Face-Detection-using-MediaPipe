import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import cv2
import mediapipe as mp
import os
import numpy as np
from datetime import datetime
import csv

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Initialize LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model (if available)
if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")
    print("Loaded pre-trained model.")

# Ensure the attendance log file exists
if not os.path.exists("attendance_log.csv"):
    with open("attendance_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["User ID", "Name", "Timestamp"])

# Function to update attendance log
def log_attendance(user_id, user_name, marked_users):
    if user_id not in marked_users:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("attendance_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([user_id, user_name, timestamp])
        marked_users.add(user_id)
        print(f"Attendance logged: {user_name} ({user_id}) at {timestamp}")
        return True
    else:
        print(f"Attendance already marked for: {user_name} ({user_id})")
        return False

# Function to prepare the dataset and train the model
def train_faces():
    os.makedirs("Detected_Faces", exist_ok=True)

    cap = cv2.VideoCapture(0)
    user_id = simpledialog.askstring("Input", "Enter a unique user ID:")
    user_name = simpledialog.askstring("Input", "Enter the name of the user:")

    if not user_id or not user_name:
        messagebox.showwarning("Input Error", "Both User ID and Name are required.")
        cap.release()
        return

    print(f"Collecting faces for user: {user_name} (ID: {user_id})")
    faces = []
    labels = []
    label_map = {}

    if os.path.exists("label_map.txt"):
        with open("label_map.txt", "r") as f:
            label_map = {int(line.split(":")[0].strip()): line.split(":")[1].strip() for line in f}

    if int(user_id) in label_map:
        messagebox.showerror("Error", "This ID already exists. Please use a unique ID.")
        cap.release()
        return

    label_map[int(user_id)] = user_name
    collected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h),
                                       int(bbox.width * w), int(bbox.height * h))

                face = frame[y:y + height, x:x + width]
                if face is not None and face.size > 0:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    faces.append(gray)
                    labels.append(int(user_id))
                    collected_count += 1

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"Collected: {collected_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Collecting Faces", frame)
        if collected_count >= 150 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if collected_count > 0:
        recognizer.update(faces, np.array(labels))
        recognizer.save("trainer.yml")
        with open("label_map.txt", "w") as f:
            for label, name in label_map.items():
                f.write(f"{label}: {name}\n")
        messagebox.showinfo("Success", "Model trained and saved.")
    else:
        messagebox.showwarning("Training Aborted", "No faces collected.")

    cap.release()
    cv2.destroyAllWindows()

# Function for real-time face recognition
def recognize_faces():
    if not os.path.exists("trainer.yml"):
        messagebox.showerror("Error", "No trained model found. Please train faces first.")
        return

    if os.path.exists("label_map.txt"):
        with open("label_map.txt", "r") as f:
            label_map = {int(line.split(":")[0].strip()): line.split(":")[1].strip() for line in f}
    else:
        messagebox.showerror("Error", "No label map found. Please train faces first.")
        return

    marked_users = set()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h),
                                       int(bbox.width * w), int(bbox.height * h))

                face = frame[y:y + height, x:x + width]
                if face is not None and face.size > 0:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    label, confidence = recognizer.predict(gray)
                    user_name = label_map.get(label, "Unknown")

                    if user_name != "Unknown" and confidence < 80:
                        log_attendance(label, user_name, marked_users)

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{user_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
def create_gui():
    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("400x200")

    tk.Label(root, text="Face Recognition Attendance System", font=("Arial", 16)).pack(pady=10)

    tk.Button(root, text="Train Faces", command=train_faces, width=20).pack(pady=10)
    tk.Button(root, text="Recognize Faces", command=recognize_faces, width=20).pack(pady=10)
    tk.Button(root, text="Exit", command=root.quit, width=20).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
