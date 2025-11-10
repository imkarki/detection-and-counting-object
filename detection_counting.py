from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import messagebox

# Load YOLO model
model = YOLO("yolov8n.pt")

# Function to open camera and start detection
def start_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not found!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, verbose=False)
        bottle_count = 0

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                conf = float(box.conf)

                if class_name == "bottle" and conf > 0.5:
                    bottle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Bottle {bottle_count} ({conf:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 2)

        cv2.putText(frame, f"Total Bottles: {bottle_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)

        cv2.imshow("object Detection & Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# GUI window setup
root = tk.Tk()
root.title("Object Detection App")
root.geometry("300x200")

# Label and button
tk.Label(root, text="Bottle Detection System", font=("Arial", 14, "bold")).pack(pady=20)
start_btn = tk.Button(root, text="Click Me to Start Detection", font=("Arial", 12),
                      bg="#4CAF50", fg="white", command=start_detection)
start_btn.pack(pady=20)

# Run GUI
root.mainloop()
