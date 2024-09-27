import tkinter as tk
from tkinter import filedialog

import cv2


def start_gui():
    root = tk.Tk()
    root.title("Face Detection")

    def use_webcam():
        import webcam_processing  # Assuming webcam_processing.py is available
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera couldn't access!")
            return
        webcam_processing.process_webcam_feed(cap)

    def use_video():
        video_path = filedialog.askopenfilename(title="Select a Video", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if video_path:
            print(f"Video file selected: {video_path}")
            import video_processing  # Import video_processing.py here
            video_processing.process_video_feed(video_path)

    btn_webcam = tk.Button(root, text="Use Webcam", command=use_webcam)
    btn_webcam.pack(pady=10)

    btn_video = tk.Button(root, text="Use Video", command=use_video)
    btn_video.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()