import cv2
from cvzone.FaceDetectionModule import FaceDetector


# import numpy as np

def process_video_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}.")
        return

    detector = FaceDetector()
    frame_count = 0
    frame_skip = 0

    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Feed", 640, 480)  # Adjust window size as needed

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from video or end of video reached.")
            break

        # Resize image for consistent display
        height, width = img.shape[:2]
        target_width = 640
        target_height = int((target_width / width) * height)
        img_resized = cv2.resize(img, (target_width, target_height))

        if frame_skip > 1 and frame_count % frame_skip == 0:
            img_resized, bboxs = detector.findFaces(img_resized, draw=False)

            # Draw circles and text
            if bboxs:
                for bbox in bboxs:
                    fx, fy = bbox["center"]
                    radius = int(bbox["bbox"][2] / 2)  # Approximate radius for circle
                    cv2.circle(img_resized, (int(fx), int(fy)), radius, (0, 255, 0), 2)  # Green circle
                    cv2.putText(img_resized, "Person Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                cv2.putText(img_resized, "Person Not Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

        # Display the resized image
        cv2.imshow("Feed", img_resized)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
