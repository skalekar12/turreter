import cv2
import numpy as np

# Global variables
selecting_object = False
init_tracking = False
topleft = (0, 0)
bottomright = (0, 0)
track_window = None
template = None

# Mouse callback function
def onMouse(event, x, y, flags, param):
    global selecting_object, topleft, bottomright, init_tracking, track_window, template

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_object = True
        topleft = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_object = False
        bottomright = (x, y)
        init_tracking = True
        track_window = (topleft[0], topleft[1],
                        bottomright[0] - topleft[0],
                        bottomright[1] - topleft[1])
        template = frame[topleft[1]:bottomright[1], topleft[0]:bottomright[0]]

def calculate_servo_values(frame_width, frame_height, bbox):
    center_x = bbox[0] + bbox[2] / 2
    center_y = bbox[1] + bbox[3] / 2
    servo_x = np.interp(center_x, [0, frame_width], [0, 180])
    servo_y = np.interp(center_y, [0, frame_height], [0, 180])
    return int(servo_x), int(servo_y)

if __name__ == '__main__':
    # Set up face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam video feed
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open video")
        exit()

    # Create a window and set mouse callback
    cv2.namedWindow('Tracking')
    cv2.setMouseCallback('Tracking', onMouse)

    while True:
        ok, frame = video.read()
        if not ok:
            print('Failed to read frame from video stream')
            break

        frame_height, frame_width = frame.shape[:2]

        if not init_tracking:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw rectangles around all detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if selecting_object:
                cv2.rectangle(frame, topleft, bottomright, (255, 0, 0), 2)

        else:
            # Tracking mode
            if template is not None and track_window is not None:
                # Perform template matching
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Update tracking window
                track_window = (max_loc[0], max_loc[1], track_window[2], track_window[3])

                # Draw rectangle around tracked object
                cv2.rectangle(frame, max_loc, (max_loc[0] + track_window[2], max_loc[1] + track_window[3]), (255, 0, 0), 2)

                # Calculate and display servo values
                servo_x, servo_y = calculate_servo_values(frame_width, frame_height, track_window)
                cv2.putText(frame, f"Servo X: {servo_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Servo Y: {servo_y}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tracking failure detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:  # ESC key
            break
        elif k == 114:  # 'r' key to reset tracking
            init_tracking = False
            selecting_object = False
            track_window = None
            template = None

    video.release()
    cv2.destroyAllWindows()