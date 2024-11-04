import numpy as np
import cv2

def read_video(filepath: str):
    cap = cv2.VideoCapture(filepath)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("frame", rgb_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


