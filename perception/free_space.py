import cv2
import numpy as np


class FreeSpaceDetector:

    def __init__(self):

        # Open camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        # Optional resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("Camera not detected")
            exit()

        print("Camera initialized")


    def process_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None, None, None


        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        blurred = cv2.GaussianBlur(gray, (5,5), 0)


        # Binary threshold
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )


        # Detect contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        measurement = None


        # If contours detected
        if contours:

            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > 800:

                x, y, w, h = cv2.boundingRect(largest)

                cx = x + w // 2
                cy = y + h // 2

                # Normalize measurement to [0,1] range
                measurement = np.array([
                    cx / frame.shape[1],
                    cy / frame.shape[0]
                ])

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                # Draw centroid
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)


        return frame, binary, measurement


    def release(self):

        self.cap.release()
        cv2.destroyAllWindows()