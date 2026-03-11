import cv2
import numpy as np
from perception.free_space import FreeSpaceDetector
from estimation.ekf import RobotEKF

detector = FreeSpaceDetector()
ekf = RobotEKF()

# Simulated robot motion
v = 0.5
w = 0.2


# Store robot trajectory
trajectory = []

def draw_uncertainty(frame, px, py, covariance):
    # Use only position covariance
    Pxy = covariance[0:2, 0:2]

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(Pxy)

    # Orientation of ellipse
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Reduce scale so ellipse stays around the robot
    width = int(2 * np.sqrt(abs(eigvals[0])) * 120)
    height = int(2 * np.sqrt(abs(eigvals[1])) * 120)

    cx = px
    cy = py

    cv2.ellipse(
        frame,
        (cx, cy),
        (width, height),
        angle,
        0,
        360,
        (255, 255, 0),
        2
    )

while True:

    frame, binary, measurement = detector.process_frame()

    if frame is None:
        break

    # EKF prediction
    state, covariance = ekf.predict(v, w)

    # EKF correction
    if measurement is not None:
        ekf.update(measurement)

    # Convert normalized state to pixel coordinates
    h, w_img = frame.shape[:2]

    px = int(np.clip(state[0] * w_img, 0, w_img - 1))
    py = int(np.clip(state[1] * h, 0, h - 1))

    trajectory.append((px, py))

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv2.line(frame,
                trajectory[i - 1],
                trajectory[i],
                (255, 0, 0),
                2)

    # Draw robot position
    cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
    # Draw EKF uncertainty ellipse
    draw_uncertainty(frame, px, py, covariance)

    # Display pose text
    text = f"x:{state[0]:.2f} y:{state[1]:.2f} theta:{state[2]:.2f}"
    cv2.putText(frame, text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    cv2.imshow("Original", frame)
    cv2.imshow("Free Space (Binary)", binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


detector.release()