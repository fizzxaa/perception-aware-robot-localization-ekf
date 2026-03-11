import numpy as np


class RobotEKF:
    def __init__(self):
        # State vector: [x, y, theta]
        self.x = np.array([0.0, 0.0, 0.0])

        # Covariance matrix
        self.P = np.eye(3) * 0.1

        # Process noise
        self.Q = np.eye(3) * 0.01

        # Time step
        self.dt = 0.1

    def predict(self, v, w):
        """
        v = linear velocity
        w = angular velocity
        """

        theta = self.x[2]

        # Motion model
        self.x[0] += v * np.cos(theta) * self.dt
        self.x[1] += v * np.sin(theta) * self.dt
        self.x[2] += w * self.dt

        # Jacobian of motion model
        F = np.array([
            [1, 0, -v * np.sin(theta) * self.dt],
            [0, 1,  v * np.cos(theta) * self.dt],
            [0, 0, 1]
        ])

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        return self.x, self.P

    def update(self, z):
        """
        Measurement update step of EKF
        z = measurement vector [x_meas, y_meas]
        """

        # Measurement matrix
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        # Measurement noise covariance
        R = np.eye(2) * 5

        # Predicted measurement
        z_pred = H @ self.x

        # Innovation (measurement residual)
        y = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
