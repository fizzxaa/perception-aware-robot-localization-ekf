# Perception-Aware Robot Localization using EKF

This project implements a perception-driven localization system for a mobile robot using computer vision and an Extended Kalman Filter (EKF).

## Overview

The goal of this project is to estimate the robot’s trajectory using visual perception combined with probabilistic state estimation.

The system integrates:

- Free space detection from camera images
- Motion prediction using robot kinematics
- State estimation using Extended Kalman Filter
- Real-time trajectory visualization

## Project Structure

perception/
    free_space.py

estimation/
    ekf.py

main.py

README.md

## Technologies Used

- Python
- OpenCV
- NumPy
- Extended Kalman Filter
- Computer Vision

## Future Work

- Integration with ROS2
- Sensor fusion with IMU and LiDAR
- SLAM-based localization
- Deep learning based perception

## Author

Fizza Sayyed  
Robotics and Automation Engineering