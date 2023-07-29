# Lane Following System with Raspberry Pi and Arduino

## Introduction
This repository contains the code and project details for the Lane Following System developed during my internship at the Indian Institute of Technology (IIT) Jammu as part of the RISE-UP 2023 internship program. The primary objective of this project was to create an autonomous vehicle capable of detecting and following lane markings on the road using computer vision techniques and controlling a stepper motor for precise positioning.

## Project Overview
The Lane Following System is an integral component of autonomous vehicles, enabling them to navigate safely on roads by detecting and staying within lane boundaries. The system leverages a Raspberry Pi for image processing and lane detection using OpenCV, and an Arduino to control the stepper motor for adjusting the vehicle's position.

## Tasks and Responsibilities
During my internship, I was responsible for the following tasks:
- Researching and understanding computer vision techniques for lane detection.
- Setting up the Raspberry Pi with a camera module and installing necessary libraries.
- Implementing lane detection algorithms using OpenCV and Python on the Raspberry Pi.
- Configuring the Arduino to control the stepper motor and communicate with the Raspberry Pi.
- Integrating the lane detection algorithm with the stepper motor control to create the Lane Following System.
- Conducting extensive testing and optimization to ensure accurate and smooth lane following.

## Implementation
### Raspberry Pi Setup
I began by setting up the Raspberry Pi and connecting a separate USB camera (C270 HD Webcam). I installed the required software libraries, including OpenCV and NumPy, to enable image processing.

### Lane Detection Algorithm
I researched and implemented various computer vision techniques for lane detection, including edge detection, Hough Transform, and perspective transformation. I processed the camera feed to extract lane markings from the road.

### Stepper Motor Control
I connected the Arduino to the Raspberry Pi using a serial communication protocol. I programmed the Arduino to control the stepper motor, allowing precise adjustments to the vehicle's position.

### Lane Following Integration
I integrated the lane detection algorithm with the stepper motor control to create the Lane Following System. The system uses the lane detection output to determine the vehicle's position relative to the lane center and adjusts the stepper motor accordingly.

## Results and Analysis
The Lane Following System successfully detected lane markings and maintained the vehicle's position within the lane during testing. The integration of the Raspberry Pi's image processing capabilities with the Arduino's motor control provided a smooth and responsive lane-following experience. The system performed well under various lighting conditions and different road surfaces. However, challenges such as noise in the camera feed and varying road conditions required continuous optimization of the lane detection algorithm.

## Conclusion
My internship experience at IIT Jammu provided valuable insights into the fields of computer vision, embedded systems, and robotics. Developing the Lane Following System using Raspberry Pi and Arduino allowed me to apply theoretical concepts to practical applications in autonomous vehicles.

## Contact Information
For any inquiries or collaboration opportunities, please feel free to contact me via:
- Email: dks082001@gmail.com
- LinkedIn [My LinkedIn]([https://www.linkedin.com/in/your_linkedin_profile/](https://www.linkedin.com/in/divyansh-kumar-singh-4423b1212/)https://www.linkedin.com/in/divyansh-kumar-singh-4423b1212/)

