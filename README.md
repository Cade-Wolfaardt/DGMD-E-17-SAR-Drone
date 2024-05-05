# DGMD E-17 SAR Drone
The adoption of autonomous unmanned aerial vehicle (UAV) technology presents a promising avenue for reducing human exposure to hazardous scenarios, particularly in search and rescue (SAR) missions. Effectively locating disaster victims or targets in rescue missions requires sophisticated UAV capabilities, including navigation and identification in diverse weather conditions. This project focuses on developing a deep reinforcement learning model tailored for path planning, integrating it with a simulation environment, and deploying it on a drone equipped with customized hardware.

The project leveraged a deep Q network for path planning algorithms and explored the capabilities of YOLOv5 for object detection tasks. Simulation environments were created using ArduPilot, enabling rigorous testing and refinement of the developed algorithms. Practical validation involved executing test flights using a DJI Mini 2 drone with a specialized hardware payload. The findings of this project demonstrate the feasibility of these technologies in a proof-of-concept setting, laying a foundation for future advancements in autonomous SAR missions.

## Hardware Configuration

- **DJI Mini 2 Drone:** Selected for its lightweight design and reliability. Integrated with advanced control and data processing hardware.
- **Raspberry Pi 4B:** Serves as the central computing unit running Kali Linux, interfacing with PixHawk for autonomous control and running complex AI models for object detection.
- **PixHawk Mini 6c Flight Controller:** Manages flight parameters and executes commands based on data received from the Raspberry Pi.

## Software and Installation

### Installation of Kali Linux on Raspberry Pi 4

To empower our drone's intelligence and processing capabilities, we install the 64-bit version of Kali Linux. This involves critical steps to ensure a correct and secure setup.

#### Installation Steps

1. **Preparation:**
   - Use a high-speed microSD card with a minimum capacity of 16GB, Class 10 recommended.

2. **Downloading the Image:**
   - Obtain the 64-bit Kali Linux image designed for Raspberry Pi 4 from the official Kali Linux downloads area.

3. **Writing the Image to the microSD Card:**
   - Use the xzcat and dd utilities to write the image to the microSD card.
```bash
   xzcat kali-linux-2024.1-raspberry-pi-arm64.img.xz | sudo dd of=/dev/sdX bs=4M status=progress
```
In this command, replace /dev/sdX with the correct device identifier for your microSD card.

First Boot and Setup:
Insert the microSD card into the Raspberry Pi 4 and follow on-screen instructions to complete the initial setup.
Post-Installation Configuration:
Enable Bluetooth and configure audio output settings.
Network Configuration:
Create a 'wpa_supplicant.conf' file for wireless connectivity.
Installation of YOLOv5 on Raspberry Pi for Object Detection
YOLOv5 on the Raspberry Pi enables real-time object detection, crucial for identifying persons or objects of interest in various terrains and conditions.

Installation Steps

1. Python Installation:
Ensure Python 3.8 or newer is installed on the Raspberry Pi.
```bash
sudo apt update
sudo apt install python3.8 python3-pip
```
2. Setting Up a Virtual Environment:
Create a virtual environment to manage Python dependencies.
```bash
python3 -m venv yolov5-venv
source yolov5-venv/bin/activate
```
3. Cloning the YOLOv5 Repository:
Clone the YOLOv5 repository from GitHub.
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```
4. Installing Dependencies:
```bash
pip install -r requirements.txt
```
5. PyTorch Installation:
Install PyTorch along with torchvision and torchaudio.
```bash
pip install torch torchvision torchaudio
```
6. Running Object Detection:
- Dowload Local RTMP Server of your coice
- run ifconfig | grep "inet " in terminal to get your ip address
- put the link of format rtmp://<IP ADDRESS>/live/<Stream Code> to the drone's RTMP transmission settings and start the transmission.
- Run
  ```bash
  python detect.py --source 'rtmp://IP Address/live/Stream Code'
  ```

Custom iOS App Development Using the DJI SDK
We developed a custom iOS application using the DJI Mobile SDK version 4.16.2, providing extensive customization and control over DJI drones.

SDK Integration Process

1. SDK and UX SDK Installation:
- Integrate DJI Mobile SDK and DJI UX SDK using CocoaPods.
```bash
sudo gem install cocoapods
cd path/to/project
pod install
```
2. App Key Configuration:
- Set up an App Key for communication between the app and DJI products.
For Objective-C Sample App:
```bash
<key>DJISDKAppKey</key>
<string>Your-App-Key-Here</string>
```
For Swift Sample App:
```bash
<key>DJISDKAppKey</key>
<string>Your-App-Key-Here</string>
```
3. Bundle Identifier Update:
- Update the bundle identifier to match the registered one.
4. Using the DJI UX SDK:
- Utilize the DJI UX SDK for UI components.
5. DJIWidget Integration:
- Integrate DJIWidget for video decoding.
```bash
pod 'DJIWidget', '~> 1.6.6'
```
Integration with iOS App and MAVProxy

The iOS app communicates with MAVProxy running on the Raspberry Pi to establish a reliable connection with the drone's flight controller. MAVProxy acts as a bridge, facilitating the exchange of commands and telemetry data between the iOS app and the Pixhawk flight controller.

To initiate the communication, the iOS app establishes a TCP/IP connection with MAVProxy using the network address of the Raspberry Pi. MAVProxy, configured to listen for incoming connections, then relays commands received from the iOS app to the Pixhawk flight controller via the MAVLink protocol. Similarly, telemetry data from the flight controller is transmitted back through MAVProxy to the iOS app, providing real-time feedback on the drone's status and performance.

This integration not only enables seamless control and feedback from the drone's flight controller but also plays a crucial role in enabling autonomous missions in challenging environments. By leveraging MAVProxy as an intermediary communication layer, the iOS app can execute complex flight maneuvers and receive telemetry updates with high precision and reliability.

To set up the integration, ensure that MAVProxy is installed and running on the Raspberry Pi with the appropriate configurations to listen for incoming connections. Then, configure the iOS app to establish a TCP/IP connection with the Raspberry Pi's network address and port used by MAVProxy. Once the connection is established, the iOS app can send commands to control the drone and receive telemetry data for monitoring and analysis.
