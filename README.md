# SNN-PID Experiments

## Overview
This project explores Spiking Neural Network-based PID control in different simulation environments, including PyBullet and MATLAB.

## Experiment Descriptions
The path to the executable file for the experiment: src/snn_pid_controller/scripts
### 1. Three Target Points Experiment in PyBullet using UR3
- **Script:** `snn_pid_velocity_3_step.py`
- **Description:** This experiment simulates a UR3 robot in PyBullet reaching three predefined target points using an SNN-PID velocity controller.
- **Run in VS Code**
   ```sh
   /usr/bin/python /workspace/src/snn_pid_controller/scripts/snn_pid_velocity_3_step.py
   ```

### 2. Obstacle-Avoidance Experiment in PyBullet using UR3
- **Script:** `snn_pid_velocity_obstacle.py`
- **Description:** This experiment extends the previous setup by introducing obstacles along the motion trajectory. The UR3 robot navigates while avoiding obstacles.
- **Run in VS Code**
   ```sh
   /usr/bin/python /workspace/src/snn_pid_controller/scripts/snn_pid_velocity_obstacle.py
   ```

### 3. Mass-Spring-Damper Model in MATLAB
This experiment tests the SNN-PID controller on a mass-spring-damper system simulated in MATLAB.

#### Steps to Run:
1. **Start ROS Core in VS Code**
   ```sh
   roscore
   ```
2. **Initialize ROS in MATLAB Terminal**
   ```matlab
   rosinit('http://192.168.1.101:11311')
   ```
3. **Run ROS Node in VS Code**
   ```sh
   /usr/bin/python /workspace/src/snn_pid_controller/scripts/ros_snn_matlab_PID.py
   ```
4. **Open and Run the Simulink Model**
   - Open `PID_mass_spring_damper_system.slx` in MATLAB
   - Click **Run** to start the simulation

> **Note:** Due to the frequency mismatch problem between ROS and MATLAB topics, adjust the **sample time** in Simulink Configuration settings accordingly.

## Dependencies
- ROS
- PyBullet
- MATLAB with Simulink
- Python 3


## Installation Issues and Fixes

### Python 3.9 and ROS Conflict
Python 3.9 may have compatibility issues with ROS. To install `pycryptodomex` for Python 3.9:
```sh
python3.9 -m pip install pycryptodomex
```
If the system version of `pycryptodomex` is outdated and incompatible with Python 3.9, force the installation of the latest user-level version:
```sh
python3.9 -m pip install --upgrade --user pycryptodomex
```

### Pinocchio Package Installation
Ensure that all dependencies for `pinocchio` are correctly installed.

### Fixing BindsNET's PyTorch Dependency Issue
BindsNET relies on an outdated PyTorch module `torch._six`, which was removed in PyTorch 1.9.0 and later. To fix this, replace the following import:
```python
from torch._six import container_abcs, string_classes, int_classes
```
with:
```python
import collections.abc as container_abcs
string_classes = (str,)
int_classes = (int,)
```

## VS Code Setup
To set up the development environment in VS Code:
1. **Install Remote Explorer:**
   - Go to Extensions (Ctrl+Shift+X) and search for **Remote - Containers**.
   - Install the extension.

2. **Build Container:**
   - Press **Ctrl+Shift+P** and select **Remote-Containers: Rebuild Container**.

3. **Install Requirements:**
   - Ensure the `requirements.txt` file is present in the project directory.
   - In the VS Code terminal, run:
     ```sh
     pip install -r requirements.txt
     ```


## Contact
For any questions or contributions, feel free to reach out.

