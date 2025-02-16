# mass-spring-damper-system

A Simulink project that simulates both single and double mass-spring-damper systems. As output, the project plots the position-time graph of the mass(es).



## Run on Terminal

```sh
matlab -nodisplay -nosplash -nodesktop -r "run('param.m');exit;"
```



# **SNN-PID Experiments with MATLAB and Docker**

## **Overview**
This section explains how to run Spiking Neural Network-based PID control in MATLAB, integrated with Docker and ROS.

---

## **Tips**

### **1. Connecting to the Running Docker Container**
Ensure the Docker container is running before executing the command. List the running containers with:
```sh
docker ps
```
If the correct container is found but is not running, start it using:
```sh
docker start [container_name]
```

Now, execute the **Python script** inside the Docker container from MATLAB:
```matlab
[status, cmdout] = system('docker exec distracted_kapitsa python /workspace/src/snn_pid_controller/scripts/snn_matlab.py');
```

---

### **2. Verify Execution Output**
After execution, check the output in MATLAB:

```matlab
disp(cmdout);
```

### **3. Check the Status Code**
```matlab
disp(['Status: ', num2str(status)]);
```
- If the status code is `0`, the command executed successfully.
- If the status code is non-zero, verify the command syntax or check the environment configuration inside the Docker container.

---

## **Passing Arguments to the Python Script**
You can pass **current_angle** and **target_angle** as arguments from MATLAB to the Python script inside Docker:

### **Method 1: Using `sys.path.append`**
```matlab
current_angle = 10;
target_angle = 5;

command = sprintf('docker exec distracted_kapitsa python -c "import sys; sys.path.append(''/workspace/src/snn_pid_controller/scripts''); import snn_matlab; print(snn_matlab.run_simulation_step(%d, %d))"', current_angle, target_angle);
[status, cmdout] = system(command);
disp(['Output from Python script: ', cmdout]);
```

### **Method 2: Direct Import**
```matlab
current_angle = 10;
target_angle = 5;

command = sprintf('docker exec distracted_kapitsa python -c "from snn_matlab import run_simulation_step; print(run_simulation_step(%f, %f))"', current_angle, target_angle);
[status, cmdout] = system(command);
disp(['Output from Python script: ', cmdout]);
```

---

## **Building the C-MEX Interface**
To compile the C-MEX function for calling Python inside Docker:
```matlab
cd
mex call_python_docker.c
```

---

## **Initializing ROS in MATLAB**
1. Set up the ROS master node:
   ```matlab
   rosinit('http://192.168.1.106:11311')
   ```

2. Subscribe to the **SNN output topic** and receive data:
   ```matlab
   snn_sub = rossubscriber('/snn_output', 'std_msgs/Float64');
   snn_msg = receive(snn_sub,10);
   disp(snn_msg.Data);
   ```

---

## **Creating a Simulink Bus for ROS Messages**
To define a **Bus Object** for `std_msgs/Float64` in Simulink:
```matlab
% Create a BusElement for the 'data' field in std_msgs/Float64
elems(1) = Simulink.BusElement;
elems(1).Name = 'data';        % Set field name to 'data'
elems(1).DataType = 'double';  % Set field type as double
elems(1).Dimensions = 1;       % Set dimension as 1 (scalar)

% Create a Simulink.Bus object
std_msgs_Float64 = Simulink.Bus;
std_msgs_Float64.Elements = elems;

% Assign the Bus object to the MATLAB workspace
assignin('base', 'std_msgs_Float64', std_msgs_Float64);

% Display the Bus Object
std_msgs_Float64
```

---

## **Saving and Managing Simulink Models**
To save and load the Simulink model:
```matlab
simulink.BlockDiagram.saveAs('PID_mass_spring_damper_system.slx', 'pid_matlab.m');
save_system('PID_mass_spring_damper_system'); % Save the model
load_system('PID_mass_spring_damper_system'); % Load the model
```

---

## **Setting ROS Environment Variables**
Ensure MATLAB is correctly configured to communicate with ROS by setting the environment variable:
```matlab
setenv('ROS_MASTER_URI','http://192.168.1.106:11311')
rosinit
```

---

## **Summary**
This setup allows MATLAB to interact with a Docker-based **Spiking Neural Network (SNN) PID Controller**, running Python inside the container while receiving and publishing ROS messages.

---

### **Troubleshooting**
- If `docker exec` fails, ensure the container is running:  
  ```sh
  docker ps
  ```
- If ROS messages are not received, verify the ROS topic list:  
  ```sh
  rostopic list
  ```
- If Simulink fails to receive ROS messages, adjust the **sample time** in Simulink’s configuration.

---

## **Contact**
For any issues or contributions, feel free to reach out.

---

This README provides a structured guide for integrating MATLAB, Docker, and ROS with SNN-PID control. 🚀

