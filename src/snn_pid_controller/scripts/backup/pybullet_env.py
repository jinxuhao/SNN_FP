import pybullet as p
import pybullet_data
import os
import time

class PyBulletEnv:
    def __init__(self, gui=True):
        """
        Initialize the PyBullet environment.
        :param gui: Set to True to launch PyBullet with a GUI, False for headless mode.
        """
        if gui:
            self.client = p.connect(p.GUI)  # Launch PyBullet GUI
        else:
            self.client = p.connect(p.DIRECT)  # Headless mode

        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Default PyBullet data
        p.setGravity(0, 0, -9.8)  # Set gravity

        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")  # Load a flat plane
        self.robot_id = None  # Placeholder for the robot ID

    def load_robot(self, urdf_path, start_pos=(0, 0, 0.1), start_orientation=(0, 0, 0, 1)):
        """
        Load a robot into the PyBullet environment.
        :param urdf_path: Path to the robot's URDF file.
        :param start_pos: Initial position of the robot (x, y, z).
        :param start_orientation: Initial orientation of the robot as a quaternion.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.robot_id = p.loadURDF(urdf_path, start_pos, p.getQuaternionFromEuler(start_orientation))
        print(f"Robot loaded with ID: {self.robot_id}")

    def step_simulation(self):
        """
        Step the simulation forward by one timestep.
        """
        p.stepSimulation()
        time.sleep(1. / 240.)  # Default timestep

    def reset(self):
        """
        Reset the simulation environment.
        """
        p.resetSimulation()
        self.__init__()

    def disconnect(self):
        """
        Disconnect from the PyBullet simulation.
        """
        p.disconnect()
