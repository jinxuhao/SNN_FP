import pybullet as p
import pybullet_data
import time

def initialize_robot(urdf_path, gui=True, gravity=-9.81):
    """
    初始化 PyBullet 仿真环境并加载 UR3 机器人。

    参数:
    - urdf_path (str): URDF 文件的路径
    - gui (bool): 是否使用 GUI 模式
    - gravity (float): 设置重力加速度

    返回:
    - physics_client (int): PyBullet 的客户端 ID
    - robot_id (int): 机器人 ID
    - num_joints (int): 机器人的关节数
    """
    # 启动仿真环境
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 默认数据路径
    p.setGravity(0, 0, gravity)  # 设置重力

    # 加载 UR3 机器人
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # 获取关节数
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot loaded with {num_joints} joints.")

    # 设置初始姿势（竖直位置：所有关节角度为 0）
    for joint_index in range(num_joints):
        p.resetJointState(robot_id, joint_index, targetValue=0)

    return physics_client, robot_id, num_joints

def set_joint_angle(robot_id, joint_index, target_angle, control_mode=p.POSITION_CONTROL):
    """
    设置机器人某个关节的目标角度。

    参数:
    - robot_id (int): 机器人 ID
    - joint_index (int): 要控制的关节索引
    - target_angle (float): 目标角度（弧度）
    - control_mode: PyBullet 提供的控制模式，默认是 POSITION_CONTROL
    """
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=control_mode,
        targetPosition=target_angle
    )

def main():
    # UR3 URDF 文件路径
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"

    # 初始化机器人到竖直姿势
    physics_client, robot_id, num_joints = initialize_robot(urdf_path)

    # 打印关节信息
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        print(f"Joint[{joint_index}]: {joint_info}")

    # 仿真一段时间以保持竖直姿势
    print("Robot initialized to vertical position. Holding for 3 seconds...")
    for _ in range(int(3 / 0.01)):  # 3 秒，每次步进 0.01 秒
        p.stepSimulation()
        time.sleep(0.01)

    # 控制某个关节角度
    joint_index_to_control = 1  # 控制第一个关节
    target_angle = 1.0  # 目标位置（弧度）

    print(f"Setting joint {joint_index_to_control} to target angle: {target_angle}")
    while True:
        set_joint_angle(robot_id, joint_index_to_control, target_angle)
        p.stepSimulation()
        time.sleep(0.01)

    # 断开仿真
    # p.disconnect()

if __name__ == "__main__":
    main()