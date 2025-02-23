from pybullet_env import PyBulletEnv
import os

def main():
    # 初始化 PyBullet 环境
    env = PyBulletEnv(gui=True)

    # 加载机器人
    urdf_path = os.path.join(os.path.dirname(__file__), "../pybullet_models/robot.urdf")
    env.load_robot(urdf_path, start_pos=(0, 0, 0.1))

    # 主循环
    try:
        while True:
            env.step_simulation()
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        env.disconnect()

if __name__ == "__main__":
    main()
