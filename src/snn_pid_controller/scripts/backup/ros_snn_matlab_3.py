#!/usr/bin/env python3

import rospy
from rosgraph_msgs.msg import Clock
import time

def main():
    rospy.init_node('clock_publisher', anonymous=True)
    clock_pub = rospy.Publisher('/clock', Clock, queue_size=1000)
    
    rate = rospy.Rate(10)  # 10 Hz 更新频率
    start_time = time.time()  # 记录启动时间

    while not rospy.is_shutdown():
        elapsed_time = time.time() - start_time  # 计算当前时间
        clock_msg = Clock()
        clock_msg.clock.secs = int(elapsed_time)  # 秒部分
        clock_msg.clock.nsecs = int((elapsed_time - int(elapsed_time)) * 1e9)  # 纳秒部分
        clock_pub.publish(clock_msg)  # 发布消息
        print("TEST INPUT main:", clock_msg)
        rate.sleep()

if __name__ == "__main__":
    main()
