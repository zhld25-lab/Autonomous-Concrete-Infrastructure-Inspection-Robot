# ROS2 Node Graph

```text
camera_node
    publishes /camera/image_raw
        v
crack_detection_node
    subscribes /camera/image_raw
    publishes /crack_detection/result
        v
rl_planner_node
    subscribes /crack_detection/result
    publishes /inspection_planner/action
        v
robot_control_node
    subscribes /inspection_planner/action
    publishes /cmd_vel
```

This graph describes the intended ROS2 communication flow for a future deployment. The current repository implements the perception and planning logic as laptop-runnable Python code and notebooks.

