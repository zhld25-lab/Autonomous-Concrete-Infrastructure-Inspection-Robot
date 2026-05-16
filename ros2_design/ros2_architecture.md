# ROS2-Ready Architecture

## Project Purpose

This project is a laptop-runnable prototype for autonomous concrete infrastructure inspection. It combines a vision-based crack detector with a reinforcement-learning inspection planner and documents how the system can be converted into ROS2 nodes later.

This repository does not claim to be a fully deployed robot system. The current version focuses on model development, planning simulation, and ROS2-ready system design.

## Why ROS2 Is Useful

ROS2 is useful for robot systems because it separates robot behavior into modular nodes that communicate through typed topics. This makes it easier to connect sensors, perception models, planners, and robot controllers while keeping each component independently testable.

For this prototype, ROS2 would help future deployment by:

- Connecting a live camera stream to the crack detector.
- Publishing crack detection results to a planner.
- Sending planner decisions to a robot controller.
- Supporting simulation in tools such as Gazebo before field testing.

## Node Descriptions

### camera_node

The camera node represents a real or simulated camera mounted on a robot, drone, or inspection platform. Its responsibility is to publish concrete surface images.

### crack_detection_node

The crack detection node subscribes to camera images and runs the trained PyTorch crack classifier. It publishes a crack/no-crack prediction and crack probability.

### rl_planner_node

The RL planner node subscribes to crack detection results and updates a risk map of the inspected surface. It uses the inspection planner to decide which location or action should be inspected next.

### robot_control_node

The robot control node receives planner actions and converts them into low-level velocity commands. In the current version this node is design-level only and is not connected to physical motors.

## Topic Descriptions

### /camera/image_raw

Publishes raw RGB camera images from the inspection platform.

### /crack_detection/result

Publishes crack detection output, such as predicted class and crack probability.

### /inspection_planner/action

Publishes the planner's next inspection action or waypoint decision.

### /cmd_vel

Publishes robot velocity commands. This topic is included for future ROS2 deployment and is not used by the laptop-only prototype.

## Current Scope

This is a ROS2-ready design, not a full ROS2 deployment yet. The repository provides reusable Python modules that can later be wrapped as ROS2 nodes.

## Future Work

- Convert Python modules into ROS2 nodes.
- Test the inspection workflow in Gazebo.
- Connect the system to a mobile robot or drone.
- Add a real-time camera stream.
- Add safety checks and operator override tools before any real-world testing.

