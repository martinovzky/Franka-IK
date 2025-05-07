#!/usr/bin/env python3

"""
Franka Research Arm Control with PID & Inverse Kinematics
========================================================
Main script that demonstrates Franka Research robotic arm control with PID 
controllers and inverse kinematics in Isaac Sim.
"""

import numpy as np
import math
import time
import os
import sys

# Import our custom modules from the flattened structure
from controllers.pid_controller import PIDController
from ik.franka_kinematics import FrankaArmKinematics

# These imports will be used when running in Isaac Sim
# Commented out for local development, uncomment when running on the VM
"""
import omni
from omni.isaac.kit import SimulationApp
import omni.isaac.core.utils.numpy as np_utils
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import World
"""

class FrankaArmController:
    """
    High-level controller for a Franka Research arm using PID control and kinematics.
    
    Attributes:
        kinematics: FrankaArmKinematics instance
        pid_controllers: List of PID controllers for each joint
        current_joint_positions: Current joint positions
        target_joint_positions: Target joint positions
    """
    
    def __init__(self, joint_limits=None, pid_params=None):
        """
        Initialize the Franka arm controller.
        
        Args:
            joint_limits: List of (min, max) joint limits for each joint
            pid_params: List of (kp, ki, kd) tuples for PID controllers
        """
        # Initialize kinematics
        self.kinematics = FrankaArmKinematics(joint_limits)
        
        # Default PID parameters if not specified
        if pid_params is None:
            # Different PID parameters for each joint
            pid_params = [
                (100.0, 1.0, 10.0),  # Joint 1
                (100.0, 1.0, 10.0),  # Joint 2
                (100.0, 1.0, 10.0),  # Joint 3
                (100.0, 1.0, 10.0),  # Joint 4
                (50.0, 0.5, 5.0),    # Joint 5
                (50.0, 0.5, 5.0),    # Joint 6
                (20.0, 0.1, 2.0)     # Joint 7
            ]
        
        # Initialize PID controllers for each joint
        self.pid_controllers = []
        for kp, ki, kd in pid_params:
            self.pid_controllers.append(PIDController(kp, ki, kd, output_limits=(-50.0, 50.0)))
        
        # Initialize joint states
        self.current_joint_positions = self.kinematics.joint_angles.copy()
        self.target_joint_positions = self.kinematics.joint_angles.copy()
    
    def set_target_end_effector_position(self, position, orientation=None):
        """
        Set target end-effector position and compute required joint positions.
        
        Args:
            position: Target (x, y, z) position
            orientation: Target orientation (optional)
            
        Returns:
            bool: True if IK solution found, False otherwise
        """
        # Solve inverse kinematics
        solution = self.kinematics.inverse_kinematics_numerical(
            position, 
            orientation, 
            self.current_joint_positions
        )
        
        if solution is not None:
            self.target_joint_positions = solution
            return True
        
        return False
    
    def update_control(self, dt=0.01):
        """
        Update joint control based on PID controllers.
        
        Args:
            dt: Time step
            
        Returns:
            list: Joint torques/forces to apply
        """
        joint_actions = []
        
        for i in range(self.kinematics.num_joints):
            # Compute control action using PID
            control_action = self.pid_controllers[i].compute(
                self.target_joint_positions[i],
                self.current_joint_positions[i],
                dt
            )
            
            joint_actions.append(control_action)
        
        return joint_actions
    
    def update_joint_positions(self, joint_positions):
        """
        Update the current joint positions.
        
        Args:
            joint_positions: New joint positions
        """
        self.current_joint_positions = joint_positions


def isaac_sim_setup():
    """
    Set up the Isaac Sim environment with a Franka Panda arm.
    This would be used on the VM.
    
    Returns:
        tuple: (world, franka) objects for simulation
    """
    # This is a placeholder for Isaac Sim setup
    # When running on the VM, this would be filled in with actual code
    
    """
    # Create a World instance
    world = World()
    
    # Set physics parameters
    world.set_solver_type("PGS")
    world.set_physics_dt(1.0/120.0)
    world.set_rendering_dt(1.0/60.0)
    
    # Add a ground plane
    world.scene.add_default_ground_plane()
    
    # Add a Franka Panda robot
    franka_asset_path = "/Isaac/Robots/Franka/franka_instanceable.usd"
    franka_position = [0, 0, 0]
    franka_orientation = [0, 0, 0]
    
    # Convert Euler angles to quaternion
    franka_quat = euler_angles_to_quat(franka_orientation)
    
    # Add the robot to the scene
    franka = world.scene.add(
        Robot(
            prim_path="/World/Franka",
            name="franka",
            usd_path=franka_asset_path,
            position=franka_position,
            orientation=franka_quat
        )
    )
    
    # Initialize the simulation
    world.reset()
    
    # Let the simulation run for a bit to let the robot settle
    for _ in range(10):
        world.step()
    
    return world, franka
    """
    
    return None, None


def main():
    """
    Main function to demonstrate Franka Research arm control.
    For local testing, we'll simulate the robot movement without Isaac Sim.
    """
    print("Franka Research Arm Control with PID & Inverse Kinematics")
    
    # Create Franka arm controller
    controller = FrankaArmController()
    
    # Generate a square trajectory in 3D space
    print("Generating square trajectory...")
    
    # Define square corners (x, y, z)
    square_corners = [
        (0.4, 0.0, 0.5),     # Front
        (0.4, 0.4, 0.5),     # Right
        (0.4, 0.4, 0.7),     # Right-up
        (0.4, -0.4, 0.7),    # Left-up
        (0.4, -0.4, 0.5),    # Left
        (0.4, 0.0, 0.5)      # Back to front
    ]
    
    # Simulation parameters
    dt = 0.01  # Time step (10ms)
    duration_per_segment = 3.0  # Time to move between corners
    
    # In a real simulation or with Isaac Sim, this would move the actual robot
    # Here we just print out the trajectory and control signals
    print("\nSimulating Franka arm movement...")
    print("Time | Target Position (x,y,z) | Current Position (x,y,z) | Control Signal")
    print("-" * 100)
    
    # Start with current position
    current_pos = controller.kinematics.forward_kinematics_simple(controller.current_joint_positions)
    
    # For each segment of the trajectory
    for i in range(len(square_corners) - 1):
        start_corner = square_corners[i]
        end_corner = square_corners[i+1]
        
        # Generate trajectory for this segment
        trajectory = controller.kinematics.generate_trajectory(start_corner, end_corner, duration_per_segment, dt)
        
        # Follow the trajectory
        for target_pos, time_point in trajectory:
            # Set the target position
            ik_success = controller.set_target_end_effector_position(target_pos)
            
            if not ik_success:
                print(f"Warning: IK failed for position {target_pos}")
                continue
            
            # Compute control action
            control_signal = controller.update_control(dt)
            
            # In a real simulation, we would apply the control signal and update the position
            # For this local simulation, we'll simulate the movement with a simple model
            for j in range(controller.kinematics.num_joints):
                # Simple dynamics: position changes based on control signal
                controller.current_joint_positions[j] += control_signal[j] * dt * 0.001  # Scaled for simulation
            
            # Get current position after update
            current_pos = controller.kinematics.forward_kinematics_simple(controller.current_joint_positions)
            
            # Print status every few steps
            if int(time_point * 100) % 20 == 0:  # Print every 0.2 seconds
                print(f"{time_point:.2f}s | ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}) | "
                      f"({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}) | "
                      f"{[f'{c:.2f}' for c in control_signal]}")
    
    print("\nTrajectory complete!")


if __name__ == "__main__":
    # Check if we should run with Isaac Sim
    use_isaac_sim = False
    
    if use_isaac_sim:
        # This code would be uncommented when running on the VM
        """
        # Launch Isaac Sim
        simulation_app = SimulationApp({"headless": False})
        
        # Setup the scene and robot
        world, franka = isaac_sim_setup()
        
        # Run the main function
        if world is not None and franka is not None:
            # TODO: Implement Isaac Sim specific control code
            pass
        
        # Cleanup
        simulation_app.close()
        """
        pass
    else:
        # Run local simulation
        main()