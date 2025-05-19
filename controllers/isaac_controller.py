#!/usr/bin/env python3

"""
Isaac Differential IK Controller Wrapper
=======================================
Wrapper for Isaac Lab's DifferentialIKController which enables direct control of
the end-effector position, eliminating the need for separate IK solving.
"""

import numpy as np

class IsaacDiffIKController:
    """
    Wrapper around Isaac Lab's DifferentialIKController for end-effector control.
    This controller directly computes joint efforts to move the end-effector to
    a target Cartesian position and orientation.
    """
    def __init__(self, robot, damping=0.05, command_type="velocity"):
        """
        Initialize the Differential IK controller wrapper.
        
        Args:
            robot: FrankaRobot instance from Isaac Lab
            damping: Damping factor for the pseudo-inverse calculation
            command_type: Type of command ("velocity" or "position")
        """
        self.robot = robot
        self.damping = damping
        self.command_type = command_type
        
        # Get joint information
        self.joint_names = robot.get_joint_names()
        self.num_joints = len(self.joint_names)
        
        # Try to initialize Isaac Lab's DifferentialIKController
        try:
            from omni.isaac.lab.controllers import DifferentialIKController
            
            # Create the differential IK controller
            self.controller = DifferentialIKController(
                name="diff_ik_controller",
                robot_articulation=robot,
                end_effector_link_name=robot.end_effector_prim_path,
                joints_indices=list(range(self.num_joints)),
                damping=self.damping,
                command_type=self.command_type
            )
            
            self.using_isaac_controller = True
            print(f"Using Isaac Lab's DifferentialIKController (command_type: {command_type})")
            
        except ImportError:
            # Fall back to a simple custom solution if Isaac Lab is not available
            self.using_isaac_controller = False
            from controllers.pid_controller import PIDController
            from ik.inverse_kinematics_solver import IKSolver
            
            # Create IK solver and PID controllers for each joint
            self.ik_solver = IKSolver(prim_paths_expr=robot.prim_path)
            self.pid_controllers = [
                PIDController(kp=100.0, ki=0.0, kd=20.0)
                for _ in range(self.num_joints)
            ]
            print("Using custom IK + PID control (Isaac Lab's DifferentialIKController not available)")
        
        # Store current targets
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w,x,y,z)
        
    def set_targets(self, position=None, orientation=None):
        """
        Set the target position and/or orientation for the end-effector.
        
        Args:
            position: Target position (x, y, z) or None to keep current
            orientation: Target orientation as quaternion (w, x, y, z) or None to keep current
        """
        if position is not None:
            self.target_position = np.array(position)
        
        if orientation is not None:
            self.target_orientation = np.array(orientation)
    
    def compute_and_apply_control(self, dt):
        """
        Compute and apply control to move the end-effector toward the target pose.
        
        Args:
            dt: Time step
            
        Returns:
            bool: True if control was applied successfully
        """
        if self.using_isaac_controller:
            # Use Isaac Lab's DifferentialIKController
            if self.command_type == "velocity":
                # Get current end-effector pose
                current_pose = self.robot.get_end_effector_pose()
                current_position = current_pose[0]
                current_orientation = current_pose[1]
                
                # Compute the positional error
                position_error = self.target_position - current_position
                
                # For orientation, you may need a more sophisticated approach
                # For simplicity, we just use a simple error
                # In practice, you might want to use quaternion difference
                orientation_error = np.zeros(3)  # Simplified
                
                # Set desired end-effector velocity based on error
                # Scale factor determines how quickly to move toward target
                scale_factor = 5.0
                linear_velocity = position_error * scale_factor
                angular_velocity = orientation_error * scale_factor
                
                # Apply control
                self.controller.forward(
                    ee_current_position=current_position,
                    ee_current_orientation=current_orientation,
                    ee_desired_linear_velocity=linear_velocity,
                    ee_desired_angular_velocity=angular_velocity
                )
                
            elif self.command_type == "position":
                # Get current end-effector pose
                current_pose = self.robot.get_end_effector_pose()
                current_position = current_pose[0]
                current_orientation = current_pose[1]
                
                # Apply control directly with position target
                self.controller.forward(
                    ee_current_position=current_position,
                    ee_current_orientation=current_orientation,
                    ee_desired_position=self.target_position,
                    ee_desired_orientation=self.target_orientation
                )
            
            return True
            
        else:
            # Fall back to custom IK + PID control
            # Solve inverse kinematics to get joint targets
            joint_targets = self.ik_solver.solve(
                self.target_position, 
                self.target_orientation
            )
            
            if joint_targets is None:
                print("Failed to find IK solution")
                return False
            
            # Get current joint positions
            current_positions = self.robot.get_joint_positions()
            
            # Compute joint efforts with PID
            joint_efforts = []
            for i in range(self.num_joints):
                effort = self.pid_controllers[i].compute(
                    target=joint_targets[i],
                    current=current_positions[i],
                    dt=dt
                )
                joint_efforts.append(effort)
            
            # Apply joint efforts
            self.robot.apply_joint_efforts(joint_efforts)
            return True
            
    def get_end_effector_error(self):
        """
        Get the current error between target and actual end-effector position.
        
        Returns:
            tuple: (position_error, orientation_error)
        """
        # Get current end-effector pose
        current_pose = self.robot.get_end_effector_pose()
        current_position = current_pose[0]
        current_orientation = current_pose[1]
        
        # Compute positional error (Euclidean distance)
        position_error = np.linalg.norm(self.target_position - current_position)
        
        # Orientation error is more complex, for now just return a placeholder
        # In practice, you'd compute quaternion difference
        orientation_error = 0.0
        
        return position_error, orientation_error
    
    def is_target_reached(self, position_threshold=0.01, orientation_threshold=0.1):
        """
        Check if the target pose has been reached within thresholds.
        
        Args:
            position_threshold: Positional error threshold (meters)
            orientation_threshold: Orientation error threshold (radians)
            
        Returns:
            bool: True if target reached, False otherwise
        """
        position_error, orientation_error = self.get_end_effector_error()
        return position_error < position_threshold and orientation_error < orientation_threshold
    
    def reset(self):
        """
        Reset the controller state.
        """
        if self.using_isaac_controller:
            if hasattr(self.controller, 'reset'):
                self.controller.reset()
        else:
            # Reset custom controllers
            for pid in self.pid_controllers:
                pid.reset()