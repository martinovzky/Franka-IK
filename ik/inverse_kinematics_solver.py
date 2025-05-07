#!/usr/bin/env python3

"""
Inverse Kinematics Solver Module
===============================
Implementation of forward and inverse kinematics for a robot arm.
"""

import numpy as np

class RobotArmKinematics:
    """
    Implementation of forward and inverse kinematics for a robot arm.
    This example uses a simple 2-DOF planar robot for demonstration.
    
    Attributes:
        link_lengths (list): Lengths of robot arm links
        joint_angles (list): Current joint angles
        joint_limits (list): Min and max limits for each joint
    """
    
    def __init__(self, link_lengths=[0.5, 0.5], joint_limits=None):
        """
        Initialize robot arm kinematics with link lengths and joint limits.
        
        Args:
            link_lengths: List of robot arm link lengths
            joint_limits: List of (min, max) joint limits for each joint
        """
        self.link_lengths = link_lengths
        self.num_joints = len(link_lengths)
        
        # Default joint limits if not specified
        if joint_limits is None:
            self.joint_limits = [(-np.pi, np.pi) for _ in range(self.num_joints)]
        else:
            self.joint_limits = joint_limits
            
        # Initialize joint angles to zero
        self.joint_angles = [0.0] * self.num_joints
    
    def forward_kinematics(self, joint_angles=None):
        """
        Calculate end-effector position based on joint angles.
        
        Args:
            joint_angles: List of joint angles (radians). Uses current state if None.
            
        Returns:
            tuple: (x, y, z) position of end-effector
        """
        if joint_angles is None:
            joint_angles = self.joint_angles
        
        # For a 2-DOF planar robot arm
        if self.num_joints == 2:
            x = self.link_lengths[0] * np.cos(joint_angles[0]) + \
                self.link_lengths[1] * np.cos(joint_angles[0] + joint_angles[1])
            y = self.link_lengths[0] * np.sin(joint_angles[0]) + \
                self.link_lengths[1] * np.sin(joint_angles[0] + joint_angles[1])
            return (x, y, 0.0)
        else:
            # For more complex arms, implement the general DH parameter approach here
            # This is a simplified placeholder
            return (0.0, 0.0, 0.0)
    
    def inverse_kinematics_2dof(self, target_pos):
        """
        Solve inverse kinematics for a 2-DOF planar robot arm.
        Uses analytical solution for 2-DOF case.
        
        Args:
            target_pos: (x, y, z) target end-effector position
            
        Returns:
            list: Joint angles solution or None if no solution exists
        """
        x, y, _ = target_pos
        l1, l2 = self.link_lengths
        
        # Check if target is reachable
        distance = np.sqrt(x**2 + y**2)
        
        if distance > l1 + l2:
            print("Target position unreachable: too far")
            return None
        
        if distance < abs(l1 - l2):
            print("Target position unreachable: too close")
            return None
        
        # Calculate second joint angle (elbow)
        cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        
        # Clamp to avoid numerical errors
        cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
        
        # Two possible solutions: elbow up and elbow down
        theta2 = np.arccos(cos_theta2)  # Elbow up solution
        
        # Calculate first joint angle (shoulder)
        theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        
        # Check joint limits
        if (self.joint_limits[0][0] <= theta1 <= self.joint_limits[0][1] and 
            self.joint_limits[1][0] <= theta2 <= self.joint_limits[1][1]):
            return [theta1, theta2]
        
        # Try elbow-down solution if elbow-up didn't work
        theta2 = -np.arccos(cos_theta2)  # Elbow down solution
        theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        
        # Check joint limits again
        if (self.joint_limits[0][0] <= theta1 <= self.joint_limits[0][1] and 
            self.joint_limits[1][0] <= theta2 <= self.joint_limits[1][1]):
            return [theta1, theta2]
        
        print("No inverse kinematics solution within joint limits")
        return None
    
    def generate_trajectory(self, start_pos, end_pos, duration=1.0, dt=0.01):
        """
        Generate a linear trajectory from start to end position.
        
        Args:
            start_pos: Starting position (x, y, z)
            end_pos: Ending position (x, y, z)
            duration: Duration of trajectory in seconds
            dt: Time step for trajectory points
            
        Returns:
            list: List of (position, time) points along trajectory
        """
        start_x, start_y, start_z = start_pos
        end_x, end_y, end_z = end_pos
        
        # Calculate number of steps
        num_steps = int(duration / dt)
        
        # Generate trajectory points
        trajectory = []
        for i in range(num_steps + 1):
            t = i / num_steps  # Normalized time (0 to 1)
            
            # Linear interpolation between start and end
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            z = start_z + t * (end_z - start_z)
            
            time_point = i * dt
            trajectory.append(((x, y, z), time_point))
        
        return trajectory