from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz
import numpy as np
import torch

class FrankaKinematics:
    def __init__(self, prim_paths_expr: str = "/World/Franka"):
        """
        Wrapper around Isaac Lab's Articulation for Franka robot kinematics.
        Provides forward kinematics, jacobian computation, and joint control.
        """
        self.prim_path = prim_paths_expr
        self.articulation = None
        self.ee_frame = "panda_hand"  #end-effector frame name
        
    def set_articulation(self, articulation: Articulation):
        """
        Sets the robot articulation object.
        """
        self.articulation = articulation
        
    def get_end_effector_pose(self):
        """
        Gets current end-effector pose (position + orientation).
        Returns: tuple of (position, orientation) as numpy arrays
        """
        if self.articulation is None:
            raise RuntimeError("Articulation not set. Call set_articulation() first.")
            
        #gets end-effector pose from the articulation
        ee_pose = self.articulation.data.body_pos_w[:, self.articulation.find_bodies(self.ee_frame)[0]]
        ee_quat = self.articulation.data.body_quat_w[:, self.articulation.find_bodies(self.ee_frame)[0]]
        
        #converts to numpy and remove batch dimension
        position = ee_pose.cpu().numpy().flatten()
        orientation = ee_quat.cpu().numpy().flatten()
        
        return position, orientation
        
    def get_joint_positions(self):
        """
        Gets current joint positions.
        Returns: numpy array of joint positions
        """
        if self.articulation is None:
            raise RuntimeError("Articulation not set")
            
        return self.articulation.data.joint_pos.cpu().numpy().flatten()
        
    def set_joint_positions(self, joint_positions: np.ndarray):
        """
        Sets joint positions for the robot.
        Args: joint_positions - array of 7 joint angles for Franka
        """
        if self.articulation is None:
            raise RuntimeError("Articulation not set")
            
        #converts to torch tensor if needed
        if isinstance(joint_positions, np.ndarray):
            joint_pos = torch.tensor(joint_positions, dtype=torch.float32, device=self.articulation.device)
        else:
            joint_pos = joint_positions
            
        #ensures batch dimension
        if joint_pos.dim() == 1:
            joint_pos = joint_pos.unsqueeze(0)
            
        #applies joint positions
        self.articulation.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        
    def get_jacobian(self):
        """
        Gets the current jacobian matrix for end-effector.
        Returns: jacobian matrix as numpy array
        """
        if self.articulation is None:
            raise RuntimeError("Articulation not set")
            
        try:
            #gets jacobian from Isaac Lab
            jacobian = self.articulation.root_physx_view.get_jacobians()
            
            #extracts end-effector jacobian (usually last 6 rows for position + orientation)
            ee_jacobian = jacobian[:, -6:, :]  #last 6 DOF correspond to end-effector
            
            return ee_jacobian.cpu().numpy()
            
        except Exception as e:
            print(f"Failed to get jacobian: {e}")
            #returns identity matrix as fallback
            return np.eye(6, 7)
            
    def compute_forward_kinematics(self, joint_positions: np.ndarray):
        """
        Computes forward kinematics for given joint positions.
        Args: joint_positions - array of 7 joint angles
        Returns: tuple of (position, orientation) of end-effector
        """
        if self.articulation is None:
            raise RuntimeError("Articulation not set")
            
        #stores current state
        current_pos = self.get_joint_positions()
        
        #sets new joint positions temporarily
        self.set_joint_positions(joint_positions)
        
        #gets end-effector pose
        position, orientation = self.get_end_effector_pose()
        
        #restores original state
        self.set_joint_positions(current_pos)
        
        return position, orientation
