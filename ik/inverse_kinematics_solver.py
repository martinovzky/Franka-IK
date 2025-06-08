from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz
import numpy as np
import torch

class IKSolver:
    def __init__(self, prim_paths_expr: str = "/World/Franka", end_effector_name: str = None):
        """
        Initialize IK solver as a wrapper around Isaac Lab's DifferentialIKController.
        This provides a simple interface for one-shot IK solving.
        """
        self.prim_path = prim_paths_expr
        self.ee_body = end_effector_name or "panda_hand"  #default end-effector for Franka
        self.articulation = None
        self._ik_controller = None
        
    def set_articulation(self, articulation: Articulation):
        """
        Sets the robot articulation for IK solving.
        """
        self.articulation = articulation
        
    def setup_ik_controller(self):
        """
        Sets up the internal IK controller for solving.
        """
        if self.articulation is None:
            raise RuntimeError("Articulation must be set before setting up IK controller")
            
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        
        #creates IK controller configuration
        cfg = DifferentialIKControllerCfg(
            command_type="position",
            ik_method="dls",  #damped least squares
            ik_params={"lambda_val": 0.05},
            use_relative_mode=False
        )
        
        #creates the IK controller
        self._ik_controller = DifferentialIKController(
            cfg=cfg,
            num_envs=1,
            device=self.articulation.device
        )

    def solve(self, target_position: np.ndarray, target_orientation: np.ndarray):
        """
        Solve inverse kinematics for target pose using Isaac Lab's IK controller.
        Returns joint positions that achieve the target end-effector pose.
        """
        if self.articulation is None:
            raise RuntimeError("Articulation not set. Call set_articulation() first.")
            
        if self._ik_controller is None:
            self.setup_ik_controller()
        
        #converts inputs to torch tensors
        if isinstance(target_position, np.ndarray):
            target_pos = torch.tensor(target_position, dtype=torch.float32, device=self.articulation.device)
        else:
            target_pos = target_position
            
        if isinstance(target_orientation, np.ndarray):
            target_ori = torch.tensor(target_orientation, dtype=torch.float32, device=self.articulation.device)
        else:
            target_ori = target_orientation
            
        #ensures tensors have batch dimension
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)
        if target_ori.dim() == 1:
            target_ori = target_ori.unsqueeze(0)
            
        #gets current joint positions as starting point
        current_joint_pos = self.articulation.data.joint_pos
        
        #uses the IK controller to solve for joint positions
        try:
            #sets the target pose
            target_pose = torch.cat([target_pos, target_ori], dim=-1)
            
            #computes IK solution
            joint_positions = self._ik_controller.compute(
                ee_pose=target_pose,
                joint_pos=current_joint_pos,
                jacobian=self.articulation.root_physx_view.get_jacobians()  #gets current jacobian
            )
            
            return joint_positions.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"IK solving failed: {e}")
            #returns current joint positions as fallback
            return current_joint_pos.cpu().numpy().flatten()