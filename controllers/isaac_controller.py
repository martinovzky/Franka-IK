import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

class IsaacDiffIKController:
    """
    Wrapper around Isaac Lab's DifferentialIKController for end-effector control.
    Provides methods to set targets, compute/apply control, query error, and reset controller state.
    """
    def __init__(self, robot, damping=0.05, command_type="position", device="cuda:0"):
        self.robot = robot
        self.device = device
        
        #initializes target pose
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  #identity quaternion
        
        #builds controller configuration according to Isaac Lab 2.0 documentation
        cfg = DifferentialIKControllerCfg(
            command_type=command_type,
            ik_method="dls",  #damped least squares
            ik_params={"lambda_val": damping},  #correct parameter name for DLS method
            use_relative_mode=False
        )
        
        #creates the controller: single environment
        self.controller = DifferentialIKController(
            cfg=cfg,
            num_envs=1,
            device=device
        )

    def set_targets(self, position=None, orientation=None):
        """
        Update the desired end-effector targets.
        """
        if position is not None:
            self.target_position = np.array(position)
        if orientation is not None:
            self.target_orientation = np.array(orientation)
        
        #converts to torch tensors and send command to controller
        #DifferentialIKController expects tensors with shape (num_envs, dim)
        cmd_pos = torch.tensor(self.target_position, device=self.device, dtype=torch.float32).unsqueeze(0)
        cmd_ori = torch.tensor(self.target_orientation, device=self.device, dtype=torch.float32).unsqueeze(0)
        cmd = torch.cat([cmd_pos, cmd_ori], dim=1)  #shape: (1, 7)
        
        self.controller.set_command(cmd)

    def compute_and_apply_control(self, dt: float):
        """
        Advance the controller and apply resulting joint commands to the robot.
        """
        #checks if required methods exist
        if not hasattr(self.robot, 'get_articulation_view'):
            raise AttributeError("Robot object must provide 'get_articulation_view()' method")

        #gets the articulation view from the robot
        art_view = self.robot.get_articulation_view()
        
        #gets current joint positions as torch tensor
        joint_positions = art_view.get_joint_positions()  #should return torch tensor
        if isinstance(joint_positions, np.ndarray):
            joint_positions = torch.tensor(joint_positions, device=self.device, dtype=torch.float32)
        
        #gets current end-effector pose
        ee_pose = self.robot.get_end_effector_pose()
        ee_pos, ee_quat = ee_pose
        
        #converts to torch tensors if needed
        if isinstance(ee_pos, np.ndarray):
            ee_pos = torch.tensor(ee_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        if isinstance(ee_quat, np.ndarray):
            ee_quat = torch.tensor(ee_quat, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        #gets jacobian matrix (this might need adjustment based on actual API)
        try:
            jacobian = art_view.get_jacobian(link_name=self.robot.end_effector_prim_path)
            if isinstance(jacobian, np.ndarray):
                jacobian = torch.tensor(jacobian, device=self.device, dtype=torch.float32)
        except AttributeError:
            #fallback: create a dummy jacobian or use alternative method
            print("Warning: Jacobian computation not available, using identity matrix")
            jacobian = torch.eye(6, joint_positions.shape[-1], device=self.device, dtype=torch.float32).unsqueeze(0)

        #computes new joint positions
        target_q = self.controller.compute(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            jacobian=jacobian,
            joint_pos=joint_positions
        )
        
        #applies to robot - convert back to numpy if needed
        if hasattr(self.robot, 'set_joint_positions'):
            if isinstance(target_q, torch.Tensor):
                target_q_np = target_q.detach().cpu().numpy().flatten()
            else:
                target_q_np = target_q.flatten()
            self.robot.set_joint_positions(target_q_np)

    def get_end_effector_error(self):
        """
        Compute current error between target and actual end-effector pose.
        Returns: (position_error, orientation_error)
        """
        current_pose = self.robot.get_end_effector_pose()
        current_pos, current_quat = current_pose
        pos_err = np.linalg.norm(self.target_position - current_pos)
        #orientation error placeholder (could compute quaternion distance)
        ori_err = np.linalg.norm(self.target_orientation - current_quat)
        return pos_err, ori_err

    def is_target_reached(self, position_threshold=1e-3, orientation_threshold=1e-2):
        """
        Check if the target has been reached within specified thresholds.
        """
        pos_err, ori_err = self.get_end_effector_error()
        return (pos_err < position_threshold) and (ori_err < orientation_threshold)

    def reset(self):
        """
        Reset the internal state of the controller.
        """
        try:
            self.controller.reset()
        except AttributeError:
            pass