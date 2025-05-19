import numpy as np
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

class IsaacDiffIKController:
    """
    Wrapper around Isaac Lab's DifferentialIKController for end-effector control.
    Provides methods to set targets, compute/apply control, query error, and reset controller state.
    """
    def __init__(self, robot, damping=0.05, command_type="position", device="cuda:0"):
        self.robot = robot
        # Initialize target pose
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        # Build controller configuration
        cfg = DifferentialIKControllerCfg(
            command_type=command_type,
            ik_method="pinv",
            use_relative_mode=False
        )
        # Create the controller: single environment
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
        # Send command to controller (absolute mode)
        # DifferentialIKController expects shape (N,3) or (N,7)
        cmd = np.concatenate([self.target_position, self.target_orientation])[None, :]
        self.controller.set_command(cmd)

    def compute_and_apply_control(self, dt: float):
        """
        Advance the controller and apply resulting joint commands to the robot.
        """
        # Check if required methods exist
        if not hasattr(self.robot, 'get_articulation_view'):
            raise AttributeError("Robot object must provide 'get_articulation_view()' method")

        # Retrieve current joint positions and jacobian
        art_view = self.robot.get_articulation_view()  # assume robot provides view
        joint_positions = art_view.get_joint_positions()  # numpy array shape (1, J)
        jacobian = art_view.get_world_cartesian_jacobian(
            prim_paths_expr=self.robot.prim_path, 
            link_name=self.robot.end_effector_prim_path
        )
        # Compute new joint positions
        # Get current end-effector pose
        ee_pose = self.robot.get_end_effector_pose()
        ee_pos, ee_quat = ee_pose

        target_q = self.controller.compute(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            jacobian=jacobian,
            joint_pos=joint_positions
        )
        # Apply to robot
        self.robot.set_joint_positions(target_q.flatten())

    def get_end_effector_error(self):
        """
        Compute current error between target and actual end-effector pose.
        Returns: (position_error, orientation_error)
        """
        current_pose = self.robot.get_end_effector_pose()
        current_pos, current_quat = current_pose
        pos_err = np.linalg.norm(self.target_position - current_pos)
        # Orientation error placeholder (could compute quaternion distance)
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