from isaaclab.sim import ArticulationKinematicsView
import numpy as np

class FrankaKinematics:
    """
    Wrapper around Isaac Lab's ArticulationKinematicsView for forward kinematics.
    """
    def __init__(self, prim_paths_expr: str = "/World/Franka"):
        self.prim_path = prim_paths_expr  # Add this for controller compatibility
        self.kin_view = ArticulationKinematicsView(prim_paths_expr=prim_paths_expr)
        self.ee_body = self.kin_view.get_ee_body_names()[0]
        self.end_effector_prim_path = self.ee_body  # Add this for controller compatibility

    def get_end_effector_pose(self):
        """
        Returns the world pose (position, orientation) of the end-effector.
        """
        return self.kin_view.get_world_pose(self.ee_body)

    def get_joint_positions(self):
        """
        Returns the current joint positions of the robot.
        """
        return self.kin_view.get_joint_positions()
        
    def get_joint_velocities(self):
        """
        Returns the current joint velocities of the robot.
        """
        return self.kin_view.get_joint_velocities()
        
    def get_joint_efforts(self):
        """
        Returns the current joint efforts (torques) of the robot.
        """
        return self.kin_view.get_joint_efforts()
        
    def get_articulation_view(self):
        """
        Returns the underlying ArticulationKinematicsView object.
        Required by IsaacDiffIKController.
        """
        return self.kin_view
        
    def set_joint_positions(self, positions):
        """
        Sets the joint positions of the robot.
        """
        return self.kin_view.set_joint_positions(positions)
