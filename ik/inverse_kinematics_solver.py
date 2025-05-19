from omni.isaac.core.articulations import ArticulationKinematicsView
import numpy as np

class IKSolver:
    """
    Wrapper for Isaac Lab's IK solver.
    """
    def __init__(
        self,
        prim_paths_expr: str = "/franka",
        end_effector_name: str = None
    ):
        self.kin_view = ArticulationKinematicsView(prim_paths_expr=prim_paths_expr)
        # If not specified, use the first end-effector in the articulation
        self.ee_body = end_effector_name or self.kin_view.get_ee_body_names()[0]

    def solve(self, target_position: np.ndarray, target_orientation: np.ndarray):
        """
        Compute joint targets to reach a given end-effector pose.
        Args:
            target_position: (x, y, z)
            target_orientation: quaternion (w, x, y, z)
        Returns:
            np.ndarray of joint positions (rad)
        """
        # Isaac Lab returns joint positions directly
        joint_targets = self.kin_view.solve_ik(
            body_name=self.ee_body,
            target_position=target_position,
            target_orientation=target_orientation
        )
        return joint_targets