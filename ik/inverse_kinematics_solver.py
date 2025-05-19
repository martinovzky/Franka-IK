from isaaclab.sim import ArticulationKinematicsView
import numpy as np

class IKSolver:
    def __init__(self, prim_paths_expr: str = "/World/Franka", end_effector_name: str = None):
        self.kin_view = ArticulationKinematicsView(prim_paths_expr=prim_paths_expr)
        self.ee_body = end_effector_name or self.kin_view.get_ee_body_names()[0]

    def solve(self, target_position: np.ndarray, target_orientation: np.ndarray):
        return self.kin_view.solve_ik(
            body_name=self.ee_body,
            target_position=target_position,
            target_orientation=target_orientation
        )