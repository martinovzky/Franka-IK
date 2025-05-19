from omni.isaac.core.articulations import ArticulationKinematicsView

class FrankaKinematics:
    """
    Wrapper around Isaac Lab's ArticulationKinematicsView for forward kinematics.
    """
    def __init__(self, prim_paths_expr: str = "/franka"):
        self.kin_view = ArticulationKinematicsView(prim_paths_expr=prim_paths_expr)
        self.ee_body = self.kin_view.get_ee_body_names()[0]

    def get_end_effector_pose(self):
        """
        Returns the world pose (position, orientation) of the end-effector.
        """
        pose = self.kin_view.get_world_pose(self.ee_body)
        return pose
    def get_joint_positions(self):
        """
        Returns the current joint positions of the robot.
        """
        joint_positions = self.kin_view.get_joint_positions()
        return joint_positions
    def get_joint_velocities(self):
        """
        Returns the current joint velocities of the robot.
        """
        joint_velocities = self.kin_view.get_joint_velocities()
        return joint_velocities
    def get_joint_efforts(self):
        """
        Returns the current joint efforts of the robot.
        """
        joint_efforts = self.kin_view.get_joint_efforts()
        return joint_efforts
    def get_joint_names(self):
        """
        Returns the names of the joints in the robot.
        """ 
        joint_names = self.kin_view.get_joint_names()
        return joint_names      
    def get_end_effector_velocity(self):
        """
        Returns the current velocity of the end-effector.
        """
        end_effector_velocity = self.kin_view.get_end_effector_velocity(self.ee_body)
        return end_effector_velocity    
    def get_end_effector_linear_velocity(self):
        """
        Returns the current linear velocity of the end-effector.
        """
        end_effector_linear_velocity = self.kin_view.get_end_effector_linear_velocity(self.ee_body)
        return end_effector_linear_velocity
    def get_end_effector_angular_velocity(self):
        """
        Returns the current angular velocity of the end-effector.
        """
        end_effector_angular_velocity = self.kin_view.get_end_effector_angular_velocity(self.ee_body)
        return end_effector_angular_velocity
    def get_end_effector_linear_acceleration(self):
        """
        Returns the current linear acceleration of the end-effector.
        """
        end_effector_linear_acceleration = self.kin_view.get_end_effector_linear_acceleration(self.ee_body)
        return end_effector_linear_acceleration
    def get_end_effector_angular_acceleration(self):
        """
        Returns the current angular acceleration of the end-effector.
        """
        end_effector_angular_acceleration = self.kin_view.get_end_effector_angular_acceleration(self.ee_body)
        return end_effector_angular_acceleration
    