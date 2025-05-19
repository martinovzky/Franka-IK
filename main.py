import numpy as np
import argparse

from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.franka import FRANKA_CFG
from isaaclab.utils import VisualizationUtils, RecordHelper
from isaaclab.utils import CartesianPath, CubicPolynomial

# Import our custom controllers and kinematics
from controllers.pid_controller import PIDController
from controllers.isaac_controller import IsaacDiffIKController
from ik.franka_kinematics import FrankaKinematics
from ik.inverse_kinematics_solver import IKSolver


def parse_args():
    parser = argparse.ArgumentParser(description="Franka arm control with Isaac Lab")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record", action="store_true", help="Record simulation")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds (for recording)")
    return parser.parse_args()


def run_isaac_lab_mode():
    args = parse_args()
    print(f"Running Isaac Lab Mode (recording: {'ON' if args.record else 'OFF'})")

    # Launch the app
    app = AppLauncher(headless=args.headless, width=1280, height=720,
                      window_title="Franka Robot Control").app

    # Create simulation context
    sim = SimulationContext(physics_dt=1/120.0, rendering_dt=1/60.0)

    # Recording
    if args.record:
        import os
        out = os.path.join(os.path.expanduser("~"), "Desktop", "franka_recordings")
        os.makedirs(out, exist_ok=True)
        recorder = RecordHelper(output_dir=out, frame_rate=30)
        recorder.start_recording()
        print(f"Recording to: {out}")

    # Add ground plane
    sim.add_ground_plane(z_position=0.0)

    # Spawn Franka via asset cfg
    franka = FRANKA_CFG.replace(prim_path="/World/Franka").build()
    sim.add_articulation(franka)

    # Camera & rendering utilities
    vis = VisualizationUtils(sim)
    vis.set_camera(eye=[1.5,1.5,1.0], target=[0,0,0.3], up=[0,0,1])
    vis.enable_shadows()

    # Controller and kinematics
    kinematics = FrankaKinematics(prim_paths_expr="/World/Franka")
    ik_solver = IKSolver(prim_paths_expr="/World/Franka")
    controller = IsaacDiffIKController(robot=franka, damping=0.05, command_type="position")

    # Reset and step
    for _ in range(10): sim.step()

    # TASK 1: reach target
    target_pos = np.array([0.5,0.0,0.4])
    target_ori = np.array([1.0,0.0,0.0,0.0])
    vis.draw_sphere(position=target_pos, radius=0.02, color=[0,1,0])
    controller.set_targets(position=target_pos, orientation=target_ori)

    reached = False
    while app.is_running() and not reached:
        dt = sim.get_physics_dt()
        controller.compute_and_apply_control(dt)
        if controller.is_target_reached():
            reached = True
            print("Target reached!")
        sim.step()

    # TASK 2: square trajectory
    center = [0.5,0.0,0.4]; size=0.2; T=2.0
    pts = [
        [center[0], center[1]+size/2, center[2]+size/2],    # Top right
        [center[0], center[1]-size/2, center[2]+size/2],    # Top left  
        [center[0], center[1]-size/2, center[2]-size/2],    # Bottom left
        [center[0], center[1]+size/2, center[2]-size/2],    # Bottom right
        [center[0], center[1]+size/2, center[2]+size/2]     # Back to top right
    ]
    traj = CartesianPath(points=pts[:-1], time_spans=[T]*4, interpolation_type=CubicPolynomial())
    print("Following square...")
    t=0
    total=4*T
    while app.is_running() and t<=total:
        pose = traj.get_pose_at_time(t)
        controller.set_targets(position=pose.translation,
                                orientation=pose.quaternion())
        dt = sim.get_physics_dt()
        controller.compute_and_apply_control(dt)
        sim.step(); t += dt

    # Finalize recording
    if args.record:
        for _ in range(int(0.5/dt)): sim.step()
        recorder.stop_recording()
        print("Recording saved.")

    app.close()


def main():
    run_isaac_lab_mode()


if __name__ == "__main__":
    main()
