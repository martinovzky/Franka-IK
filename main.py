import numpy as np
import argparse
import torch
import os
import gymnasium
from gymnasium.wrappers import RecordVideo

from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation
from isaaclab_assets.robots.franka import FRANKA_CFG
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from controllers.pid_controller import PIDController
from controllers.isaac_controller import IsaacDiffIKController
from ik.franka_kinematics import FrankaKinematics
from ik.inverse_kinematics_solver import IKSolver


def parse_args():
    parser = argparse.ArgumentParser(description="Franka arm control with Isaac Lab")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record", action="store_true", help="Record simulation")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration in seconds (for recording)")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running Isaac Lab Mode (recording: {'ON' if args.record else 'OFF'})")

    #launches the app with proper configuration
    app_launcher = AppLauncher(
        headless=args.headless,
        width=1280,
        height=720,
        window_title="Franka Robot Control"
    )
    simulation_app = app_launcher.app

    #sets up recording using gymnasium.wrappers.RecordVideo - Isaac Lab 2.0 official method
    video_recorder = None
    if args.record:
        record_dir = os.path.expanduser("~/Desktop/franka_recordings")
        os.makedirs(record_dir, exist_ok=True)
        
        #creates a dummy environment for recording wrapper
        class DummyEnv:
            def __init__(self):
                self.metadata = {"render_fps": 30}
                
        dummy_env = DummyEnv()
        video_recorder = RecordVideo(
            dummy_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True,  #records every episode
            name_prefix="franka_simulation"
        )
        print(f"Recording enabled using gymnasium.wrappers.RecordVideo")
        print(f"Videos will be saved to: {record_dir}")

    #configures simulation settings
    sim_cfg = SimulationCfg(
        dt=1/120.0,
        render_interval=2,  #renders every 2 physics steps (60 FPS)
        physx=PhysxCfg(
            solver_type=1,  #TGS solver
            use_gpu=True,
        ),
        physics_material=None,
    )

    #creates simulation context
    sim = SimulationContext(sim_cfg)

    #creates ground plane configuration
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        visual_material=None,
        debug_vis=False,
    )
    terrain = TerrainImporter(terrain_cfg)

    #spawns Franka robot using Isaac Lab asset configuration
    franka_cfg = FRANKA_CFG.replace(prim_path="/World/Franka")
    franka = Articulation(franka_cfg)

    #initializes simulation scene
    sim.reset()

    #starts recording if enabled
    if video_recorder:
        video_recorder.start_video_recorder()
        print("Video recording started...")

    #controller and kinematics setup
    kinematics = FrankaKinematics(prim_paths_expr="/World/Franka")
    ik_solver = IKSolver(prim_paths_expr="/World/Franka") 
    controller = IsaacDiffIKController(robot=kinematics, damping=0.05, command_type="position")

    #resets and steps simulation for initialization
    for _ in range(10):
        sim.step()
        if video_recorder:
            video_recorder.capture_frame()

    #TASK 1: reach target position
    target_pos = np.array([0.5, 0.0, 0.4])
    target_ori = np.array([1.0, 0.0, 0.0, 0.0])  #identity quaternion
    
    #sets target for controller
    controller.set_targets(position=target_pos, orientation=target_ori)

    print("Moving to target position...")
    reached = False
    step_count = 0
    max_steps = 1000  #safety limit
    
    while simulation_app.is_running() and not reached and step_count < max_steps:
        dt = sim_cfg.dt
        controller.compute_and_apply_control(dt)
        
        if controller.is_target_reached():
            reached = True
            print("Target reached!")
        
        sim.step()
        if video_recorder:
            video_recorder.capture_frame()
        step_count += 1

    #TASK 2: Square trajectory 
    print("Following square trajectory...")
    center = [0.5, 0.0, 0.4]
    size = 0.2
    T = 2.0  #time per segment
    
    #defines square corners
    pts = [
        [center[0], center[1] + size/2, center[2] + size/2],    #top right
        [center[0], center[1] - size/2, center[2] + size/2],    #top left  
        [center[0], center[1] - size/2, center[2] - size/2],    #bottom left
        [center[0], center[1] + size/2, center[2] - size/2],    #bottom right
    ]
    
    #executes square trajectory with simple linear interpolation
    for i, target_point in enumerate(pts):
        print(f"Moving to corner {i+1}/4")
        controller.set_targets(position=np.array(target_point), orientation=target_ori)
        
        #waits for target to be reached or timeout
        reached = False
        step_count = 0
        max_steps = int(T / sim_cfg.dt)  #converts time to steps
        
        while simulation_app.is_running() and not reached and step_count < max_steps:
            dt = sim_cfg.dt
            controller.compute_and_apply_control(dt)
            
            if controller.is_target_reached():
                reached = True
                print(f"Corner {i+1} reached!")
                break
                
            sim.step()
            if video_recorder:
                video_recorder.capture_frame()
            step_count += 1

    print("Square trajectory completed!")
    
    #stops recording and saves video
    if video_recorder:
        video_recorder.close_video_recorder()
        print("Recording completed and saved using gymnasium.wrappers.RecordVideo")
    
    #closes simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
