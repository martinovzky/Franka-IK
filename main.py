import numpy as np
import argparse

# Import Isaac Lab components when running on VM
try:
    from omni.isaac.lab import App, SimulationContext
    from omni.isaac.lab.robots import FrankaRobot
    from omni.isaac.lab.utils import VisualizationUtils
    from omni.isaac.lab.utils import RecordHelper
    from omni.isaac.lab.motion import CartesianPath, CubicPolynomial
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    print("Isaac Lab not available, running in compatibility mode")
    ISAAC_LAB_AVAILABLE = False
    # Fall back to standard Isaac Sim imports
    from omni.isaac.kit import SimulationApp
    from omni.isaac.core import World
    from omni.isaac.franka import Franka
    from omni.isaac.core.articulations import ArticulationView

# Import our custom controllers
from controllers.pid_controller import PIDController
from controllers.isaac_controller import IsaacDiffIKController

# Import our custom kinematics and IK
from ik.franka_kinematics import FrankaKinematics
from ik.inverse_kinematics_solver import IKSolver


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Franka arm control with Isaac Lab")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record", action="store_true", help="Record simulation")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds (for recording)")
    return parser.parse_args()


def run_isaac_lab_mode():
    """Run the simulation using Isaac Lab (higher-level API)"""
    # Parse args
    args = parse_args()
    print(f"Running with Isaac Lab (Recording: {'ON' if args.record else 'OFF'})")
    
    # Initialize Isaac Lab application with viewport settings
    app = App({
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "window_title": "Franka Robot Control - Isaac Lab Mode"
    })
    
    # Create simulation context, equivalent to World in Isaac Sim
    sim_context = SimulationContext(
        physics_dt=1.0/120.0,
        rendering_dt=1.0/60.0,
        stage_units_in_meters=1.0
    )
    
    # Simple recording setup if enabled
    if args.record:
        # Create output directory
        import os
        record_dir = os.path.join(os.path.expanduser("~"), "Desktop", "franka_recordings")
        os.makedirs(record_dir, exist_ok=True)
        
        # Initialize the built-in RecordHelper - just one line!
        recorder = RecordHelper(
            output_dir=record_dir,
            frame_rate=30
        )
        
        # Start recording
        print(f"Recording enabled. Output will be saved to: {record_dir}")
        recorder.start_recording()
    
    # Add a ground plane
    sim_context.add_ground_plane(z_position=0.0)
    
    # Add Franka robot
    franka = FrankaRobot(
        prim_path="/World/Franka",
        name="franka", 
        position=[0, 0, 0]
    )
    sim_context.add_articulation(franka)
    
    # Setup camera to view the scene
    from omni.isaac.lab.utils import ViewportHelper
    viewport_helper = ViewportHelper()
    viewport_helper.set_camera_view(
        eye=[1.5, 1.5, 1.0],
        target=[0.0, 0.0, 0.3],
        up=[0, 0, 1]
    )
    
    # Enable shadows and better rendering
    from omni.isaac.lab.utils import RenderProductHelper
    render_product = RenderProductHelper()
    render_product.enable_shadows()
    
    # Replace the PID controller setup and IK solving with the DifferentialIKController
    print("Setting up Differential IK Controller...")
    controller = IsaacDiffIKController(
        robot=franka,
        damping=0.05,
        command_type="position"  # "position" is more intuitive than "velocity"
    )

    # Reset simulation to settle physics
    for _ in range(10):
        sim_context.step()
        
    # =====================================================
    # TASK 1: REACH THE TARGET POSITION
    # =====================================================
    
    # Define the target position and orientation
    target_position = np.array([0.5, 0.0, 0.4])
    target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion w,x,y,z
    
    # Visualize the target position
    VisualizationUtils.create_sphere(
        scene=sim_context.scene,
        position=target_position,
        radius=0.02,
        color=[0.0, 1.0, 0.0]  # Green sphere
    )
    
    # Set the target for the controller
    print("TASK 1: Moving to target position...")
    controller.set_targets(
        position=target_position,
        orientation=target_orientation
    )
    
    # Wait until the robot reaches the target position
    target_reached = False
    while app.is_running() and not target_reached:
        dt = sim_context.get_physics_dt()
        
        # Apply control
        controller.compute_and_apply_control(dt)
        
        # Check if target is reached
        if controller.is_target_reached():
            target_reached = True
            print("TASK 1 COMPLETED: Target position reached!")
            
            # Optional: pause briefly at the target position
            for _ in range(60):  # pause for ~0.5 seconds (60 frames at 120Hz)
                controller.compute_and_apply_control(dt)
                sim_context.step()
        
        sim_context.step()
    
    # =====================================================
    # TASK 2: FOLLOW A SQUARE TRAJECTORY
    # =====================================================
    
    # Define square parameters
    square_center = [0.5, 0.0, 0.4]  # Same as target position
    square_size = 0.2
    time_per_segment = 2.0  # seconds
    
    # Define the square corners (5 points to close the square)
    half_size = square_size / 2.0
    square_points = [
        [square_center[0], square_center[1] + half_size, square_center[2] + half_size],  # Top right
        [square_center[0], square_center[1] - half_size, square_center[2] + half_size],  # Top left
        [square_center[0], square_center[1] - half_size, square_center[2] - half_size],  # Bottom left
        [square_center[0], square_center[1] + half_size, square_center[2] - half_size],  # Bottom right
        [square_center[0], square_center[1] + half_size, square_center[2] + half_size]   # Back to top right
    ]
    
    # Visualize the square
    corner_colors = [
        [0.0, 1.0, 0.0],   # Green
        [1.0, 1.0, 0.0],   # Yellow
        [1.0, 0.5, 0.0],   # Orange
        [1.0, 0.0, 0.0],   # Red
        [0.0, 1.0, 0.0]    # Green again
    ]
    
    # Visualize corners
    for i, point in enumerate(square_points):
        VisualizationUtils.create_sphere(
            scene=sim_context.scene,
            position=point,
            radius=0.015,
            color=corner_colors[i]
        )
    
    # Visualize connecting lines
    for i in range(len(square_points) - 1):
        VisualizationUtils.create_line(
            scene=sim_context.scene,
            start=square_points[i],
            end=square_points[i + 1],
            color=[0.0, 0.7, 1.0],  # Blue lines
            thickness=0.004
        )
    
    # Create a smooth trajectory using Isaac Lab's CartesianPath
    segment_times = [time_per_segment] * (len(square_points) - 1)
    square_trajectory = CartesianPath(
        points=square_points,
        time_spans=segment_times,
        interpolation_type=CubicPolynomial()
    )
    
    # Follow the square trajectory
    print("TASK 2: Following square trajectory...")
    sim_time = 0.0
    total_trajectory_time = sum(segment_times)
    
    while app.is_running() and sim_time <= total_trajectory_time:
        dt = sim_context.get_physics_dt()
        
        # Sample the trajectory at the current time
        desired_pose = square_trajectory.get_pose_at_time(sim_time)
        position = desired_pose.translation
        orientation = desired_pose.quaternion()
        
        # Set target for controller
        controller.set_targets(position=position, orientation=orientation)
        
        # Apply control
        controller.compute_and_apply_control(dt)
        
        # Update simulation
        sim_context.step()
        sim_time += dt
        
        # Display progress
        if int(sim_time) % 1 == 0 and int(sim_time * 10) % 10 == 0:  # Every second
            completion = min(100.0, (sim_time / total_trajectory_time) * 100.0)
            print(f"Square trajectory progress: {completion:.1f}%")
    
    print("TASK 2 COMPLETED: Square trajectory finished!")
    
    if args.record and 'recorder' in locals() and recorder is not None:
        print("Finalizing recording...")
        for _ in range(30): # Step a few more frames
            sim_context.step()
            if not app.is_running(): # Check if app closed during these steps
                break
    
    # Cleanup
    print("All tasks completed.")
    app.close()


def run_isaac_sim_mode():
    """Run the simulation using standard Isaac Sim (original implementation)"""
    print("Running with standard Isaac Sim")
    
    # Launch Isaac Sim
    args = parse_args()
    sim_app = SimulationApp({"headless": args.headless})
    
    # Create the world and add ground
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Spawn the Franka robot
    franka = Franka(prim_path="/franka", name="franka")
    
    # Initialize views
    art_view = ArticulationView(prim_paths_expr="/franka")
    kin = FrankaKinematics(prim_paths_expr="/franka")
    ik_solver = IKSolver(
        prim_paths_expr="/franka",
        end_effector_name=kin.ee_body
    )
    
    # Reset world to initial state
    world.reset()
    
    # Create the controller
    controller = IsaacDiffIKController(
        robot=franka,
        damping=0.05,
        command_type="position"
    )
    
    # Define target pose (unchanged)
    target_position = np.array([0.5, 0.0, 0.4])
    target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion w,x,y,z
    
    # Set the target
    controller.set_targets(
        position=target_position,
        orientation=target_orientation
    )
    
    # Main loop
    while sim_app.is_running():
        # Get dt
        dt = world.get_physics_dt()
        
        # Apply control - single line replaces all the IK and PID computation
        controller.compute_and_apply_control(dt)
        
        # Step the simulation
        world.step()
    
    # Close the app
    sim_app.close()

def main():
    """
    Main entry point that chooses between Isaac Lab and Isaac Sim based on availability
    """
    if ISAAC_LAB_AVAILABLE:
        run_isaac_lab_mode()
    else:
        run_isaac_sim_mode()


if __name__ == "__main__":
    main()
  