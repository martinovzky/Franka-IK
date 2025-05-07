#!/usr/bin/env python3

"""
PID Controller Module
====================
Implementation of a PID controller for robot joint control.
"""

import time
import numpy as np

class PIDController:
    """
    PID Controller implementation for robot joint control.
    
    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        prev_error (float): Previous error for derivative calculation
        integral (float): Accumulated error for integral calculation
        output_limits (tuple): Min and max output limits
    """
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-float('inf'), float('inf'))):
        """
        Initialize PID controller with gains and limits.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.output_limits = output_limits
        self.last_time = None
        
    def reset(self):
        """Reset the PID controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def compute(self, setpoint, process_value, dt=None):
        """
        Compute PID control value based on setpoint and current process value.
        
        Args:
            setpoint: Target value
            process_value: Current measured value
            dt: Time delta since last computation (optional)
            
        Returns:
            float: Control output
        """
        # Calculate current error
        error = setpoint - process_value
        
        # Handle time delta
        current_time = time.time()
        if self.last_time is None:
            self.last_time = current_time
        if dt is None:
            dt = current_time - self.last_time
        
        # Ensure minimum time delta to avoid division by zero
        dt = max(dt, 0.001)
        self.last_time = current_time
        
        # Calculate the integral term with anti-windup
        self.integral += error * dt
        
        # Calculate the derivative term
        derivative = (error - self.prev_error) / dt
        
        # Calculate output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Apply output limits if specified
        if self.output_limits[0] is not None and output < self.output_limits[0]:
            output = self.output_limits[0]
        elif self.output_limits[1] is not None and output > self.output_limits[1]:
            output = self.output_limits[1]
        
        # Store error for next iteration
        self.prev_error = error
        
        return output