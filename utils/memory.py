"""
Memory Management Utilities
===========================

Utilities for GPU and system memory management to prevent memory leaks
and optimize resource usage during inference.
"""

import gc
import psutil
import torch
import numpy as np
from typing import Dict, Optional


def cleanup_gpu_memory():
    """
    Clean up GPU memory to prevent memory leaks
    
    This function should be called periodically during inference or after
    processing batches to free unused GPU memory.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure operations are completed
        gc.collect()  # Force garbage collection
    except Exception as e:
        print(f"Warning: GPU cleanup failed: {e}")


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory usage information
    
    Returns:
        Dictionary containing memory information in MB
    """
    memory_info = {}
    
    # System memory
    sys_memory = psutil.virtual_memory()
    memory_info['system_total'] = sys_memory.total / (1024**2)
    memory_info['system_used'] = sys_memory.used / (1024**2)
    memory_info['system_available'] = sys_memory.available / (1024**2)
    memory_info['system_percent'] = sys_memory.percent
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.memory_stats(device=i)
            memory_info[f'gpu_{i}_allocated'] = gpu_memory['allocated_bytes.all.current'] / (1024**2)
            memory_info[f'gpu_{i}_reserved'] = gpu_memory['reserved_bytes.all.current'] / (1024**2)
            memory_info[f'gpu_{i}_max_allocated'] = gpu_memory['allocated_bytes.all.peak'] / (1024**2)
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            memory_info[f'gpu_{i}_total'] = props.total_memory / (1024**2)
    else:
        memory_info['gpu_available'] = False
    
    return memory_info


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage of a function
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function that logs memory usage
    """
    def wrapper(*args, **kwargs):
        # Get memory before execution
        memory_before = get_memory_info()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Get memory after execution
            memory_after = get_memory_info()
            
            # Calculate differences
            sys_diff = memory_after['system_used'] - memory_before['system_used']
            
            print(f"Memory usage for {func.__name__}:")
            print(f"  System memory change: {sys_diff:+.2f} MB")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_before = memory_before.get(f'gpu_{i}_allocated', 0)
                    gpu_after = memory_after.get(f'gpu_{i}_allocated', 0)
                    gpu_diff = gpu_after - gpu_before
                    print(f"  GPU {i} memory change: {gpu_diff:+.2f} MB")
            
            return result
            
        except Exception as e:
            # Clean up memory on error
            cleanup_gpu_memory()
            raise e
    
    return wrapper


def check_memory_threshold(threshold_percent: float = 80.0) -> bool:
    """
    Check if memory usage is above threshold
    
    Args:
        threshold_percent: Memory usage threshold (0-100)
        
    Returns:
        True if memory usage is above threshold
    """
    memory_info = get_memory_info()
    
    # Check system memory
    if memory_info['system_percent'] > threshold_percent:
        return True
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_used = memory_info.get(f'gpu_{i}_allocated', 0)
            gpu_total = memory_info.get(f'gpu_{i}_total', 1)  # Avoid division by zero
            gpu_percent = (gpu_used / gpu_total) * 100
            
            if gpu_percent > threshold_percent:
                return True
    
    return False


def force_memory_cleanup():
    """
    Aggressive memory cleanup for critical situations
    
    This should only be used when memory usage is critically high.
    """
    # Multiple rounds of cleanup
    for _ in range(3):
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Additional cleanup for numpy
    try:
        # Force cleanup of numpy arrays if any are lingering
        import numpy as np
        np.seterr(all='ignore')  # Ignore warnings during cleanup
    except:
        pass


class MemoryMonitor:
    """
    Context manager for monitoring memory usage
    
    Usage:
        with MemoryMonitor("inference"):
            result = model(input_tensor)
    """
    
    def __init__(self, operation_name: str, cleanup_on_exit: bool = True):
        """
        Initialize memory monitor
        
        Args:
            operation_name: Name of operation being monitored
            cleanup_on_exit: Whether to cleanup memory on exit
        """
        self.operation_name = operation_name
        self.cleanup_on_exit = cleanup_on_exit
        self.start_memory = None
        self.end_memory = None
    
    def __enter__(self):
        """Enter context - record starting memory"""
        try:
            self.start_memory = get_memory_info()
        except Exception as e:
            print(f"Warning: Could not get starting memory info: {e}")
            self.start_memory = {}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - record ending memory and cleanup"""
        try:
            self.end_memory = get_memory_info()
        except Exception as e:
            print(f"Warning: Could not get ending memory info: {e}")
            self.end_memory = {}
        
        # Display memory usage comparison (only if both measurements succeeded)
        if self.start_memory and self.end_memory:
            self._display_memory_usage()
        
        # Cleanup if requested
        if self.cleanup_on_exit:
            try:
                cleanup_gpu_memory()
            except Exception as e:
                print(f"Warning: Memory cleanup failed: {e}")
        
        # Don't suppress exceptions
        return False
    
    def _display_memory_usage(self):
        """Display memory usage comparison"""
        if not self.start_memory or not self.end_memory:
            return
        
        try:
            print(f"\nMemory usage for '{self.operation_name}':")
            
            # System memory
            start_sys = self.start_memory.get('system_used', 0)
            end_sys = self.end_memory.get('system_used', 0)
            sys_diff = end_sys - start_sys
            print(f"  System memory: {sys_diff:+.2f} MB")
            
            # GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    start_gpu = self.start_memory.get(f'gpu_{i}_allocated', 0)
                    end_gpu = self.end_memory.get(f'gpu_{i}_allocated', 0)
                    gpu_diff = end_gpu - start_gpu
                    print(f"  GPU {i} memory: {gpu_diff:+.2f} MB")
        except Exception as e:
            print(f"Warning: Could not display memory usage: {e}")


def optimize_for_inference():
    """
    Optimize memory settings for inference
    
    Call this once at the beginning of inference to optimize memory usage.
    """
    # Set memory fraction for GPU (if available)
    if torch.cuda.is_available():
        # Don't pre-allocate all GPU memory
        torch.cuda.empty_cache()
        
        # Set memory growth for TensorFlow if available
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except ImportError:
            pass  # TensorFlow not available
    
    # Set optimal garbage collection threshold
    gc.set_threshold(700, 10, 10)  # More frequent collection
    
    print("Memory optimization for inference completed")


def get_memory_summary() -> str:
    """
    Get a formatted summary of current memory usage
    
    Returns:
        Formatted string with memory information
    """
    memory_info = get_memory_info()
    
    summary = "Memory Usage Summary:\n"
    summary += "=" * 25 + "\n"
    
    # System memory
    summary += f"System Memory:\n"
    summary += f"  Total: {memory_info['system_total']:.1f} MB\n"
    summary += f"  Used:  {memory_info['system_used']:.1f} MB ({memory_info['system_percent']:.1f}%)\n"
    summary += f"  Free:  {memory_info['system_available']:.1f} MB\n\n"
    
    # GPU memory
    if torch.cuda.is_available():
        summary += f"GPU Memory:\n"
        for i in range(torch.cuda.device_count()):
            total = memory_info.get(f'gpu_{i}_total', 0)
            allocated = memory_info.get(f'gpu_{i}_allocated', 0)
            reserved = memory_info.get(f'gpu_{i}_reserved', 0)
            percent = (allocated / total * 100) if total > 0 else 0
            
            summary += f"  GPU {i}:\n"
            summary += f"    Total:     {total:.1f} MB\n"
            summary += f"    Allocated: {allocated:.1f} MB ({percent:.1f}%)\n"
            summary += f"    Reserved:  {reserved:.1f} MB\n"
    else:
        summary += "GPU: Not available\n"
    
    return summary