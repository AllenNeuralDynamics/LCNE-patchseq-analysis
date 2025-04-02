import numpy as np

def five_point_stencil_derivative(x, dt):
    """
    Compute the derivative of a discrete signal x with corresponding time t
    using the five-point centered difference for interior points.
    
    For the boundaries, a simple two-point difference is used.
    
    Parameters:
        x : array-like
            Discrete signal values.
        dt : float
            Time step.
            
    Returns:
        dx : numpy array
            Approximated derivative of x with respect to t.
    """
    x = np.array(x)
    
    # Initialize the derivative array
    dx = np.zeros_like(x)
    
    # Use the five-point centered difference for interior points
    # This is valid for indices 2 to len(x)-3
    for i in range(2, len(x) - 2):
        dx[i] = (-x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2]) / (12 * dt)
    
    # For the boundaries where the five-point formula cannot be applied, use lower-order differences:
    dx[0] = (x[1] - x[0]) / dt  # forward difference
    dx[1] = (x[2] - x[1]) / dt
    dx[-2] = (x[-2] - x[-3]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    
    return dx