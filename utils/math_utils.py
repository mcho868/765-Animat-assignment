"""
Math utility functions for the Animat simulation.
Vector operations, collision detection, etc.
"""
import numpy as np

def normalize_vector(vector):
    """Normalize a vector to unit length.
    
    Args:
        vector: (x, y) vector to normalize
        
    Returns:
        Normalized vector as (x, y) tuple
    """
    length = np.sqrt(vector[0]**2 + vector[1]**2)
    if length > 0:
        return (vector[0] / length, vector[1] / length)
    return (0, 0)

def vector_length(vector):
    """Calculate the length of a vector.
    
    Args:
        vector: (x, y) vector
        
    Returns:
        Length of the vector
    """
    return np.sqrt(vector[0]**2 + vector[1]**2)

def vector_to_angle(vector):
    """Convert a 2D vector to an angle in degrees.
    
    Args:
        vector: (x, y) vector
        
    Returns:
        Angle in degrees [0, 360)
    """
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    return angle % 360

def angle_to_vector(angle):
    """Convert an angle in degrees to a unit vector.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Unit vector as (x, y) tuple
    """
    radians = angle * np.pi / 180
    return (np.cos(radians), np.sin(radians))

def rotate_vector(vector, angle):
    """Rotate a vector by an angle in degrees.
    
    Args:
        vector: (x, y) vector to rotate
        angle: Angle in degrees
        
    Returns:
        Rotated vector as (x, y) tuple
    """
    radians = angle * np.pi / 180
    cos_angle = np.cos(radians)
    sin_angle = np.sin(radians)
    
    return (
        vector[0] * cos_angle - vector[1] * sin_angle,
        vector[0] * sin_angle + vector[1] * cos_angle
    )

def distance(point1, point2):
    """Calculate the Euclidean distance between two points.
    
    Args:
        point1: (x, y) first point
        point2: (x, y) second point
        
    Returns:
        Distance between the points
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def collision_circle_circle(center1, radius1, center2, radius2):
    """Check if two circles are colliding.
    
    Args:
        center1: (x, y) center of first circle
        radius1: Radius of first circle
        center2: (x, y) center of second circle
        radius2: Radius of second circle
        
    Returns:
        True if circles are colliding, False otherwise
    """
    # Calculate distance between centers
    dist = distance(center1, center2)
    
    # Circles collide if distance is less than sum of radii
    return dist < (radius1 + radius2) 