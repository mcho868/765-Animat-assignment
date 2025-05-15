"""
Mathematical utility functions for the simulation environment.
"""
import math
import numpy as np

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def normalize_vector(v):
    """Normalize a vector to have unit length."""
    magnitude = math.sqrt(v[0]**2 + v[1]**2)
    if magnitude == 0:
        return (0, 0)
    return (v[0] / magnitude, v[1] / magnitude)

def angle_to_vector(angle_degrees):
    """Convert an angle in degrees to a unit vector."""
    angle_radians = math.radians(angle_degrees)
    return (math.cos(angle_radians), math.sin(angle_radians))

def vector_to_angle(v):
    """Convert a vector to an angle in degrees."""
    return math.degrees(math.atan2(v[1], v[0]))

def rotate_vector(v, angle_degrees):
    """Rotate a vector by a given angle in degrees."""
    angle_radians = math.radians(angle_degrees)
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    return (v[0] * cos_theta - v[1] * sin_theta, 
            v[0] * sin_theta + v[1] * cos_theta)

def check_circle_collision(p1, r1, p2, r2):
    """Check if two circles are colliding."""
    return distance(p1, p2) < (r1 + r2)

def limit_magnitude(v, max_val):
    """Limit the magnitude of a vector to max_val."""
    magnitude = math.sqrt(v[0]**2 + v[1]**2)
    if magnitude > max_val:
        scaling_factor = max_val / magnitude
        return (v[0] * scaling_factor, v[1] * scaling_factor)
    return v

def lerp(a, b, t):
    """Linear interpolation between a and b with parameter t."""
    return a + t * (b - a)

def ray_circle_intersection(ray_origin, ray_direction, circle_center, circle_radius):
    """Check if a ray intersects with a circle and return the distance if it does."""
    oc = (ray_origin[0] - circle_center[0], ray_origin[1] - circle_center[1])
    a = ray_direction[0]**2 + ray_direction[1]**2
    b = 2 * (oc[0] * ray_direction[0] + oc[1] * ray_direction[1])
    c = oc[0]**2 + oc[1]**2 - circle_radius**2
    
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return None  # No intersection
    
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)
    
    if t1 < 0 and t2 < 0:
        return None  # Both intersections are behind the ray
    
    t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2)
    
    if t < 0:
        return None  # Intersection is behind the ray
        
    return t  # Return distance to intersection 