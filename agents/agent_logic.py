"""
Decision-making logic for agents in the simulation.
Implements rule-based systems and/or behavior trees for agent control.
"""
import random
from enum import Enum

class RuleCondition:
    """A simple condition class for rule-based systems."""
    def __init__(self, attribute, operator, value):
        self.attribute = attribute
        self.operator = operator
        self.value = value
        
    def evaluate(self, agent, sensor_readings, environment):
        """Evaluate this condition on the given agent and environment."""
        # Get the actual value to compare
        if self.attribute.startswith("sensor."):
            sensor_name = self.attribute[7:]  # Remove "sensor." prefix
            actual_value = sensor_readings.get(sensor_name, 0)
        elif self.attribute.startswith("agent."):
            attr_name = self.attribute[6:]  # Remove "agent." prefix
            actual_value = getattr(agent, attr_name, 0)
        else:
            return False
            
        # Apply the operator
        if self.operator == "==":
            return actual_value == self.value
        elif self.operator == "!=":
            return actual_value != self.value
        elif self.operator == ">":
            return actual_value > self.value
        elif self.operator == ">=":
            return actual_value >= self.value
        elif self.operator == "<":
            return actual_value < self.value
        elif self.operator == "<=":
            return actual_value <= self.value
        else:
            return False

class Rule:
    """A simple rule class for rule-based systems."""
    def __init__(self, conditions, actions, priority=0):
        self.conditions = conditions  # List of conditions
        self.actions = actions  # Dictionary of motor activations
        self.priority = priority
        
    def matches(self, agent, sensor_readings, environment):
        """Check if all conditions of this rule match."""
        return all(condition.evaluate(agent, sensor_readings, environment) for condition in self.conditions)

class RuleBasedSystem:
    """A simple rule-based system for agent control."""
    def __init__(self):
        self.rules = []
        
    def add_rule(self, rule):
        """Add a rule to the system."""
        self.rules.append(rule)
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        
    def decide(self, agent, sensor_readings, environment):
        """Run the rule-based system to decide on motor activations."""
        # Check each rule in order of priority
        for rule in self.rules:
            if rule.matches(agent, sensor_readings, environment):
                return rule.actions
                
        # Default behavior if no rules match
        return {"left_motor": 0.5, "right_motor": 0.5}

class BehaviorNode:
    """Base class for behavior tree nodes."""
    def execute(self, agent, sensor_readings, environment):
        """Execute this node. Must be implemented by subclasses."""
        pass

class ActionNode(BehaviorNode):
    """A node that performs a specific action."""
    def __init__(self, action_func):
        self.action_func = action_func
        
    def execute(self, agent, sensor_readings, environment):
        """Execute the action function."""
        return self.action_func(agent, sensor_readings, environment)

class ConditionNode(BehaviorNode):
    """A node that checks a condition."""
    def __init__(self, condition_func):
        self.condition_func = condition_func
        
    def execute(self, agent, sensor_readings, environment):
        """Check the condition function."""
        return self.condition_func(agent, sensor_readings, environment)

class SequenceNode(BehaviorNode):
    """A node that executes its children in sequence until one fails."""
    def __init__(self, children):
        self.children = children
        
    def execute(self, agent, sensor_readings, environment):
        """Execute all children in sequence."""
        for child in self.children:
            result = child.execute(agent, sensor_readings, environment)
            if not result:
                return False
        return True

class SelectorNode(BehaviorNode):
    """A node that executes its children in sequence until one succeeds."""
    def __init__(self, children):
        self.children = children
        
    def execute(self, agent, sensor_readings, environment):
        """Execute children until one succeeds."""
        for child in self.children:
            result = child.execute(agent, sensor_readings, environment)
            if result:
                return True
        return False

class BehaviorTree:
    """A simple behavior tree for agent control."""
    def __init__(self, root_node):
        self.root = root_node
        self.last_result = None
        
    def decide(self, agent, sensor_readings, environment):
        """Run the behavior tree to decide on motor activations."""
        self.last_result = self.root.execute(agent, sensor_readings, environment)
        return self.last_result
        
# Predefined behavior functions for easy reuse

def seek_light(agent, sensor_readings, environment):
    """Behavior to seek light sources."""
    left_light = sensor_readings.get("left_light", 0)
    right_light = sensor_readings.get("right_light", 0)
    
    # Normalize readings
    max_reading = max(1, max(left_light, right_light))
    left_norm = left_light / max_reading
    right_norm = right_light / max_reading
    
    # Cross-wiring: right sensor controls left motor and vice versa
    left_motor = right_norm
    right_motor = left_norm
    
    # Add noise to prevent getting stuck
    left_motor = max(0.1, min(1.0, left_motor + random.uniform(-0.1, 0.1)))
    right_motor = max(0.1, min(1.0, right_motor + random.uniform(-0.1, 0.1)))
    
    return {"left_motor": left_motor, "right_motor": right_motor}

def avoid_light(agent, sensor_readings, environment):
    """Behavior to avoid light sources."""
    left_light = sensor_readings.get("left_light", 0)
    right_light = sensor_readings.get("right_light", 0)
    
    # Normalize readings
    max_reading = max(1, max(left_light, right_light))
    left_norm = left_light / max_reading
    right_norm = right_light / max_reading
    
    # Direct-wiring: left sensor controls left motor and vice versa
    left_motor = left_norm
    right_motor = right_norm
    
    # Add noise to prevent getting stuck
    left_motor = max(0.1, min(1.0, left_motor + random.uniform(-0.1, 0.1)))
    right_motor = max(0.1, min(1.0, right_motor + random.uniform(-0.1, 0.1)))
    
    return {"left_motor": left_motor, "right_motor": right_motor}

def random_movement(agent, sensor_readings, environment):
    """Behavior for random movement."""
    left_motor = random.uniform(0.2, 1.0)
    right_motor = random.uniform(0.2, 1.0)
    
    return {"left_motor": left_motor, "right_motor": right_motor}

def move_forward(agent, sensor_readings, environment):
    """Behavior to move forward at full speed."""
    return {"left_motor": 1.0, "right_motor": 1.0}

def stop(agent, sensor_readings, environment):
    """Behavior to stop all movement."""
    return {"left_motor": 0.0, "right_motor": 0.0}

def is_stressed(agent, sensor_readings, environment, threshold=50):
    """Condition to check if agent is stressed."""
    return agent.stress > threshold

def is_hungry(agent, sensor_readings, environment, threshold=30):
    """Condition to check if agent is hungry."""
    return agent.energy < threshold

def light_detected(agent, sensor_readings, environment, threshold=20):
    """Condition to check if light is detected."""
    left_light = sensor_readings.get("left_light", 0)
    right_light = sensor_readings.get("right_light", 0)
    return max(left_light, right_light) > threshold 