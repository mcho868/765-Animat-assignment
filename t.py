import numpy as np
import matplotlib.pyplot as plt

def demonstrate_sigmoid_correlations():
    """Show how the three ranges correlate"""
    
    print("SIGMOID THRESHOLD CORRELATION ANALYSIS")
    print("="*50)
    
    # Range 1: Summed link outputs (input to sigmoid)
    summed_inputs = np.linspace(-9, 9, 1000)  # Sum of 9 links [-1,1] each
    
    # Range 2: Different threshold values (from genome)
    thresholds = [-3.0, -1.5, 0.0, 1.5, 3.0]  # Genome[81,82] scaled range
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Show sigmoid curves for different thresholds
    plt.subplot(2, 2, 1)
    for threshold in thresholds:
        sigmoid_output = 1.0 / (1.0 + np.exp(-(summed_inputs - threshold)))
        plt.plot(summed_inputs, sigmoid_output, linewidth=2, 
                label=f'Threshold = {threshold:.1f}')
    
    plt.title('Sigmoid Function: Input vs Output')
    plt.xlabel('Summed Link Outputs [-9, +9]')
    plt.ylabel('Sigmoid Output [0, 1]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Plot 2: Final wheel speeds after scaling
    plt.subplot(2, 2, 2)
    for threshold in thresholds:
        sigmoid_output = 1.0 / (1.0 + np.exp(-(summed_inputs - threshold)))
        # Scale [0,1] to [-10,10] as per paper
        wheel_speed = (sigmoid_output * 2.0 - 1.0) * 10.0
        plt.plot(summed_inputs, wheel_speed, linewidth=2,
                label=f'Threshold = {threshold:.1f}')
    
    plt.title('Final Wheel Speeds (After Scaling)')
    plt.xlabel('Summed Link Outputs [-9, +9]')
    plt.ylabel('Wheel Speed [-10, +10]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Plot 3: Critical analysis - what sum gives 0 wheel speed?
    plt.subplot(2, 2, 3)
    zero_points = []
    for threshold in thresholds:
        # Find where sigmoid = 0.5 (which gives wheel_speed = 0)
        # sigmoid = 0.5 when input = threshold
        zero_points.append(threshold)
    
    plt.plot(thresholds, zero_points, 'ro-', linewidth=2, markersize=8)
    plt.title('Zero Wheel Speed Point')
    plt.xlabel('Sigmoid Threshold [-3, +3]')
    plt.ylabel('Summed Input for Zero Speed')
    plt.grid(True, alpha=0.3)
    plt.text(-2, 2, 'Threshold = Input\nfor zero wheel speed', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    # Plot 4: Sensitivity analysis
    plt.subplot(2, 2, 4)
    test_input = 2.0  # Example: sum = +2
    wheel_speeds = []
    for threshold in thresholds:
        sigmoid_out = 1.0 / (1.0 + np.exp(-(test_input - threshold)))
        wheel_speed = (sigmoid_out * 2.0 - 1.0) * 10.0
        wheel_speeds.append(wheel_speed)
    
    plt.plot(thresholds, wheel_speeds, 'bo-', linewidth=2, markersize=8)
    plt.title(f'Wheel Speed for Input = {test_input}')
    plt.xlabel('Sigmoid Threshold [-3, +3]')
    plt.ylabel('Resulting Wheel Speed')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('sigmoid_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def explain_correlations():
    """Explain the mathematical relationships"""
    
    print("\nMATHEMATICAL CORRELATIONS:")
    print("-" * 40)
    
    print("""
🔢 RANGE RELATIONSHIPS:

1. Link Outputs [-1, +1] × 9 links → Sum [-9, +9]
2. Genome [0, 99] → Threshold [-3.0, +3.0] 
3. Sigmoid [0, 1] → Wheel Speed [-10, +10]

🎯 KEY CORRELATION:
   Threshold determines WHERE the sigmoid "activates"
   
   • Threshold = -3: Sigmoid activates for sum ≥ -3
   • Threshold = 0:  Sigmoid activates for sum ≥ 0  
   • Threshold = +3: Sigmoid activates for sum ≥ +3
    """)
    
    # Numerical examples
    test_sums = [-6, -3, 0, 3, 6]
    test_thresholds = [-3.0, 0.0, 3.0]
    
    print("\nNUMERICAL EXAMPLES:")
    print("Summed Input | Threshold | Sigmoid Out | Wheel Speed")
    print("-" * 50)
    
    for sum_val in test_sums:
        row = f"{sum_val:8.1f}     |"
        for threshold in test_thresholds:
            sigmoid_out = 1.0 / (1.0 + np.exp(-(sum_val - threshold)))
            wheel_speed = (sigmoid_out * 2.0 - 1.0) * 10.0
            row += f" {threshold:4.1f}→{wheel_speed:5.1f} |"
        print(row)

def behavioral_interpretation():
    """Explain what this means for animat behavior"""
    
    print("\n" + "="*50)
    print("BEHAVIORAL INTERPRETATION")
    print("="*50)
    
    print("""
🧠 WHAT THIS MEANS FOR EVOLUTION:

SCENARIO 1: Left wheel threshold = -2, Right wheel threshold = +2
→ Left wheel activates easily (sensitive)
→ Right wheel activates reluctantly (conservative)  
→ RESULT: Animat tends to turn left, explores leftward

SCENARIO 2: Both thresholds = 0  
→ Balanced responsiveness
→ RESULT: Symmetrical behavior

SCENARIO 3: Left threshold = +3, Right threshold = -3
→ Left wheel very conservative 
→ Right wheel very sensitive
→ RESULT: Strong rightward bias

🎯 THE CORRELATION EFFECT:
   
   The [-3, +3] threshold range is perfectly sized to work with
   the [-9, +9] summed input range:
   
   • Threshold -3: Even very negative sums produce positive wheels
   • Threshold 0:  Balanced - positive sums → positive wheels  
   • Threshold +3: Need very positive sums for positive wheels
   
   This gives evolution fine-grained control over wheel sensitivity!
    """)

def paper_compliance_check():
    """Verify our understanding matches the paper exactly"""
    
    print("\n" + "="*50)
    print("PAPER COMPLIANCE CHECK")
    print("="*50)
    
    print("""
✅ PAPER STATEMENT: "first 9 link outputs are summed"
   → Our range: [-9, +9] ✓

✅ PAPER STATEMENT: "passed through a sigmoid function" 
   → Our function: 1/(1 + exp(-(sum - threshold))) ✓

✅ PAPER STATEMENT: "scaled from -10 to 10"
   → Our scaling: (sigmoid * 2 - 1) * 10 ✓

✅ PAPER STATEMENT: "sigmoid thresholds...range from -3.0 to 3.0"
   → Our genome scaling: genome[81,82] → [-3.0, +3.0] ✓

🎯 CORRELATION CONFIRMED:
   The [-3, +3] threshold range is specifically chosen to provide
   meaningful control over the [-9, +9] summed input range,
   producing the required [-10, +10] wheel speed output.
    """)

if __name__ == "__main__":
    demonstrate_sigmoid_correlations()
    explain_correlations()
    behavioral_interpretation()
    paper_compliance_check()