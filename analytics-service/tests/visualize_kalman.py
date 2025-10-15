"""
Kalman Filter Visualization
Demonstrates the filtering effect with visual plots
"""

import sys
import os

# Add parent directory to path to import analytics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from analytics.kalman import apply_kalman_filter, calculate_noise_reduction


def plot_basic_kalman_filtering():
    """Visualize basic Kalman filtering on battery voltage data"""
    # Realistic battery voltage readings with noise
    measurements = [3.7, 3.71, 3.69, 3.72, 3.70, 3.68, 3.73, 3.67, 3.74, 3.66, 
                   3.75, 3.65, 3.76, 3.64, 3.77, 3.63, 3.78, 3.62, 3.79, 3.61, 
                   3.80, 3.60, 3.81, 3.59, 3.82]
    
    filtered = apply_kalman_filter(measurements)
    stats = calculate_noise_reduction(measurements, filtered)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot
    time_steps = range(len(measurements))
    plt.plot(time_steps, measurements, 'b.-', label='Raw Measurements', alpha=0.6, linewidth=1, markersize=8)
    plt.plot(time_steps, filtered, 'r-', label='Kalman Filtered', linewidth=2)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Kalman Filter: Battery Voltage Smoothing', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    textstr = f'Noise Reduction: {stats["varianceReduction"]:.1f}%\n'
    textstr += f'Original Variance: {stats["originalVariance"]:.4f}\n'
    textstr += f'Filtered Variance: {stats["filteredVariance"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('kalman_basic_filtering.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kalman_basic_filtering.png")
    plt.show()


def plot_spike_handling():
    """Visualize how Kalman filter handles outliers/spikes"""
    measurements = [3.7, 3.71, 3.69, 5.0, 3.70, 3.68, 3.73, 3.67, 3.74, 3.66]
    filtered = apply_kalman_filter(measurements)
    
    plt.figure(figsize=(12, 6))
    
    time_steps = range(len(measurements))
    plt.plot(time_steps, measurements, 'b.-', label='Raw Measurements (with spike)', 
             alpha=0.6, linewidth=1, markersize=10)
    plt.plot(time_steps, filtered, 'r-', label='Kalman Filtered', linewidth=2)
    
    # Highlight the spike
    plt.scatter([3], [measurements[3]], color='orange', s=200, zorder=5, 
                edgecolors='red', linewidth=2, label='Outlier/Spike')
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Kalman Filter: Outlier/Spike Handling', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_spike_handling.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kalman_spike_handling.png")
    plt.show()


def plot_battery_discharge():
    """Visualize Kalman filter on realistic battery discharge pattern"""
    np.random.seed(42)
    
    # Simulate battery discharge from 4.2V to 3.0V with noise
    actual_voltage = np.linspace(4.2, 3.0, 50)
    noise = np.random.normal(0, 0.05, 50)
    measurements = (actual_voltage + noise).tolist()
    
    filtered = apply_kalman_filter(measurements)
    stats = calculate_noise_reduction(measurements, filtered)
    
    plt.figure(figsize=(14, 7))
    
    time_steps = np.arange(len(measurements))
    
    # Plot ideal discharge curve
    plt.plot(time_steps, actual_voltage, 'g--', label='Ideal Discharge', 
             linewidth=2, alpha=0.7)
    
    # Plot noisy measurements
    plt.plot(time_steps, measurements, 'b.', label='Noisy Measurements', 
             alpha=0.4, markersize=6)
    
    # Plot filtered result
    plt.plot(time_steps, filtered, 'r-', label='Kalman Filtered', linewidth=2.5)
    
    plt.xlabel('Time (arbitrary units)', fontsize=12)
    plt.ylabel('Battery Voltage (V)', fontsize=12)
    plt.title('Kalman Filter: Battery Discharge Pattern', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    textstr = f'Noise Reduction: {stats["varianceReduction"]:.1f}%\n'
    textstr += f'Original Std Dev: {stats["originalStdDev"]:.4f} V\n'
    textstr += f'Filtered Std Dev: {stats["filteredStdDev"]:.4f} V'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.02, 0.35, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('kalman_battery_discharge.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kalman_battery_discharge.png")
    plt.show()


def plot_comparison_different_noise_levels():
    """Compare Kalman filter performance with different noise levels"""
    np.random.seed(42)
    
    # Generate signal with different noise levels
    true_signal = np.sin(np.linspace(0, 4*np.pi, 100)) * 2 + 5
    
    low_noise = true_signal + np.random.normal(0, 0.1, 100)
    medium_noise = true_signal + np.random.normal(0, 0.3, 100)
    high_noise = true_signal + np.random.normal(0, 0.5, 100)
    
    # Apply Kalman filter
    filtered_low = apply_kalman_filter(low_noise.tolist())
    filtered_medium = apply_kalman_filter(medium_noise.tolist())
    filtered_high = apply_kalman_filter(high_noise.tolist())
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time_steps = np.arange(100)
    
    # Low noise
    axes[0].plot(time_steps, true_signal, 'g--', label='True Signal', linewidth=2, alpha=0.7)
    axes[0].plot(time_steps, low_noise, 'b.', label='Noisy', alpha=0.3, markersize=4)
    axes[0].plot(time_steps, filtered_low, 'r-', label='Filtered', linewidth=2)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('Low Noise (σ=0.1)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Medium noise
    axes[1].plot(time_steps, true_signal, 'g--', label='True Signal', linewidth=2, alpha=0.7)
    axes[1].plot(time_steps, medium_noise, 'b.', label='Noisy', alpha=0.3, markersize=4)
    axes[1].plot(time_steps, filtered_medium, 'r-', label='Filtered', linewidth=2)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Medium Noise (σ=0.3)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # High noise
    axes[2].plot(time_steps, true_signal, 'g--', label='True Signal', linewidth=2, alpha=0.7)
    axes[2].plot(time_steps, high_noise, 'b.', label='Noisy', alpha=0.3, markersize=4)
    axes[2].plot(time_steps, filtered_high, 'r-', label='Filtered', linewidth=2)
    axes[2].set_xlabel('Time Step', fontsize=11)
    axes[2].set_ylabel('Value', fontsize=11)
    axes[2].set_title('High Noise (σ=0.5)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Filter Performance vs Noise Levels', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('kalman_noise_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kalman_noise_comparison.png")
    plt.show()


def plot_multi_variable():
    """Visualize multi-variable Kalman filtering"""
    from analytics.kalman import apply_multi_variable_kalman
    
    np.random.seed(42)
    
    # Generate correlated sensor data
    time_steps = np.arange(30)
    
    # Voltage decreases as battery drains
    voltage_true = 4.2 - 0.02 * time_steps
    voltage_data = (voltage_true + np.random.normal(0, 0.05, 30)).tolist()
    
    # Capacity decreases proportionally
    capacity_true = 3000 - 20 * time_steps
    capacity_data = (capacity_true + np.random.normal(0, 50, 30)).tolist()
    
    # Temperature varies slightly
    temperature_true = 25 + 0.1 * np.sin(time_steps * 0.5)
    temperature_data = (temperature_true + np.random.normal(0, 0.5, 30)).tolist()
    
    # Apply multi-variable Kalman filter
    result = apply_multi_variable_kalman(voltage_data, capacity_data, temperature_data)
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Voltage
    axes[0].plot(time_steps, voltage_data, 'b.', label='Noisy', alpha=0.5, markersize=8)
    axes[0].plot(time_steps, result['voltage'], 'r-', label='Filtered', linewidth=2)
    axes[0].set_ylabel('Voltage (V)', fontsize=11)
    axes[0].set_title('Battery Voltage', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Capacity
    axes[1].plot(time_steps, capacity_data, 'b.', label='Noisy', alpha=0.5, markersize=8)
    axes[1].plot(time_steps, result['capacity'], 'r-', label='Filtered', linewidth=2)
    axes[1].set_ylabel('Capacity (mAh)', fontsize=11)
    axes[1].set_title('Battery Capacity', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Temperature
    axes[2].plot(time_steps, temperature_data, 'b.', label='Noisy', alpha=0.5, markersize=8)
    axes[2].plot(time_steps, result['temperature'], 'r-', label='Filtered', linewidth=2)
    axes[2].set_xlabel('Time Step', fontsize=11)
    axes[2].set_ylabel('Temperature (°C)', fontsize=11)
    axes[2].set_title('Battery Temperature', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Variable Kalman Filter: Battery Telemetry', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('kalman_multi_variable.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kalman_multi_variable.png")
    plt.show()


def main():
    """Run all visualizations"""
    print("\n" + "="*70)
    print("KALMAN FILTER VISUALIZATIONS")
    print("="*70 + "\n")
    
    print("Generating visualization 1: Basic Kalman Filtering...")
    plot_basic_kalman_filtering()
    
    print("\nGenerating visualization 2: Spike/Outlier Handling...")
    plot_spike_handling()
    
    print("\nGenerating visualization 3: Battery Discharge Pattern...")
    plot_battery_discharge()
    
    print("\nGenerating visualization 4: Noise Level Comparison...")
    plot_comparison_different_noise_levels()
    
    print("\nGenerating visualization 5: Multi-Variable Filtering...")
    plot_multi_variable()
    
    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - kalman_basic_filtering.png")
    print("  - kalman_spike_handling.png")
    print("  - kalman_battery_discharge.png")
    print("  - kalman_noise_comparison.png")
    print("  - kalman_multi_variable.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
