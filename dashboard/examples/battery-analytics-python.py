#!/usr/bin/env python3
"""
Battery Analytics - Python Implementation with FilterPy
Advanced Kalman Filtering for IoT Battery Monitoring

This implementation uses the filterpy library for professional-grade
Kalman filtering with enhanced visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import json

# ============================================================================
# KALMAN FILTER IMPLEMENTATION (using filterpy)
# ============================================================================

def create_battery_kalman_filter(dt=1.0, process_noise=0.01, measurement_noise=0.1):
    """
    Create a Kalman filter for battery voltage/capacity monitoring
    
    State vector: [value, velocity]
    - value: current voltage/capacity
    - velocity: rate of change (degradation rate)
    
    Args:
        dt: Time step (default 1.0)
        process_noise: Process noise covariance Q
        measurement_noise: Measurement noise covariance R
    
    Returns:
        KalmanFilter object
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([[1., dt],
                     [0., 1.]])
    
    # Measurement function (we only measure position, not velocity)
    kf.H = np.array([[1., 0.]])
    
    # Measurement noise covariance
    kf.R = np.array([[measurement_noise]])
    
    # Process noise covariance (using white noise model)
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise)
    
    # Initial state covariance
    kf.P *= 1000.
    
    return kf


def apply_kalman_filter_voltage(measurements, initial_estimate=3.7, 
                                process_noise=0.01, measurement_noise=0.1):
    """
    Apply Kalman filter to voltage measurements
    
    Args:
        measurements: Array of voltage measurements
        initial_estimate: Initial voltage estimate
        process_noise: Process noise (Q)
        measurement_noise: Measurement noise (R)
    
    Returns:
        dict with filtered values, velocities, and covariances
    """
    if len(measurements) == 0:
        return {'filtered': [], 'velocity': [], 'covariance': []}
    
    # Create Kalman filter
    kf = create_battery_kalman_filter(dt=1.0, 
                                      process_noise=process_noise,
                                      measurement_noise=measurement_noise)
    
    # Initialize state
    kf.x = np.array([[measurements[0]], [0.]])
    
    # Storage for results
    filtered_values = []
    velocities = []
    covariances = []
    
    for z in measurements:
        # Predict step
        kf.predict()
        
        # Update step
        kf.update(z)
        
        # Store results
        filtered_values.append(kf.x[0, 0])
        velocities.append(kf.x[1, 0])
        covariances.append(kf.P[0, 0])
    
    return {
        'filtered': np.array(filtered_values),
        'velocity': np.array(velocities),
        'covariance': np.array(covariances),
        'original': np.array(measurements)
    }


# ============================================================================
# Z-SCORE ANALYSIS
# ============================================================================

def calculate_zscore_analysis(current_value, historical_values, metric_name='voltage'):
    """
    Calculate Z-score and perform anomaly detection
    
    Args:
        current_value: Current measurement
        historical_values: Array of historical measurements
        metric_name: Name of the metric
    
    Returns:
        dict with Z-score analysis results
    """
    if len(historical_values) == 0:
        return None
    
    mean = np.mean(historical_values)
    std_dev = np.std(historical_values, ddof=1)
    
    if std_dev == 0:
        z_score = 0
    else:
        z_score = (current_value - mean) / std_dev
    
    # Classify severity
    abs_z = abs(z_score)
    if abs_z > 3.0:
        severity = 'critical'
    elif abs_z > 2.0:
        severity = 'warning'
    else:
        severity = 'normal'
    
    return {
        'metric': metric_name,
        'value': current_value,
        'mean': mean,
        'std_dev': std_dev,
        'z_score': z_score,
        'is_anomaly': abs_z > 2.0,
        'severity': severity,
        'confidence': stats.norm.cdf(abs_z) * 2 - 1  # Two-tailed confidence
    }


# ============================================================================
# BATTERY DATA SIMULATION
# ============================================================================

def generate_battery_degradation_data(days=90, samples_per_day=288, 
                                     initial_voltage=3.7, 
                                     degradation_rate=-0.001,
                                     noise_level=0.05):
    """
    Generate realistic battery degradation data with noise
    
    Args:
        days: Number of days to simulate
        samples_per_day: Samples per day (288 = every 5 minutes)
        initial_voltage: Starting voltage
        degradation_rate: Daily degradation rate
        noise_level: Measurement noise standard deviation
    
    Returns:
        DataFrame with timestamps and measurements
    """
    total_samples = days * samples_per_day
    
    # Time array
    timestamps = [datetime.now() - timedelta(days=days) + 
                  timedelta(minutes=i*5) for i in range(total_samples)]
    
    # Generate true voltage (with degradation)
    day_indices = np.arange(total_samples) / samples_per_day
    true_voltage = initial_voltage + degradation_rate * day_indices
    
    # Add realistic noise (measurement error)
    measurement_noise = np.random.normal(0, noise_level, total_samples)
    measured_voltage = true_voltage + measurement_noise
    
    # Add occasional spikes (sensor glitches)
    spike_indices = np.random.choice(total_samples, size=int(total_samples * 0.01), replace=False)
    measured_voltage[spike_indices] += np.random.uniform(-0.3, 0.3, len(spike_indices))
    
    # Generate capacity data (correlated with voltage)
    true_capacity = 100 - (100 - ((true_voltage - 2.8) / (4.2 - 2.8) * 100))
    measured_capacity = true_capacity + np.random.normal(0, 2, total_samples)
    measured_capacity = np.clip(measured_capacity, 0, 100)
    
    # Generate temperature data
    temperature = 25 + np.random.normal(0, 5, total_samples) + \
                  np.sin(day_indices * 2 * np.pi) * 3  # Daily temperature cycle
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'true_voltage': true_voltage,
        'measured_voltage': measured_voltage,
        'true_capacity': true_capacity,
        'measured_capacity': measured_capacity,
        'temperature': temperature
    })


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_kalman_filter_comparison(data, kalman_result, metric='voltage'):
    """
    Create comprehensive visualization of Kalman filter performance
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Kalman Filter Analysis - Battery {metric.capitalize()}', 
                 fontsize=16, fontweight='bold')
    
    time_steps = np.arange(len(data))
    
    # Plot 1: Original vs Filtered
    ax1 = axes[0]
    ax1.plot(time_steps, kalman_result['original'], 'b.', 
             alpha=0.3, markersize=2, label='Noisy Measurements')
    ax1.plot(time_steps, kalman_result['filtered'], 'r-', 
             linewidth=2, label='Kalman Filtered')
    if 'true_voltage' in data.columns:
        ax1.plot(time_steps, data['true_voltage'], 'g--', 
                linewidth=1.5, alpha=0.7, label='True Value')
    ax1.set_ylabel('Voltage (V)', fontsize=11)
    ax1.set_title('Original vs Filtered Signal', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Estimation Error
    ax2 = axes[1]
    if 'true_voltage' in data.columns:
        measurement_error = kalman_result['original'] - data['true_voltage']
        filter_error = kalman_result['filtered'] - data['true_voltage']
        
        ax2.plot(time_steps, measurement_error, 'b.', 
                alpha=0.3, markersize=2, label='Measurement Error')
        ax2.plot(time_steps, filter_error, 'r-', 
                linewidth=1.5, label='Filter Error')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Show error statistics
        measurement_rmse = np.sqrt(np.mean(measurement_error**2))
        filter_rmse = np.sqrt(np.mean(filter_error**2))
        improvement = (1 - filter_rmse/measurement_rmse) * 100
        
        ax2.text(0.02, 0.98, 
                f'Measurement RMSE: {measurement_rmse:.4f}V\n'
                f'Filter RMSE: {filter_rmse:.4f}V\n'
                f'Improvement: {improvement:.1f}%',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        residuals = kalman_result['original'] - kalman_result['filtered']
        ax2.plot(time_steps, residuals, 'b-', linewidth=1, label='Residuals')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel('Error (V)', fontsize=11)
    ax2.set_title('Estimation Error Analysis', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity (Degradation Rate)
    ax3 = axes[2]
    velocity_ma = pd.Series(kalman_result['velocity']).rolling(window=50, center=True).mean()
    ax3.plot(time_steps, kalman_result['velocity'], 'b-', 
            alpha=0.3, linewidth=0.5, label='Instantaneous')
    ax3.plot(time_steps, velocity_ma, 'r-', 
            linewidth=2, label='Moving Average (50 samples)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_ylabel('dV/dt (V/sample)', fontsize=11)
    ax3.set_title('Estimated Degradation Rate (Velocity)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty (Covariance)
    ax4 = axes[3]
    ax4.plot(time_steps, kalman_result['covariance'], 'purple', linewidth=2)
    ax4.set_ylabel('Covariance', fontsize=11)
    ax4.set_xlabel('Sample Number', fontsize=11)
    ax4.set_title('Filter Uncertainty (State Covariance)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add shaded region showing convergence
    convergence_point = np.argmax(kalman_result['covariance'] < 
                                   kalman_result['covariance'][-1] * 2)
    if convergence_point > 0:
        ax4.axvspan(0, convergence_point, alpha=0.2, color='yellow', 
                   label=f'Convergence Phase ({convergence_point} samples)')
        ax4.legend(loc='best')
    
    plt.tight_layout()
    return fig


def plot_noise_reduction_comparison(original, filtered):
    """
    Create detailed noise reduction analysis plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Noise Reduction Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Time domain
    ax1 = axes[0, 0]
    time_steps = np.arange(len(original))
    ax1.plot(time_steps[:200], original[:200], 'b-', 
            alpha=0.5, linewidth=1, label='Original')
    ax1.plot(time_steps[:200], filtered[:200], 'r-', 
            linewidth=2, label='Filtered')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Time Domain (First 200 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram comparison
    ax2 = axes[0, 1]
    ax2.hist(original, bins=50, alpha=0.5, color='blue', label='Original', density=True)
    ax2.hist(filtered, bins=50, alpha=0.5, color='red', label='Filtered', density=True)
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    orig_std = np.std(original)
    filt_std = np.std(filtered)
    reduction = (1 - filt_std/orig_std) * 100
    ax2.text(0.02, 0.98, 
            f'Original σ: {orig_std:.4f}\n'
            f'Filtered σ: {filt_std:.4f}\n'
            f'Reduction: {reduction:.1f}%',
            transform=ax2.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Frequency domain (FFT)
    ax3 = axes[1, 0]
    fft_orig = np.abs(np.fft.fft(original - np.mean(original)))
    fft_filt = np.abs(np.fft.fft(filtered - np.mean(filtered)))
    freqs = np.fft.fftfreq(len(original))
    
    # Only plot positive frequencies
    pos_mask = freqs > 0
    ax3.semilogy(freqs[pos_mask], fft_orig[pos_mask], 'b-', 
                alpha=0.5, label='Original')
    ax3.semilogy(freqs[pos_mask], fft_filt[pos_mask], 'r-', 
                linewidth=2, label='Filtered')
    ax3.set_xlabel('Frequency (normalized)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Frequency Domain (FFT)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax4 = axes[1, 1]
    residuals = original - filtered
    ax4.plot(time_steps, residuals, 'g-', linewidth=1, alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='--')
    ax4.fill_between(time_steps, -2*np.std(residuals), 2*np.std(residuals), 
                     alpha=0.2, color='yellow', label='±2σ')
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Residual (V)')
    ax4.set_title('Filtered Noise (Residuals)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_zscore_analysis(historical_data, current_value, metric_name='voltage'):
    """
    Visualize Z-score analysis and anomaly detection
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Z-Score Anomaly Detection - {metric_name.capitalize()}', 
                 fontsize=16, fontweight='bold')
    
    analysis = calculate_zscore_analysis(current_value, historical_data, metric_name)
    
    # Plot 1: Time series with current value
    ax1 = axes[0, 0]
    time_steps = np.arange(len(historical_data))
    ax1.plot(time_steps, historical_data, 'b-', linewidth=1, label='Historical')
    ax1.axhline(y=analysis['mean'], color='g', linestyle='--', 
               linewidth=2, label=f'Mean = {analysis["mean"]:.3f}')
    ax1.axhline(y=analysis['mean'] + 2*analysis['std_dev'], 
               color='orange', linestyle=':', label='±2σ')
    ax1.axhline(y=analysis['mean'] - 2*analysis['std_dev'], 
               color='orange', linestyle=':')
    ax1.axhline(y=analysis['mean'] + 3*analysis['std_dev'], 
               color='red', linestyle=':', label='±3σ')
    ax1.axhline(y=analysis['mean'] - 3*analysis['std_dev'], 
               color='red', linestyle=':')
    ax1.scatter([len(historical_data)], [current_value], 
               s=200, c='red', marker='*', zorder=5, 
               label=f'Current = {current_value:.3f}')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel(f'{metric_name.capitalize()} (V)')
    ax1.set_title('Time Series Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution with Z-score
    ax2 = axes[0, 1]
    ax2.hist(historical_data, bins=30, density=True, alpha=0.6, color='blue')
    
    # Overlay normal distribution
    x = np.linspace(analysis['mean'] - 4*analysis['std_dev'], 
                   analysis['mean'] + 4*analysis['std_dev'], 100)
    ax2.plot(x, stats.norm.pdf(x, analysis['mean'], analysis['std_dev']), 
            'g-', linewidth=2, label='Normal Distribution')
    
    # Mark current value
    ax2.axvline(x=current_value, color='red', linewidth=3, 
               label=f'Current (Z={analysis["z_score"]:.2f})')
    ax2.set_xlabel(f'{metric_name.capitalize()} (V)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Z-score visualization
    ax3 = axes[1, 0]
    z_scores = [(v - analysis['mean']) / analysis['std_dev'] for v in historical_data]
    ax3.plot(z_scores, 'b-', linewidth=1, alpha=0.7)
    ax3.axhline(y=0, color='g', linestyle='-', linewidth=2)
    ax3.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='Warning (±2σ)')
    ax3.axhline(y=-2, color='orange', linestyle='--', linewidth=1.5)
    ax3.axhline(y=3, color='red', linestyle='--', linewidth=1.5, label='Critical (±3σ)')
    ax3.axhline(y=-3, color='red', linestyle='--', linewidth=1.5)
    ax3.scatter([len(z_scores)], [analysis['z_score']], 
               s=200, c='red', marker='*', zorder=5)
    ax3.fill_between(range(len(z_scores)), -2, 2, alpha=0.1, color='green')
    ax3.fill_between(range(len(z_scores)), -3, -2, alpha=0.1, color='orange')
    ax3.fill_between(range(len(z_scores)), 2, 3, alpha=0.1, color='orange')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Z-Score')
    ax3.set_title('Z-Score Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Determine status color
    status_color = 'green' if analysis['severity'] == 'normal' else \
                   'orange' if analysis['severity'] == 'warning' else 'red'
    
    summary_text = f"""
    Z-SCORE ANALYSIS SUMMARY
    {'='*40}
    
    Metric: {metric_name.capitalize()}
    Current Value: {current_value:.4f}
    
    Statistical Baseline:
      Mean (μ): {analysis['mean']:.4f}
      Std Dev (σ): {analysis['std_dev']:.4f}
      Min: {np.min(historical_data):.4f}
      Max: {np.max(historical_data):.4f}
    
    Z-Score Analysis:
      Z-Score: {analysis['z_score']:.4f}
      Absolute |Z|: {abs(analysis['z_score']):.4f}
      Confidence: {analysis['confidence']:.2%}
    
    Classification:
      Is Anomaly: {analysis['is_anomaly']}
      Severity: {analysis['severity'].upper()}
    
    Interpretation:
      The current value is {abs(analysis['z_score']):.1f} standard
      deviations {'below' if analysis['z_score'] < 0 else 'above'}
      the historical mean.
      
      Status: {analysis['severity'].upper()}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    plt.tight_layout()
    return fig, analysis


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Run comprehensive Kalman filter and Z-score demonstrations
    """
    print("\n" + "="*80)
    print("  BATTERY ANALYTICS - PYTHON IMPLEMENTATION WITH FILTERPY")
    print("  Advanced Kalman Filtering and Statistical Analysis")
    print("="*80 + "\n")
    
    # Generate realistic battery data
    print("📊 Generating realistic battery degradation data...")
    print("   - 90 days of data")
    print("   - 288 samples per day (5-minute intervals)")
    print("   - Total: 25,920 samples")
    print("   - Includes measurement noise and sensor glitches\n")
    
    data = generate_battery_degradation_data(
        days=90,
        samples_per_day=288,
        initial_voltage=3.7,
        degradation_rate=-0.001,
        noise_level=0.05
    )
    
    print(f"✅ Generated {len(data)} samples\n")
    
    # Apply Kalman filter
    print("🔬 Applying Kalman Filter...")
    print("   - State: [voltage, degradation_rate]")
    print("   - Using constant velocity model")
    print("   - Process noise Q = 0.01")
    print("   - Measurement noise R = 0.1\n")
    
    kalman_result = apply_kalman_filter_voltage(
        measurements=data['measured_voltage'].values,
        process_noise=0.01,
        measurement_noise=0.1
    )
    
    # Calculate noise reduction
    orig_variance = np.var(kalman_result['original'])
    filt_variance = np.var(kalman_result['filtered'])
    reduction = (1 - filt_variance/orig_variance) * 100
    
    print(f"✅ Kalman Filter Applied:")
    print(f"   - Original variance: {orig_variance:.6f}")
    print(f"   - Filtered variance: {filt_variance:.6f}")
    print(f"   - Noise reduction: {reduction:.1f}%")
    print(f"   - Average degradation rate: {np.mean(kalman_result['velocity']):.6f} V/sample\n")
    
    # Z-Score Analysis
    print("📈 Performing Z-Score Analysis...")
    
    # Use first 80 days as baseline, analyze last value
    baseline_samples = int(len(data) * 0.89)
    historical = kalman_result['filtered'][:baseline_samples]
    current = kalman_result['filtered'][-1]
    
    zscore_result = calculate_zscore_analysis(current, historical, 'voltage')
    
    print(f"✅ Z-Score Analysis Complete:")
    print(f"   - Historical mean: {zscore_result['mean']:.4f} V")
    print(f"   - Standard deviation: {zscore_result['std_dev']:.4f} V")
    print(f"   - Current value: {zscore_result['value']:.4f} V")
    print(f"   - Z-Score: {zscore_result['z_score']:.4f}")
    print(f"   - Severity: {zscore_result['severity'].upper()}")
    print(f"   - Is Anomaly: {zscore_result['is_anomaly']}\n")
    
    # Generate visualizations
    print("📊 Generating Visualizations...\n")
    
    print("1. Creating Kalman Filter comparison plot...")
    fig1 = plot_kalman_filter_comparison(data, kalman_result)
    fig1.savefig('battery_kalman_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: battery_kalman_analysis.png")
    
    print("2. Creating noise reduction analysis plot...")
    fig2 = plot_noise_reduction_comparison(kalman_result['original'], 
                                           kalman_result['filtered'])
    fig2.savefig('battery_noise_reduction.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: battery_noise_reduction.png")
    
    print("3. Creating Z-score analysis plot...")
    fig3, _ = plot_zscore_analysis(historical, current, 'voltage')
    fig3.savefig('battery_zscore_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: battery_zscore_analysis.png")
    
    # Export data
    print("\n💾 Exporting data...")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'timestamp': data['timestamp'],
        'true_voltage': data['true_voltage'],
        'measured_voltage': data['measured_voltage'],
        'filtered_voltage': kalman_result['filtered'],
        'degradation_rate': kalman_result['velocity'],
        'uncertainty': kalman_result['covariance']
    })
    
    results_df.to_csv('battery_analytics_results.csv', index=False)
    print("   ✅ Saved: battery_analytics_results.csv")
    
    # Export summary statistics
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_samples': int(len(data)),
        'time_period_days': 90,
        'noise_reduction': {
            'original_variance': float(orig_variance),
            'filtered_variance': float(filt_variance),
            'reduction_percent': float(reduction)
        },
        'degradation': {
            'average_rate_per_sample': float(np.mean(kalman_result['velocity'])),
            'total_degradation': float(kalman_result['filtered'][0] - kalman_result['filtered'][-1]),
            'initial_voltage': float(kalman_result['filtered'][0]),
            'final_voltage': float(kalman_result['filtered'][-1])
        },
        'zscore_analysis': {
            'metric': zscore_result['metric'],
            'value': float(zscore_result['value']),
            'mean': float(zscore_result['mean']),
            'std_dev': float(zscore_result['std_dev']),
            'z_score': float(zscore_result['z_score']),
            'is_anomaly': bool(zscore_result['is_anomaly']),
            'severity': zscore_result['severity'],
            'confidence': float(zscore_result['confidence'])
        }
    }
    
    with open('battery_analytics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("   ✅ Saved: battery_analytics_summary.json")
    
    print("\n" + "="*80)
    print("  ✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. battery_kalman_analysis.png - Comprehensive Kalman filter visualization")
    print("  2. battery_noise_reduction.png - Noise reduction analysis")
    print("  3. battery_zscore_analysis.png - Statistical anomaly detection")
    print("  4. battery_analytics_results.csv - Complete dataset with filtered values")
    print("  5. battery_analytics_summary.json - Analysis summary statistics")
    print("\n")
    
    # Uncomment to show plots interactively
    # print("📊 Displaying plots (close windows to exit)...")
    # plt.show()


if __name__ == '__main__':
    main()
