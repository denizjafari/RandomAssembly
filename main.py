"""
Time Series Forecasting with TimesFM on BCI Competition IV Dataset 4

This script performs preliminary data exploration of the BCI Competition IV Dataset 4
(ECoG recordings for finger movements) and uses Google's TimesFM foundation model
for time series forecasting.

Dataset: BCICompetitionIVDataset4 - ECoG recordings with finger flexion targets
Model: google/timesfm-1.0-200m - Pretrained time-series foundation model

Note: TimesFM requires installation from GitHub:
    pip install git+https://github.com/google-research/timesfm.git
    
    Note: TimesFM does not support ARM architectures (Apple Silicon) due to
    lingvo dependency. If you're on Apple Silicon, you may need to use
    x86_64 emulation or a different environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try importing braindecode - if not available, provide installation instructions
try:
    from braindecode.datasets import BCICompetitionIVDataset4
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    print("Warning: braindecode not installed. Install with: uv add braindecode (or pip install braindecode)")

# Try importing timesfm - if not available, provide installation instructions
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    print("Warning: timesfm not installed. Install with: uv add \"git+https://github.com/google-research/timesfm.git\"")
    print("Or with pip: pip install git+https://github.com/google-research/timesfm.git")
    print("Note: TimesFM does not support ARM architectures (Apple Silicon)")

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_bci_dataset(subject_ids=None):
    """
    Load the BCI Competition IV Dataset 4.
    
    Parameters:
    -----------
    subject_ids : list of int or int or None
        Subject(s) to load. If None, loads all available subjects (1-3).
    
    Returns:
    --------
    dataset : BaseConcatDataset
        Loaded dataset containing ECoG recordings
    """
    print("=" * 80)
    print("Loading BCI Competition IV Dataset 4")
    print("=" * 80)
    
    if not BRAINDECODE_AVAILABLE:
        raise ImportError(
            "braindecode is required. Install with: uv add braindecode moabb\n"
            "Note: moabb is also required for dataset downloads."
        )
    
    # Download dataset if not already available
    print("\nDownloading dataset (if not already available)...")
    try:
        BCICompetitionIVDataset4.download()
    except ModuleNotFoundError as e:
        if 'moabb' in str(e):
            raise ImportError(
                "moabb is required for dataset downloads. Install with: uv add moabb\n"
                "Or: pip install moabb"
            ) from e
        raise
    
    # Load dataset
    print(f"\nLoading dataset for subjects: {subject_ids if subject_ids else 'all'}")
    dataset = BCICompetitionIVDataset4(subject_ids=subject_ids)
    
    print(f"Dataset loaded successfully!")
    print(f"Number of recordings: {len(dataset.datasets)}")
    
    return dataset


def explore_dataset_structure(dataset):
    """
    Explore the structure and properties of the BCI dataset.
    
    Parameters:
    -----------
    dataset : BaseConcatDataset
        The loaded BCI dataset
    """
    print("\n" + "=" * 80)
    print("DATASET STRUCTURE EXPLORATION")
    print("=" * 80)
    
    print(f"\nTotal number of recordings: {len(dataset.datasets)}")
    print(f"Dataset type: {type(dataset)}")
    
    # Explore first recording
    if len(dataset.datasets) > 0:
        first_recording = dataset.datasets[0]
        print(f"\nFirst recording type: {type(first_recording)}")
        print(f"First recording description: {first_recording.description}")
        
        # Get raw data
        raw = first_recording.raw
        print(f"\nRaw data info:")
        print(f"  - Number of channels: {len(raw.ch_names)}")
        print(f"  - Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"  - Duration: {raw.times[-1]:.2f} seconds")
        print(f"  - Number of time points: {len(raw.times)}")
        print(f"  - Channel names: {raw.ch_names[:10]}..." if len(raw.ch_names) > 10 else f"  - Channel names: {raw.ch_names}")
        
        # Check for target channels (finger flexion)
        print(f"\nTarget information:")
        if hasattr(first_recording, 'description'):
            print(f"  - Description: {first_recording.description}")
        
        # Get data shape
        data, times = raw[:, :]
        print(f"\nData shape: {data.shape} (channels x time points)")
        print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
        
        # Statistics
        print(f"\nData statistics:")
        print(f"  - Mean: {np.mean(data):.4f}")
        print(f"  - Std: {np.std(data):.4f}")
        print(f"  - Min: {np.min(data):.4f}")
        print(f"  - Max: {np.max(data):.4f}")
        
        return raw, data, times
    
    return None, None, None


def visualize_eeg_data(raw, data, times, save_path=None):
    """
    Visualize ECoG data from the dataset.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw MNE data object
    data : np.ndarray
        ECoG data array
    times : np.ndarray
        Time points array
    save_path : str or Path, optional
        Path to save the figure
    """
    print("\n" + "=" * 80)
    print("VISUALIZING ECoG DATA")
    print("=" * 80)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Sample of all channels (first 10 seconds)
    n_channels_to_plot = min(10, len(raw.ch_names))
    time_mask = times <= 10.0  # First 10 seconds
    
    ax1 = axes[0]
    for i in range(n_channels_to_plot):
        # Normalize for visualization
        channel_data = data[i, time_mask]
        channel_data_norm = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
        ax1.plot(times[time_mask], channel_data_norm + i * 2, label=raw.ch_names[i], alpha=0.7)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Channel (normalized amplitude)')
    ax1.set_title(f'ECoG Signals - First {n_channels_to_plot} Channels (First 10 seconds)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power spectral density for a sample channel
    ax2 = axes[1]
    sample_channel_idx = 0
    sample_channel_data = data[sample_channel_idx, :]
    
    # Compute power spectral density
    from scipy import signal
    freqs, psd = signal.welch(sample_channel_data, fs=raw.info['sfreq'], nperseg=1024)
    
    ax2.semilogy(freqs, psd)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title(f'Power Spectral Density - Channel: {raw.ch_names[sample_channel_idx]}')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)  # Focus on 0-100 Hz range
    
    # Plot 3: Amplitude distribution across channels
    ax3 = axes[2]
    channel_stds = [np.std(data[i, :]) for i in range(len(raw.ch_names))]
    
    ax3.bar(range(len(raw.ch_names)), channel_stds, alpha=0.7)
    ax3.set_xlabel('Channel Index')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Signal Variability Across Channels')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def prepare_timeseries_for_timesfm(data, times, channel_idx=None, aggregation='mean'):
    """
    Prepare ECoG data for TimesFM forecasting.
    
    TimesFM works with univariate time series, so we need to aggregate
    or select a single channel from the multivariate ECoG data.
    
    Parameters:
    -----------
    data : np.ndarray
        ECoG data array (channels x time points)
    times : np.ndarray
        Time points array
    channel_idx : int, optional
        Index of specific channel to use. If None, aggregates channels.
    aggregation : str
        Aggregation method if channel_idx is None: 'mean', 'median', or 'std'
    
    Returns:
    --------
    timeseries : np.ndarray
        Univariate time series array
    """
    print("\n" + "=" * 80)
    print("PREPARING TIME SERIES FOR TIMESFM")
    print("=" * 80)
    
    if channel_idx is not None:
        print(f"Using channel {channel_idx}")
        timeseries = data[channel_idx, :]
    else:
        print(f"Aggregating channels using method: {aggregation}")
        if aggregation == 'mean':
            timeseries = np.mean(data, axis=0)
        elif aggregation == 'median':
            timeseries = np.median(data, axis=0)
        elif aggregation == 'std':
            timeseries = np.std(data, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    print(f"Time series shape: {timeseries.shape}")
    print(f"Time series length: {len(timeseries)}")
    print(f"Time series statistics:")
    print(f"  - Mean: {np.mean(timeseries):.4f}")
    print(f"  - Std: {np.std(timeseries):.4f}")
    print(f"  - Min: {np.min(timeseries):.4f}")
    print(f"  - Max: {np.max(timeseries):.4f}")
    
    return timeseries


def forecast_with_timesfm(timeseries, context_len=256, horizon_len=64, freq=0):
    """
    Perform time series forecasting using TimesFM.
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Univariate time series array
    context_len : int
        Length of context window (max 512 for this model)
    horizon_len : int
        Length of forecast horizon
    freq : int
        Frequency indicator: 0 (high freq), 1 (medium freq), 2 (low freq)
    
    Returns:
    --------
    forecast : np.ndarray
        Forecasted values
    """
    print("\n" + "=" * 80)
    print("FORECASTING WITH TIMESFM")
    print("=" * 80)
    
    if not TIMESFM_AVAILABLE:
        print("\nSkipping TimesFM forecasting - library not available")
        print("Install with: uv add \"git+https://github.com/google-research/timesfm.git\"")
        print("Or with pip: pip install git+https://github.com/google-research/timesfm.git")
        return None
    
    try:
        # Initialize TimesFM model
        print(f"\nInitializing TimesFM model...")
        print(f"  - Context length: {context_len}")
        print(f"  - Horizon length: {horizon_len}")
        print(f"  - Frequency indicator: {freq}")
        
        # Note: backend can be 'cpu' or 'cuda' if available
        import torch
        backend = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  - Backend: {backend}")
        
        tfm = timesfm.TimesFm(
            context_len=context_len,
            horizon_len=horizon_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=backend,
        )
        
        print("\nLoading TimesFM checkpoint from HuggingFace...")
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
        print("Model loaded successfully!")
        
        # Prepare input - need to handle data length
        # TimesFM can handle variable context lengths, but we'll use the specified context_len
        if len(timeseries) < context_len:
            print(f"Warning: Time series length ({len(timeseries)}) is less than context_len ({context_len})")
            print("Padding with zeros...")
            padded_series = np.pad(timeseries, (0, context_len - len(timeseries)), mode='constant')
            forecast_input = [padded_series]
        else:
            # Use the last context_len points
            forecast_input = [timeseries[-context_len:]]
        
        frequency_input = [freq]
        
        print(f"\nPerforming forecast...")
        print(f"  - Input length: {len(forecast_input[0])}")
        
        point_forecast, experimental_quantile_forecast = tfm.forecast(
            forecast_input,
            freq=frequency_input,
        )
        
        print(f"Forecast completed!")
        print(f"  - Forecast shape: {point_forecast.shape}")
        print(f"  - Forecast length: {len(point_forecast[0])}")
        
        return point_forecast[0], experimental_quantile_forecast[0] if experimental_quantile_forecast is not None else None
        
    except Exception as e:
        print(f"\nError during TimesFM forecasting: {str(e)}")
        print("\nCommon issues:")
        print("  1. TimesFM doesn't support ARM architectures (Apple Silicon)")
        print("  2. Missing dependencies (lingvo, etc.)")
        print("  3. Network issues downloading model checkpoint")
        return None, None


def visualize_forecast(timeseries, forecast, context_len, horizon_len, save_path=None):
    """
    Visualize the original time series and the forecast.
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Original time series
    forecast : np.ndarray
        Forecasted values
    context_len : int
        Length of context window used
    horizon_len : int
        Length of forecast horizon
    save_path : str or Path, optional
        Path to save the figure
    """
    if forecast is None:
        print("\nSkipping forecast visualization - no forecast available")
        return
    
    print("\n" + "=" * 80)
    print("VISUALIZING FORECAST")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot original time series
    time_original = np.arange(len(timeseries))
    ax.plot(time_original, timeseries, label='Original Time Series', alpha=0.7, linewidth=1.5)
    
    # Plot context window used
    context_start = len(timeseries) - context_len
    context_end = len(timeseries)
    ax.axvspan(context_start, context_end, alpha=0.2, color='yellow', label='Context Window')
    
    # Plot forecast
    forecast_start = len(timeseries)
    forecast_end = forecast_start + len(forecast)
    time_forecast = np.arange(forecast_start, forecast_end)
    ax.plot(time_forecast, forecast, label='TimesFM Forecast', linewidth=2, color='red')
    
    # Add vertical line separating context and forecast
    ax.axvline(x=context_end, color='black', linestyle='--', linewidth=2, label='Forecast Start')
    
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Amplitude')
    ax.set_title('Time Series Forecasting with TimesFM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nForecast visualization saved to: {save_path}")
    
    plt.show()


def generate_summary_report(dataset, raw, data, times, timeseries, forecast):
    """
    Generate a summary report of the analysis.
    
    Parameters:
    -----------
    dataset : BaseConcatDataset
        The loaded dataset
    raw : mne.io.Raw
        Raw MNE data object
    data : np.ndarray
        ECoG data array
    times : np.ndarray
        Time points array
    timeseries : np.ndarray
        Prepared univariate time series
    forecast : np.ndarray or None
        Forecasted values
    """
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    report = {
        "Dataset": {
            "Name": "BCI Competition IV Dataset 4",
            "Type": "ECoG recordings for finger movements",
            "Number of recordings": len(dataset.datasets),
            "Subjects": "1-3 (patients with finger movement tasks)"
        },
        "Data Characteristics": {
            "Number of channels": len(raw.ch_names) if raw else "N/A",
            "Sampling frequency": f"{raw.info['sfreq']} Hz" if raw else "N/A",
            "Duration": f"{times[-1]:.2f} seconds" if times is not None else "N/A",
            "Time points": len(times) if times is not None else "N/A",
            "Data shape": data.shape if data is not None else "N/A"
        },
        "Time Series for Forecasting": {
            "Length": len(timeseries) if timeseries is not None else "N/A",
            "Mean": f"{np.mean(timeseries):.4f}" if timeseries is not None else "N/A",
            "Std": f"{np.std(timeseries):.4f}" if timeseries is not None else "N/A"
        },
        "Forecasting": {
            "Model": "TimesFM (google/timesfm-1.0-200m)",
            "Status": "Success" if forecast is not None else "Not available",
            "Forecast length": len(forecast) if forecast is not None else "N/A"
        }
    }
    
    for section, items in report.items():
        print(f"\n{section}:")
        print("-" * 40)
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("WHAT YOU CAN DO WITH THIS DATASET")
    print("=" * 80)
    print("""
    1. Time Series Forecasting:
       - Use TimesFM or other models to forecast finger flexion patterns
       - Predict future ECoG signal patterns
       - Analyze temporal dynamics of neural signals
    
    2. Finger Movement Decoding:
       - Predict which finger is being moved based on ECoG signals
       - Multi-class classification problem (5 fingers)
       - Regression to predict flexion amplitude for each finger
    
    3. Signal Analysis:
       - Frequency domain analysis (power spectral density)
       - Time-frequency analysis (wavelets, spectrograms)
       - Channel selection and feature engineering
    
    4. Brain-Computer Interface Development:
       - Real-time prediction of finger movements
       - Control of prosthetic devices
       - Understanding motor cortex encoding
    
    5. Transfer Learning:
       - Use TimesFM's pretrained representations
       - Fine-tune on specific subjects or tasks
       - Domain adaptation across subjects
    """)


def main():
    """Main function to run the complete analysis pipeline."""
    print("=" * 80)
    print("BCI COMPETITION IV DATASET 4 - TIMESFM ANALYSIS")
    print("=" * 80)
    
    # Create output directory for figures
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load dataset
        dataset = load_bci_dataset(subject_ids=1)  # Start with subject 1
        
        # Step 2: Explore dataset structure
        raw, data, times = explore_dataset_structure(dataset)
        
        if raw is None or data is None:
            print("Error: Could not load data from dataset")
            return
        
        # Step 3: Visualize ECoG data
        visualize_eeg_data(raw, data, times, save_path=output_dir / "eeg_visualization.png")
        
        # Step 4: Prepare time series for TimesFM
        # Use mean aggregation across channels to create univariate series
        timeseries = prepare_timeseries_for_timesfm(data, times, channel_idx=None, aggregation='mean')
        
        # Step 5: Forecast with TimesFM
        # Using context_len=256 and horizon_len=64 (adjust based on your needs)
        # freq=0 for high frequency data (ECoG is high frequency)
        forecast, quantile_forecast = forecast_with_timesfm(
            timeseries, 
            context_len=256, 
            horizon_len=64,
            freq=0  # High frequency (ECoG signals)
        )
        
        # Step 6: Visualize forecast
        if forecast is not None:
            visualize_forecast(
                timeseries, 
                forecast, 
                context_len=256, 
                horizon_len=64,
                save_path=output_dir / "forecast_visualization.png"
            )
        
        # Step 7: Generate summary report
        generate_summary_report(dataset, raw, data, times, timeseries, forecast)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nOutput files saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
