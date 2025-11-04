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
from pathlib import Path
import os
import platform
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless servers (no display)
# Must be done BEFORE importing pyplot
import matplotlib
if os.environ.get('DISPLAY') is None and platform.system() != 'Windows':
    # Use non-interactive backend for headless servers
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

# Try importing braindecode - if not available, provide installation instructions
try:
    from braindecode.datasets import BCICompetitionIVDataset4
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    print("Warning: braindecode not installed. Install with: uv add braindecode (or pip install braindecode)")

# Detect platform architecture
def detect_platform():
    """Detect the platform and architecture."""
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor()
    
    is_apple_silicon = (
        system == 'Darwin' and 
        (machine == 'arm64' or 'arm' in processor.lower() or 'Apple' in processor)
    )
    is_linux = system == 'Linux'
    is_windows = system == 'Windows'
    
    return {
        'system': system,
        'machine': machine,
        'processor': processor,
        'is_apple_silicon': is_apple_silicon,
        'is_linux': is_linux,
        'is_windows': is_windows,
        'is_arm': is_apple_silicon or 'arm' in machine.lower()
    }

PLATFORM_INFO = detect_platform()

# Try importing timesfm - if not available, provide installation instructions
TIMESFM_AVAILABLE = False
TIMESFM_COMPATIBLE = False

try:
    import timesfm
    TIMESFM_AVAILABLE = True
    # Check if TimesFM actually works (might fail on ARM even if installed)
    if not PLATFORM_INFO['is_arm']:
        TIMESFM_COMPATIBLE = True
    else:
        print("Warning: Detected ARM architecture (Apple Silicon). TimesFM may not work.")
        print("Attempting to use TimesFM anyway, but it may fail...")
        # Try to import lingvo to see if it works
        try:
            import lingvo
            TIMESFM_COMPATIBLE = True
        except ImportError:
            TIMESFM_COMPATIBLE = False
            print("Note: TimesFM's lingvo dependency doesn't support ARM. Using fallback methods.")
except ImportError:
    TIMESFM_AVAILABLE = False
    if PLATFORM_INFO['is_arm']:
        print("Info: TimesFM not installed (ARM architecture detected - not supported).")
        print("Will use alternative forecasting methods that work on Apple Silicon.")
    else:
        print("Warning: timesfm not installed. Install with: uv add \"git+https://github.com/google-research/timesfm.git\"")
        print("Or with pip: pip install git+https://github.com/google-research/timesfm.git")

# Check if running in headless mode
if matplotlib.get_backend() == 'Agg':
    print("Running in headless mode (no display available) - plots will be saved to files only")

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def get_dataset_path():
    """
    Get the path where the BCI dataset is stored.
    
    Returns:
    --------
    path : Path
        Path to the dataset directory
    """
    import os
    from pathlib import Path
    
    # Check MNE_DATA_PATH environment variable first
    mne_data_path = os.environ.get('MNE_DATA_PATH')
    if mne_data_path:
        base_path = Path(mne_data_path)
    else:
        # Default location: ~/mne_data
        base_path = Path.home() / 'mne_data'
    
    # MOABB typically stores datasets in a subdirectory
    # BCI Competition IV dataset is usually in moabb/BCICIV4
    dataset_path = base_path / 'moabb' / 'BCICIV4'
    
    return base_path, dataset_path


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
    
    # Show where data will be stored
    base_path, dataset_path = get_dataset_path()
    print(f"\nDataset storage information:")
    print(f"  - Base data path: {base_path}")
    print(f"  - Dataset path: {dataset_path}")
    print(f"  - MNE_DATA_PATH env var: {os.environ.get('MNE_DATA_PATH', 'Not set (using default)')}")
    
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
    
    # Check if dataset directory exists after download
    if dataset_path.exists():
        print(f"\nDataset found at: {dataset_path}")
        print(f"  - Directory exists: Yes")
        # List contents if available
        try:
            contents = list(dataset_path.iterdir())
            if contents:
                print(f"  - Number of items: {len(contents)}")
        except:
            pass
    
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
    
    # Only show plot if display is available
    try:
        plt.show()
    except Exception:
        print("Note: Display not available - figure saved to file only")
    finally:
        plt.close()


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


def forecast_with_alternative_methods(timeseries, context_len=256, horizon_len=64):
    """
    Perform time series forecasting using alternative methods that work on all platforms.
    
    Uses statistical and machine learning methods that are compatible with Apple Silicon.
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Univariate time series array
    context_len : int
        Length of context window to use
    horizon_len : int
        Length of forecast horizon
    
    Returns:
    --------
    forecast : np.ndarray
        Forecasted values
    """
    print("\n" + "=" * 80)
    print("FORECASTING WITH ALTERNATIVE METHODS (Platform-Compatible)")
    print("=" * 80)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Prepare input
    if len(timeseries) < context_len:
        print(f"Warning: Time series length ({len(timeseries)}) is less than context_len ({context_len})")
        print("Using all available data as context...")
        context = timeseries
    else:
        context = timeseries[-context_len:]
    
    print(f"\nUsing context length: {len(context)}")
    print(f"Forecasting {horizon_len} steps ahead")
    
    # Method 1: Simple Linear Regression with lag features
    print("\nMethod: Linear Regression with lag features")
    
    # Create lag features
    n_lags = min(10, len(context) // 2)  # Use up to 10 lags
    X = []
    y = []
    
    for i in range(n_lags, len(context)):
        X.append(context[i-n_lags:i])
        y.append(context[i])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) > 0:
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        forecast = []
        last_window = context[-n_lags:].copy()
        
        for _ in range(horizon_len):
            next_pred = model.predict(last_window.reshape(1, -1))[0]
            forecast.append(next_pred)
            # Update window: shift and add prediction
            last_window = np.roll(last_window, -1)
            last_window[-1] = next_pred
        
        forecast = np.array(forecast)
        
        print(f"Forecast completed using Linear Regression")
        print(f"  - Forecast shape: {forecast.shape}")
        print(f"  - Forecast mean: {np.mean(forecast):.4f}")
        print(f"  - Forecast std: {np.std(forecast):.4f}")
        
        return forecast
    else:
        print("Error: Not enough data for forecasting")
        return None


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
        Forecasted values, or None if TimesFM is not available/compatible
    """
    print("\n" + "=" * 80)
    print("FORECASTING WITH TIMESFM")
    print("=" * 80)
    
    if not TIMESFM_AVAILABLE or not TIMESFM_COMPATIBLE:
        if PLATFORM_INFO['is_arm']:
            print("\nTimesFM not compatible with ARM architecture (Apple Silicon)")
            print("Will use alternative forecasting methods instead.")
        else:
            print("\nTimesFM not available or not compatible")
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
        
        forecast_values = point_forecast[0]
        quantile_values = experimental_quantile_forecast[0] if experimental_quantile_forecast is not None else None
        
        return forecast_values, quantile_values
        
    except Exception as e:
        print(f"\nError during TimesFM forecasting: {str(e)}")
        print("\nCommon issues:")
        print("  1. TimesFM doesn't support ARM architectures (Apple Silicon)")
        print("  2. Missing dependencies (lingvo, etc.)")
        print("  3. Network issues downloading model checkpoint")
        return None, None


def visualize_forecast(timeseries, forecast, context_len, horizon_len, forecast_method="TimesFM", save_path=None):
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
    forecast_method : str
        Name of the forecasting method used
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
    context_start = max(0, len(timeseries) - context_len)
    context_end = len(timeseries)
    ax.axvspan(context_start, context_end, alpha=0.2, color='yellow', label='Context Window')
    
    # Plot forecast
    forecast_start = len(timeseries)
    forecast_end = forecast_start + len(forecast)
    time_forecast = np.arange(forecast_start, forecast_end)
    ax.plot(time_forecast, forecast, label=f'{forecast_method} Forecast', linewidth=2, color='red')
    
    # Add vertical line separating context and forecast
    ax.axvline(x=context_end, color='black', linestyle='--', linewidth=2, label='Forecast Start')
    
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Time Series Forecasting with {forecast_method}')
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
            "Platform": f"{PLATFORM_INFO['system']} ({PLATFORM_INFO['machine']})",
            "TimesFM Compatible": f"{TIMESFM_COMPATIBLE}",
            "Model": "TimesFM (google/timesfm-1.0-200m)" if TIMESFM_COMPATIBLE else "Alternative (Linear Regression)",
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
    
    # Display platform information
    print("\n" + "=" * 80)
    print("PLATFORM INFORMATION")
    print("=" * 80)
    print(f"  - System: {PLATFORM_INFO['system']}")
    print(f"  - Machine: {PLATFORM_INFO['machine']}")
    print(f"  - Processor: {PLATFORM_INFO['processor']}")
    print(f"  - Apple Silicon: {PLATFORM_INFO['is_apple_silicon']}")
    print(f"  - ARM Architecture: {PLATFORM_INFO['is_arm']}")
    print(f"  - TimesFM Available: {TIMESFM_AVAILABLE}")
    print(f"  - TimesFM Compatible: {TIMESFM_COMPATIBLE}")
    
    if PLATFORM_INFO['is_arm'] and not TIMESFM_COMPATIBLE:
        print("\n  Note: Running on Apple Silicon - will use alternative forecasting methods")
        print("  that work on all platforms (Linear Regression with lag features).")
    
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
        
        # Step 5: Forecast with TimesFM or alternative methods
        # Using context_len=256 and horizon_len=64 (adjust based on your needs)
        forecast = None
        forecast_method = "None"
        
        # Try TimesFM first (if compatible)
        if TIMESFM_COMPATIBLE:
            try:
                forecast_result = forecast_with_timesfm(
                    timeseries, 
                    context_len=256, 
                    horizon_len=64,
                    freq=0  # High frequency (ECoG signals)
                )
                if forecast_result is not None:
                    if isinstance(forecast_result, tuple):
                        forecast, quantile_forecast = forecast_result
                    else:
                        forecast = forecast_result
                    if forecast is not None:
                        forecast_method = "TimesFM"
            except Exception as e:
                print(f"\nTimesFM failed: {e}")
                print("Falling back to alternative methods...")
                forecast = None
        
        # Use alternative methods if TimesFM not available or failed
        if forecast is None:
            forecast = forecast_with_alternative_methods(
                timeseries,
                context_len=256,
                horizon_len=64
            )
            if forecast is not None:
                forecast_method = "Linear Regression (Alternative)"
        
        # Step 6: Visualize forecast
        if forecast is not None:
            visualize_forecast(
                timeseries, 
                forecast, 
                context_len=256, 
                horizon_len=64,
                forecast_method=forecast_method,
                save_path=output_dir / "forecast_visualization.png"
            )
        
        # Step 7: Generate summary report
        generate_summary_report(dataset, raw, data, times, timeseries, forecast)
        
        # Display final platform note
        if PLATFORM_INFO['is_arm'] and not TIMESFM_COMPATIBLE:
            print("\n" + "=" * 80)
            print("PLATFORM COMPATIBILITY NOTE")
            print("=" * 80)
            print("Running on Apple Silicon - used alternative forecasting methods.")
            print("For TimesFM support, consider:")
            print("  1. Using x86_64 emulation (Rosetta 2)")
            print("  2. Running on a Linux server with x86_64 architecture")
            print("  3. Using Docker with x86_64 base image")
        
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
