# Code Explanation: main.py

## Overall Goal

This script performs **preliminary data exploration** and **time series forecasting** on the BCI Competition IV Dataset 4, which contains ECoG (Electrocorticography) recordings of brain activity during finger movements. The main objectives are:

1. **Load and explore** the neuroscience dataset structure
2. **Visualize** ECoG signals to understand the data
3. **Transform** multivariate ECoG data (multiple channels) into univariate time series
4. **Forecast** future neural signal patterns using Google's TimesFM foundation model
5. **Visualize** and report the forecasting results

This is useful for:
- Understanding how brain signals can predict future patterns
- Brain-computer interface research
- Time series analysis of neural data
- Demonstrating foundation model capabilities on neuroscience data

---

## Function-by-Function Breakdown

### Initial Setup (Lines 19-47)

**Purpose**: Import libraries and check for required dependencies

**What it does**:
- Imports NumPy, Pandas, Matplotlib, Seaborn for data manipulation and visualization
- Attempts to import `braindecode` (for loading the BCI dataset) and `timesfm` (for forecasting)
- Sets flags (`BRAINDECODE_AVAILABLE`, `TIMESFM_AVAILABLE`) to track which libraries are available
- Configures plotting style for better visualizations
- If libraries are missing, prints helpful installation instructions

**Why it's important**: Gracefully handles missing dependencies instead of crashing immediately

---

### `load_bci_dataset(subject_ids=None)` (Lines 50-82)

**Purpose**: Download and load the BCI Competition IV Dataset 4

**Parameters**:
- `subject_ids`: Which patient/subject to load (1, 2, or 3), or `None` for all

**What it does**:
1. Checks if `braindecode` is available (raises error if not)
2. Downloads the dataset if it's not already on your computer (first run only)
3. Loads the dataset for the specified subject(s)
4. Prints confirmation messages and basic info about the dataset

**Returns**: 
- `dataset`: A `BaseConcatDataset` object containing the ECoG recordings

**Example**:
```python
dataset = load_bci_dataset(subject_ids=1)  # Load only subject 1
```

---

### `explore_dataset_structure(dataset)` (Lines 85-135)

**Purpose**: Extract and print detailed information about the dataset structure

**Parameters**:
- `dataset`: The loaded BCI dataset

**What it does**:
1. Prints total number of recordings in the dataset
2. Examines the first recording to understand data structure:
   - Number of ECoG channels (electrodes)
   - Sampling frequency (how many measurements per second)
   - Duration of the recording
   - Channel names
3. Extracts the raw data array and time points
4. Calculates basic statistics (mean, std, min, max) of the signal values

**Returns**:
- `raw`: MNE Raw object (contains metadata and raw signal data)
- `data`: NumPy array of shape (channels × time_points) - the actual ECoG signals
- `times`: NumPy array of time points in seconds

**Key Information Extracted**:
- ECoG data is **multivariate** (multiple channels/electrodes recorded simultaneously)
- Each channel represents brain activity at a different location
- Data is sampled at high frequency (typically 1000 Hz for ECoG)

---

### `visualize_eeg_data(raw, data, times, save_path=None)` (Lines 138-208)

**Purpose**: Create three visualizations to understand the ECoG data

**Parameters**:
- `raw`: MNE Raw object with metadata
- `data`: ECoG signal array (channels × time points)
- `times`: Time array in seconds
- `save_path`: Optional path to save the figure

**What it does** (creates 3 subplots):

**Plot 1 - Time Series Signals**:
- Shows the first 10 seconds of up to 10 channels
- Each channel is normalized (scaled to similar amplitude) and vertically offset
- Visualizes how brain signals vary over time across different electrode locations

**Plot 2 - Power Spectral Density**:
- Analyzes the frequency content of one sample channel
- Shows which frequencies are most prominent in the signal
- Uses Welch's method (averaged periodogram) to estimate power spectrum
- Focuses on 0-100 Hz range (where most neural activity occurs)

**Plot 3 - Channel Variability**:
- Bar chart showing standard deviation across all channels
- Identifies which channels have the most variable signals
- Helps identify active vs. quiet brain regions

**Why it's useful**: These visualizations help you understand:
- Signal quality and patterns
- Which channels are most informative
- Frequency characteristics of the neural data

---

### `prepare_timeseries_for_timesfm(data, times, channel_idx=None, aggregation='mean')` (Lines 211-260)

**Purpose**: Convert multivariate ECoG data into univariate time series for TimesFM

**Problem**: TimesFM only works with **univariate** (single-channel) time series, but ECoG data has **multiple channels**. We need to reduce dimensionality.

**Parameters**:
- `data`: Multichannel ECoG array (channels × time points)
- `times`: Time array (not used but kept for consistency)
- `channel_idx`: If specified, use only this channel (e.g., `channel_idx=5` for channel 5)
- `aggregation`: If `channel_idx=None`, how to combine channels:
  - `'mean'`: Average across all channels (default)
  - `'median'`: Median across channels
  - `'std'`: Standard deviation across channels

**What it does**:
1. If `channel_idx` is specified: extracts that single channel
2. Otherwise: aggregates all channels using the specified method (default: mean)
3. Prints statistics about the resulting univariate time series

**Returns**:
- `timeseries`: 1D NumPy array representing the univariate time series

**Example**:
```python
# Use mean of all channels
timeseries = prepare_timeseries_for_timesfm(data, times, aggregation='mean')

# Or use a specific channel
timeseries = prepare_timeseries_for_timesfm(data, times, channel_idx=10)
```

**Why mean aggregation?**: Averaging channels can reduce noise and capture overall brain activity patterns, though it loses spatial information.

---

### `forecast_with_timesfm(timeseries, context_len=256, horizon_len=64, freq=0)` (Lines 263-352)

**Purpose**: Use Google's TimesFM foundation model to forecast future values of the time series

**Parameters**:
- `timeseries`: The univariate time series array to forecast
- `context_len`: How many past time points to use for prediction (default: 256, max: 512)
- `horizon_len`: How many future time points to predict (default: 64)
- `freq`: Frequency category indicator:
  - `0`: High frequency (for ECoG, daily/hourly data)
  - `1`: Medium frequency (weekly/monthly)
  - `2`: Low frequency (quarterly/yearly)

**What it does**:
1. Checks if TimesFM is available (returns `None` if not)
2. Detects if CUDA GPU is available (uses GPU if available, otherwise CPU)
3. Initializes TimesFM model with fixed architecture parameters:
   - `input_patch_len=32`: Model processes data in patches of 32
   - `output_patch_len=128`: Model outputs forecasts in patches of 128
   - `num_layers=20`: Transformer has 20 layers
   - `model_dims=1280`: Model dimension is 1280
4. Loads pretrained weights from HuggingFace (`google/timesfm-1.0-200m`)
5. Prepares input:
   - If time series is shorter than `context_len`, pads with zeros
   - Otherwise, uses the last `context_len` points as context
6. Performs forecasting using the model
7. Returns the forecasted values

**Returns**:
- `point_forecast`: Array of forecasted values (main output)
- `quantile_forecast`: Experimental quantile forecasts (uncertainty estimates, may be None)

**Error Handling**: Catches and reports common issues (ARM architecture, missing dependencies, network issues)

**Key Concept**: 
- **Context window**: The model looks at the last `context_len` points to understand patterns
- **Horizon**: The model predicts `horizon_len` points into the future
- This is similar to how language models predict next words, but for time series

---

### `visualize_forecast(timeseries, forecast, context_len, horizon_len, save_path=None)` (Lines 355-412)

**Purpose**: Create a visualization comparing the original time series with the forecast

**Parameters**:
- `timeseries`: Original full time series
- `forecast`: Forecasted values from TimesFM
- `context_len`: Context window length used
- `horizon_len`: Forecast horizon length
- `save_path`: Optional path to save the figure

**What it does**:
1. Plots the entire original time series
2. Highlights the context window (yellow shaded region) - the part used for prediction
3. Plots the forecast (red line) starting from the end of the original series
4. Adds a vertical dashed line marking where forecast begins
5. Adds labels, legend, and grid for clarity

**Why it's useful**: 
- Visual assessment of forecast quality
- See if the forecast continues the pattern reasonably
- Compare forecasted values with what actually happened (if you have ground truth)

**Note**: If `forecast` is `None` (TimesFM not available), this function does nothing.

---

### `generate_summary_report(dataset, raw, data, times, timeseries, forecast)` (Lines 415-498)

**Purpose**: Print a comprehensive summary of the entire analysis

**Parameters**: All the key objects from the analysis pipeline

**What it does**:
1. Organizes information into sections:
   - **Dataset**: Name, type, number of recordings, subjects
   - **Data Characteristics**: Channels, sampling rate, duration, shape
   - **Time Series for Forecasting**: Length, statistics of prepared series
   - **Forecasting**: Model used, status, forecast length

2. Prints suggestions for what you can do with this dataset:
   - Time series forecasting
   - Finger movement decoding (classification)
   - Signal analysis (frequency domain)
   - Brain-computer interface development
   - Transfer learning applications

**Why it's useful**: Provides a quick reference of what was analyzed and what's possible next

---

### `main()` (Lines 501-560)

**Purpose**: Orchestrates the entire analysis pipeline

**What it does** (step-by-step):

1. **Create output directory**: Makes `output/` folder for saving figures
2. **Load dataset**: Calls `load_bci_dataset()` for subject 1
3. **Explore structure**: Calls `explore_dataset_structure()` to understand the data
4. **Visualize ECoG**: Calls `visualize_eeg_data()` and saves to `output/eeg_visualization.png`
5. **Prepare time series**: Calls `prepare_timeseries_for_timesfm()` using mean aggregation
6. **Forecast**: Calls `forecast_with_timesfm()` with:
   - Context length: 256 points
   - Horizon length: 64 points
   - Frequency: 0 (high frequency for ECoG)
7. **Visualize forecast**: Calls `visualize_forecast()` if forecast succeeded, saves to `output/forecast_visualization.png`
8. **Generate report**: Calls `generate_summary_report()` to print final summary

**Error Handling**: Wraps everything in try-except to catch and display any errors with full traceback

**Output Files Created**:
- `output/eeg_visualization.png`: ECoG signal visualizations
- `output/forecast_visualization.png`: Forecast comparison plot

---

## Data Flow Summary

```
BCI Dataset (ECoG recordings)
    ↓
load_bci_dataset()
    ↓
explore_dataset_structure() → Extracts raw, data, times
    ↓
visualize_eeg_data() → Creates visualizations
    ↓
prepare_timeseries_for_timesfm() → Converts multichannel → univariate
    ↓
forecast_with_timesfm() → Generates forecast
    ↓
visualize_forecast() → Plots forecast vs original
    ↓
generate_summary_report() → Prints summary
```

---

## Key Concepts

### Multivariate vs Univariate Time Series
- **Multivariate**: Multiple signals recorded simultaneously (like ECoG with many channels)
- **Univariate**: Single signal over time (what TimesFM requires)
- **Solution**: Aggregate multiple channels (e.g., take mean) to create univariate series

### Context Window and Horizon
- **Context window**: Historical data the model uses to understand patterns
- **Horizon**: How far into the future to predict
- Example: With context_len=256 and horizon_len=64, model uses last 256 points to predict next 64 points

### ECoG Data
- **ECoG (Electrocorticography)**: Electrodes placed directly on brain surface (in surgery)
- **High temporal resolution**: Can capture fast neural dynamics
- **Multiple channels**: Each electrode records activity at different brain locations
- **Application**: Understanding motor cortex activity during finger movements

### TimesFM Model
- **Foundation model**: Pretrained on large amounts of time series data
- **Transformer architecture**: Uses attention mechanism (like GPT for language)
- **Transfer learning**: Can forecast new time series without retraining
- **Limitation**: Only works with univariate time series (single channel)

---

## Customization Options

You can modify the `main()` function to:

1. **Change subjects**: `load_bci_dataset(subject_ids=[1, 2, 3])` for all subjects
2. **Different aggregation**: `aggregation='median'` or `'std'` in `prepare_timeseries_for_timesfm()`
3. **Use specific channel**: `channel_idx=10` instead of aggregation
4. **Adjust forecast parameters**: Change `context_len` (up to 512) and `horizon_len`
5. **Different frequency**: Change `freq` parameter (0, 1, or 2)

---

## Common Issues and Solutions

1. **TimesFM not available**: Script continues with data exploration only
2. **ARM/Apple Silicon**: TimesFM doesn't work - use x86_64 emulation
3. **Dataset download fails**: Check internet connection and disk space
4. **Memory issues**: Reduce `context_len` or process fewer subjects
5. **Forecast quality**: Try different aggregation methods or specific channels

---

This script provides a complete pipeline from raw neuroscience data to time series forecasting, demonstrating how foundation models can be applied to neural signal analysis.

