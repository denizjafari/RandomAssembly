# Time Series Forecasting with TimesFM on BCI Competition IV Dataset 4

This project demonstrates how to use Google's TimesFM foundation model for time series forecasting on the BCI Competition IV Dataset 4 (ECoG recordings for finger movements).

## Overview

- **Dataset**: [BCI Competition IV Dataset 4](https://braindecode.org/0.7/generated/braindecode.datasets.BCICompetitionIVDataset4.html)
  - ECoG (Electrocorticography) recordings from 3 patients
  - Targets correspond to finger flexion time courses (5 fingers)
  - Motor cortex activity during finger movements

- **Model**: [TimesFM (google/timesfm-1.0-200m)](https://huggingface.co/google/timesfm-1.0-200m)
  - Pretrained time-series foundation model from Google Research
  - Supports univariate time series forecasting
  - Context length up to 512 time points
  - Any horizon length

## Installation

### 0. Initialize Project (if starting fresh)

```bash
uv init
```

### 1. Install Base Dependencies

Install dependencies from `requirements.txt` using `uv`:

```bash
uv pip install -r requirements.txt
```

Or if you prefer to sync from `pyproject.toml`:

```bash
uv sync
```

### 2. Install Braindecode and MOABB

Braindecode requires MOABB (Mother of All BCI Benchmarks) for dataset downloads:

```bash
uv add braindecode moabb
```

Or install separately:

```bash
uv add braindecode
uv add moabb
```

### 3. Install TimesFM

**Important**: TimesFM requires installation from GitHub and does **not support ARM architectures** (Apple Silicon/M1/M2/M3 Macs) due to the `lingvo` dependency.

With `uv`, you have two options:

**Option A: Add to project dependencies (recommended)**
```bash
uv add "git+https://github.com/google-research/timesfm.git"
```

**Option B: Install directly (without adding to pyproject.toml)**
```bash
uv pip install git+https://github.com/google-research/timesfm.git
```

Note: If Option A doesn't work, you can also add it manually to `pyproject.toml` in the dependencies section.

**Note for Apple Silicon users**: 
- TimesFM will not work on native ARM architecture
- You may need to use x86_64 emulation (Rosetta 2) or a different environment
- The script will gracefully handle missing TimesFM and continue with data exploration

## Usage

Run the main script:

```bash
uv run python main.py
```

Or simply:

```bash
uv run main.py
```

The script will:
1. Download and load the BCI Competition IV Dataset 4
2. Explore dataset structure (channels, sampling frequency, duration, etc.)
3. Visualize ECoG signals (time series, power spectral density, channel variability)
4. Prepare univariate time series for forecasting (aggregate multiple channels)
5. Perform forecasting using TimesFM (if available)
6. Visualize forecast results
7. Generate a comprehensive summary report

## Output

The script generates:
- **ECoG Visualizations**: `output/eeg_visualization.png`
  - Sample channel signals
  - Power spectral density
  - Channel variability analysis

- **Forecast Visualization**: `output/forecast_visualization.png`
  - Original time series
  - Context window used
  - Forecasted values

- **Console Output**: Comprehensive analysis and statistics

## Dataset Information

The BCI Competition IV Dataset 4 contains:
- **3 subjects** (patients with ECoG implants)
- **ECoG recordings** during finger movement tasks
- **Target variables**: Time courses of flexion for each of 5 fingers
- **High-frequency neural signals** suitable for time series forecasting

## What You Can Do With This Dataset

1. **Time Series Forecasting**: Predict future ECoG signal patterns
2. **Finger Movement Decoding**: Predict which finger is being moved (classification)
3. **Signal Analysis**: Frequency domain analysis, time-frequency analysis
4. **Brain-Computer Interface Development**: Real-time movement prediction
5. **Transfer Learning**: Fine-tune TimesFM on specific subjects or tasks

## Configuration

You can modify the script parameters:

- `subject_ids`: Which subjects to load (default: 1, can be [1, 2, 3] or None for all)
- `context_len`: Length of context window for forecasting (default: 256, max: 512)
- `horizon_len`: Length of forecast horizon (default: 64)
- `freq`: Frequency indicator (0=high, 1=medium, 2=low; default: 0 for ECoG)
- `aggregation`: How to aggregate channels ('mean', 'median', 'std'; default: 'mean')

## Troubleshooting

### TimesFM Installation Issues

If TimesFM installation fails:
- Check if you're on ARM architecture (Apple Silicon)
- Try using x86_64 emulation
- The script will continue with data exploration even without TimesFM

### Dataset Download Issues

The dataset will be automatically downloaded on first run. If download fails:
- Check internet connection
- Verify MNE_DATA_PATH environment variable is set correctly
- Default download location: `~/mne_data`

### Memory Issues

For large datasets:
- Process subjects individually
- Reduce context_len or horizon_len
- Use data chunking for very long time series

## References

- **TimesFM Paper**: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) (ICML 2024)
- **BCI Competition IV**: [Dataset description](http://www.bbci.de/competition/iv/)
- **Braindecode**: [Documentation](https://braindecode.org/)
- **Dataset Citation**: Miller, K.J. "A library of human electrocorticographic data and analyses." Nature human behaviour 3, no. 11 (2019): 1225-1235

## License

- TimesFM: Apache 2.0
- BCI Competition IV Dataset: See dataset license

