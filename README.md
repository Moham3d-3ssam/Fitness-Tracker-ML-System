# 🏋️ Fitness Tracker ML System

A comprehensive machine learning system for tracking and analyzing fitness activities using wearable sensor data. This project applies advanced signal processing techniques including **Low Pass Filtering** and **Fourier Transformation** to extract meaningful features from accelerometer and gyroscope data, enabling accurate exercise classification and repetition counting.

## 🎯 Key Features

- **Advanced Signal Processing**
  - **Low Pass Butterworth Filter**: Removes high-frequency noise from sensor data using a configurable Butterworth filter implementation
  - **Fourier Transformation**: Extracts frequency domain features using Fast Fourier Transform (FFT) to identify periodic patterns in exercises
  - **Temporal Abstraction**: Aggregates time-series data using rolling windows with statistical functions (mean, max, min, median, std)

- **Feature Engineering**
  - Principal Component Analysis (PCA) for dimensionality reduction
  - Frequency domain feature extraction (dominant frequency, weighted frequency, power spectral entropy)
  - Time domain statistical features over rolling windows
  - Outlier detection and removal

- **Machine Learning**
  - Exercise classification using multiple ML algorithms
  - Automated repetition counting for exercises
  - Model evaluation and hyperparameter tuning
  - Feature selection and optimization

- **Visualization**
  - Comprehensive data exploration and visualization tools
  - Interactive plots for sensor data analysis
  - Model performance visualization

## 📁 Project Structure

```
Fitness-Tracker-ML-System/
│
├── data/                          # Data directory
│   ├── raw/                       # Raw sensor data (MetaMotion)
│   ├── interim/                   # Intermediate processed data
│   ├── processed/                 # Final processed datasets
│   └── external/                  # External data sources
│
├── src/                           # Source code
│   ├── data/                      # Data processing scripts
│   │   ├── make_dataset.py        # Data loading and preparation
│   │   └── make_dataset.ipynb     # Interactive data exploration
│   │
│   ├── features/                  # Feature engineering modules
│   │   ├── DataTransformation.py  # Low Pass Filter & PCA implementation
│   │   ├── FrequencyAbstraction.py # Fourier Transform feature extraction
│   │   ├── TemporalAbstraction.py  # Time-domain feature engineering
│   │   ├── build_features.py       # Feature pipeline
│   │   ├── remove_outliers.py      # Outlier detection and removal
│   │   ├── count_repetitions.py    # Exercise repetition counting
│   │   └── *.ipynb                 # Interactive notebooks for each module
│   │
│   ├── models/                    # Model training and prediction
│   │   ├── LearningAlgorithms.py  # ML algorithm implementations
│   │   ├── train_model.py         # Model training pipeline
│   │   ├── predict_model.py       # Prediction scripts
│   │   └── train_model.ipynb      # Interactive model training
│   │
│   └── visualization/             # Visualization tools
│       ├── visualize.py           # Plotting utilities
│       ├── plot_settings.py       # Plot configuration
│       └── visualize.ipynb        # Interactive visualizations
│
├── models/                        # Trained model artifacts
├── notebooks/                     # Jupyter notebooks for exploration
├── reports/                       # Generated reports and figures
├── references/                    # Reference materials and documentation
├── docs/                          # Additional documentation
│
└── README.md                      # This file
```

## 🔧 Technologies Used

- **Python 3.x**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **SciPy**: Scientific computing and signal processing
  - `scipy.signal.butter`: Butterworth filter design
  - `scipy.signal.filtfilt`: Zero-phase filtering
  - `scipy.signal.lfilter`: Standard filtering
- **NumPy FFT**: Fast Fourier Transform implementation
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive development and analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Moham3d-3ssam/Fitness-Tracker-ML-System.git
cd Fitness-Tracker-ML-System
```

2. Install required dependencies:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
```

3. Place your sensor data in the `data/raw/` directory

### Usage

1. **Data Preparation**:
   ```bash
   python src/data/make_dataset.py
   ```

2. **Feature Engineering**:
   ```bash
   python src/features/build_features.py
   ```

3. **Model Training**:
   ```bash
   python src/models/train_model.py
   ```

4. **Interactive Analysis**:
   Open any of the Jupyter notebooks in the `src/` directory:
   ```bash
   jupyter notebook src/features/build_features.ipynb
   ```

## 🛠 How It Works

### 1. Low Pass Filtering
The system uses a Butterworth low-pass filter to remove high-frequency noise from accelerometer and gyroscope data:
- Configurable cutoff frequency
- Adjustable filter order (default: 5)
- Zero-phase filtering option to prevent phase shift
- Nyquist frequency consideration for proper signal processing

### 2. Fourier Transformation
Fast Fourier Transform (FFT) is applied to extract frequency domain features:
- Identifies dominant frequencies in exercise patterns
- Computes frequency-weighted averages
- Calculates power spectral entropy
- Extracts amplitude information for different frequency components

### 3. Feature Engineering Pipeline
- **Temporal Features**: Rolling window statistics (mean, std, min, max)
- **Frequency Features**: FFT-based frequency domain characteristics
- **PCA**: Dimensionality reduction for correlated sensor axes
- **Outlier Removal**: Statistical outlier detection and handling

## 📖 Acknowledgments

Based on concepts from:
- **"Machine Learning for the Quantified Self"** by Mark Hoogendoorn and Burkhardt Funk (2017), Springer
- Updated and enhanced by Dave Ebbelaar (2022-2023)

## 📄 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This system is designed for research and educational purposes. Ensure proper data privacy and ethical considerations when using sensor data from individuals.
