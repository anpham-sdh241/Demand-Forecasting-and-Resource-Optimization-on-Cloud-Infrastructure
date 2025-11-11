# Cloud Resource Demand Forecasting

## Introduction

This project performs data analysis and preparation for resource demand forecasting on cloud computing infrastructure. This is the first phase of the "Resource Demand Forecasting and Optimal Resource Allocation on Cloud Computing Infrastructure" project.

## Data Source

**Dataset**: The Westermo Test System Performance Data Set  
**File**: `data/system-1.csv`

### Field Descriptions

#### System Load
- `load-1m`: System load over the last 1 minute
- `load-5m`: System load over the last 5 minutes  
- `load-15m`: System load over the last 15 minutes

#### Memory Metrics
- `sys-mem-swap-total`: Swap size (constant)
- `sys-mem-swap-free`: Available swap
- `sys-mem-free`: Unused memory
- `sys-mem-cache`: Memory used for cache
- `sys-mem-buffered`: Memory used by kernel buffers
- `sys-mem-available`: Memory available to be allocated (free and cache)
- `sys-mem-total`: Total size of memory (constant)

#### CPU Metrics
- `cpu-iowait`: Summerized rate of change of seconds spent on waiting for I/O
- `cpu-system`: Summerized rate of change of seconds spent on kernel space threads
- `cpu-user`: Summerized rate of change of seconds spent on user space processes/threads

#### Disk I/O Metrics
- `disk-io-time`: Rate of change in time spent on storage i/o operations
- `disk-bytes-read`: Rate of change in bytes read
- `disk-bytes-written`: Rate of change in bytes written
- `disk-io-read`: Rate of change in amount read operations
- `disk-io-write`: Rate of change in amount write operations

#### System Metrics
- `sys-fork-rate`: Rate of change in number of forks
- `sys-interrupt-rate`: Rate of change of interrupts
- `sys-context-switch-rate`: Rate of change of context switches
- `sys-thermal`: Average rate of change in measured system temperature (Celsius)
- `server-up`: Freshness check of reporting server (values above 0 indicate server is available)

## Target Variables

Based on cloud resource forecasting requirements, the selected target variables include:

### 1. Memory Usage
- **Target**: `memory_usage_pct` (Memory usage percentage)
- **Formula**: `(sys-mem-total - sys-mem-available) / sys-mem-total * 100`

### 2. CPU Usage
- **Target**: `cpu_total_usage` (Total CPU usage)
- **Formula**: `cpu-user + cpu-system + cpu-iowait`

### 3. System Load
- **Target**: `load-1m` (System load 1-minute)
- Important indicator for assessing current system workload

## ETL Pipeline

### 1. Extract
- Load data from CSV file
- Explore data structure and statistics
- Convert timestamps to datetime format

### 2. Transform
- Create target variables
- Check missing values
- Analyze distributions
- Calculate full correlation matrix
- Identify high-correlation features for each target
- Data cleaning (remove constant columns, handle outliers)
- Normalize features using StandardScaler

### 3. Load
- **Feature selection**: Use only high correlation features (|r| > 0.3)
- Split data into train (80%) and test (20%) sets
- Sequential split to preserve time order
- **Data leakage prevention**: Exclude component variables from features
  - `memory_usage_pct`: excludes `sys-mem-total`, `sys-mem-available`
  - `cpu_total_usage`: excludes `cpu-user`, `cpu-system`, `cpu-iowait`
  - `system_load`: excludes `load-1m`
- Export processed datasets for each target

## Results

**Data Shape**: 85,749 rows × 24 columns  
**Time Span**: 30 days
**Target Variables**: 3 (Memory Usage %, CPU Total Usage, System Load)  
**Feature Selection**: High correlation only (|r| > 0.3)
- `memory_usage_pct`: 9 features
- `cpu_total_usage`: 10 features  
- `system_load`: 3 features

### High Correlation Features (|r| > 0.3)

**Memory Usage**:
- load-15m (0.54)
- sys-context-switch-rate (0.44)
- cpu_total_usage (0.44)
- cpu-system (0.43)

**CPU Usage**:
- cpu-user (0.97)
- cpu-system (0.75)
- sys-context-switch-rate (0.64)
- sys-fork-rate (0.50)

**System Load**:
- load-1m (1.00)
- load-5m (0.76)
- load-15m (0.49)

## Files

**Data & Processing:**
- `data/system-1.csv`: Raw data
- `etl_cloud_resource_forecasting.ipynb`: Main ETL pipeline
- `processed_data/`: Processed datasets (created after running ETL)
  - `cleaned_data.csv`: Cleaned dataset
  - `normalized_data.csv`: Normalized dataset
  - `feature_metadata.json`: Feature information for each target
  - `[target]/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: Train/test splits

**Modeling:**
- `model_arimax.ipynb`: ARIMAX (statistical) model training and evaluation
- `model_svr.ipynb`: SVR (Support Vector Regression) model training and evaluation
- `model_random_forest.ipynb`: Random Forest model training and evaluation
- `model_utils.py`: Reusable utility functions (save/load models, metrics, forecasting)
- `models/`: Saved trained models (created after training)
- `results_arimax.json`: ARIMAX evaluation results
- `results_svr.json`: SVR evaluation results
- `results_random_forest.json`: Random Forest evaluation results

**Utilities:**
- `model_utils.py`: Reusable utility functions
  - `save_model()`, `load_model()`: Model persistence
  - `forecast_with_horizon()`: Configurable forecast horizon
  - `calculate_metrics()`, `print_metrics()`: Evaluation
  - `save_results()`, `compare_results()`: Results management


