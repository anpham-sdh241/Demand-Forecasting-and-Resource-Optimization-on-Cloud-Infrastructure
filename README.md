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