# Cấu Trúc Báo Cáo - Dự Báo Nhu Cầu và Tối Ưu Hóa Tài Nguyên trên Cloud Infrastructure

## Context Dự Án

Dự án này nghiên cứu và triển khai hệ thống dự báo nhu cầu tài nguyên và tối ưu hóa phân bổ VM (Virtual Machine) trên cloud computing infrastructure. Hệ thống sử dụng các mô hình dự báo thời gian (ARIMAX, SVR, Random Forest, Hybrid Prophet-LSTM) để dự đoán nhu cầu CPU và Memory, sau đó áp dụng các thuật toán tối ưu (Linear Programming, Deep Reinforcement Learning với PPO) để phân bổ VM một cách hiệu quả.

### Dữ Liệu
- **Nguồn**: Westermo Test System Performance Data Set
- **File**: `data/system-1.csv`
- **Kích thước**: 85,749 dòng × 24 cột
- **Thời gian**: 30 ngày
- **Biến mục tiêu**: Memory Usage %, CPU Total Usage, System Load

### Các Thành Phần Chính
1. **ETL Pipeline**: Tiền xử lý dữ liệu, feature engineering, train/test split
2. **Forecasting Models**: ARIMAX, SVR, Random Forest, Hybrid Prophet-LSTM
3. **Optimization Algorithms**: Linear Programming (baseline), Deep Reinforcement Learning (PPO)
4. **Evaluation**: So sánh hiệu quả giữa các phương pháp

---

## Cấu Trúc Báo Cáo Chi Tiết

### Chương 1. Giới Thiệu

#### 1.1 Hiện Trạng và Vấn Đề
- **Vấn đề**: Cloud infrastructure cần phân bổ tài nguyên động để đáp ứng workload thay đổi
- **Thách thức**: 
  - Dự báo chính xác nhu cầu tài nguyên trong tương lai
  - Tối ưu hóa chi phí vận hành VM
  - Đảm bảo SLA (Service Level Agreement)
  - Giảm thiểu chi phí switching (khởi động/dừng VM)
- **Tác động**: Chi phí cloud cao, lãng phí tài nguyên, vi phạm SLA

#### 1.2 Mục Tiêu và Phạm Vi Nghiên Cứu
- **Mục tiêu chính**:
  - Xây dựng hệ thống dự báo nhu cầu tài nguyên chính xác
  - Phát triển thuật toán tối ưu phân bổ VM hiệu quả
  - So sánh hiệu quả giữa các phương pháp (LP vs DRL)
- **Phạm vi**:
  - Dữ liệu hệ thống thực tế (Westermo dataset)
  - Các mô hình dự báo thời gian phổ biến
  - Hai phương pháp tối ưu: LP (deterministic) và DRL (learning-based)

#### 1.3 Câu Hỏi Nghiên Cứu
1. Mô hình dự báo nào cho độ chính xác cao nhất cho nhu cầu CPU/Memory?
2. Thuật toán tối ưu nào (LP hay DRL) cho hiệu quả tốt hơn về chi phí và SLA?
3. Làm thế nào để cân bằng giữa chi phí vận hành và chất lượng dịch vụ?
4. Switching cost ảnh hưởng như thế nào đến quyết định phân bổ VM?

#### 1.4 Các Nghiên Cứu Liên Quan
- **Time Series Forecasting**: ARIMA, LSTM, Prophet trong cloud resource prediction
- **Resource Allocation**: Linear Programming, Integer Programming cho VM scheduling
- **Reinforcement Learning**: PPO, DQN cho dynamic resource allocation
- **Cloud Optimization**: Auto-scaling, predictive scaling strategies

#### 1.5 Các Giai Đoạn Thực Hiện
1. **Giai đoạn 1**: Thu thập và tiền xử lý dữ liệu
2. **Giai đoạn 2**: Xây dựng và đánh giá các mô hình dự báo
3. **Giai đoạn 3**: Thiết kế và triển khai thuật toán tối ưu (LP)
4. **Giai đoạn 4**: Phát triển và huấn luyện DRL agent (PPO)
5. **Giai đoạn 5**: So sánh và đánh giá kết quả

#### 1.6 Tổng Quan Báo Cáo
- Chương 2: Kiến thức nền tảng (SLA, forecasting methods, optimization algorithms)
- Chương 3: Thiết kế và thực hiện (kiến trúc, pipeline, experiments)
- Chương 4: Kết luận và hướng phát triển

---

### Chương 2. Kiến Thức Nền Tảng

#### 2.1 Chuẩn Chất Lượng Dịch Vụ và Các Chỉ Số Đánh Giá
- **SLA (Service Level Agreement)**: Thỏa thuận về chất lượng dịch vụ
- **Các chỉ số**:
  - **Availability**: Tỷ lệ thời gian hệ thống hoạt động
  - **Response Time**: Thời gian phản hồi
  - **Resource Utilization**: Mức độ sử dụng tài nguyên (CPU, Memory)
  - **SLA Violation**: Vi phạm khi allocated < required resources

#### 2.2 Các Phương Pháp Dự Báo Tải Theo Thời Gian

##### 1. ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)
- **Nguyên lý**: Mở rộng ARIMA với biến ngoại sinh
- **Ưu điểm**: Xử lý được trend, seasonality, và external factors
- **Nhược điểm**: Giả định tuyến tính, khó xử lý non-linear patterns
- **Ứng dụng**: Dự báo nhu cầu tài nguyên với các features hệ thống

##### 2. Support Vector Regression (SVR)
- **Nguyên lý**: Sử dụng kernel trick để mô hình hóa quan hệ phi tuyến
- **Ưu điểm**: Xử lý được non-linear patterns, robust với outliers
- **Nhược điểm**: Tốn thời gian training với dataset lớn
- **Ứng dụng**: Dự báo với patterns phức tạp

##### 3. Random Forest
- **Nguyên lý**: Ensemble của nhiều decision trees
- **Ưu điểm**: Xử lý non-linear, feature importance, robust
- **Nhược điểm**: Khó interpret, có thể overfit
- **Ứng dụng**: Dự báo với nhiều features tương tác

##### 4. Hybrid Prophet + LSTM
- **Nguyên lý**: Kết hợp Prophet (xử lý seasonality) và LSTM (xử lý patterns phức tạp)
- **Ưu điểm**: Tận dụng ưu điểm của cả hai mô hình
- **Nhược điểm**: Phức tạp hơn, cần tuning nhiều hyperparameters
- **Ứng dụng**: Dự báo với seasonality và long-term dependencies

#### 2.3 Các Thuật Toán Cho Bài Toán Sắp Xếp Tối Ưu

**File tham khảo**: `docs/LINEAR_PROGRAMMING_THEORY.md`

##### 2.3.1 Linear Programming (Quy Hoạch Tuyến Tính)
- **Định nghĩa**: Tối ưu hóa hàm mục tiêu tuyến tính với ràng buộc tuyến tính
- **Dạng toán học**: 
  - Minimize/Maximize: `c'x`
  - Subject to: `Ax {≤, =, ≥} b`, `x ≥ 0`
- **Phương pháp giải**:
  - **Simplex Method**: Di chuyển dọc theo biên của feasible region
  - **Interior Point Method**: Di chuyển qua bên trong feasible region
- **Ứng dụng**: Phân bổ VM với mục tiêu minimize cost, ràng buộc CPU/Memory
- **Ưu điểm**: Đảm bảo tối ưu toàn cục, giải nhanh
- **Nhược điểm**: Giả định tuyến tính, nghiệm có thể không nguyên

##### 2.3.2 Deep Reinforcement Learning (PPO)
- **Nguyên lý**: Agent học policy tối ưu thông qua tương tác với environment
- **PPO (Proximal Policy Optimization)**: 
  - On-policy algorithm, stable training
  - Clipped objective function để tránh policy update quá lớn
- **Ứng dụng**: Tối ưu phân bổ VM với switching cost và multi-objective
- **Ưu điểm**: Xử lý được non-linear, học được patterns phức tạp
- **Nhược điểm**: Cần nhiều dữ liệu training, khó interpret

#### 2.4 Quy Tắc Chuyển Đổi Dựa Trên Kết Quả Dự Báo
- **Chuyển đổi forecast → requirements**:
  - So sánh forecast với host capacity thresholds
  - Tính toán `cpu_overflow_cores` và `memory_overflow_gb`
  - Nếu overflow > 0 → cần allocate VM
- **Mapping requirements → VM allocation**:
  - Sử dụng LP hoặc DRL để quyết định số lượng từng loại VM
  - Xét switching cost từ allocation trước đó

#### 2.5 Các Tiêu Chí Đánh Giá Hiệu Quả

##### 1. Các Chỉ Số Đánh Giá Độ Chính Xác Mô Hình Dự Báo
- **MAE (Mean Absolute Error)**: Trung bình sai số tuyệt đối
- **RMSE (Root Mean Squared Error)**: Căn bậc hai của trung bình bình phương sai số
- **MAPE (Mean Absolute Percentage Error)**: Trung bình phần trăm sai số
- **R² (Coefficient of Determination)**: Độ phù hợp của mô hình

##### 2. Các Tiêu Chí Đánh Giá Hiệu Quả Phân Bổ Tài Nguyên
- **Total VM Cost**: Tổng chi phí vận hành VM
- **Switching Cost**: Chi phí khởi động/dừng VM
- **SLA Violations**: Số lần vi phạm SLA (allocated < required)
- **Resource Utilization**: Mức độ sử dụng CPU/Memory (%)
- **Total VMs Used**: Tổng số VM được sử dụng
- **Cost Efficiency**: Tỷ lệ chi phí trên đơn vị capacity

---

### Chương 3. Thiết Kế và Thực Hiện Đề Tài

#### 3.1 Kiến Trúc Tổng Quan
- **Luồng xử lý**:
  1. Data Collection → ETL Pipeline
  2. Feature Engineering → Forecasting Models
  3. Forecast → Resource Requirements
  4. Requirements → Optimization (LP/DRL)
  5. Allocation → Evaluation
- **Components**:
  - Data processing module
  - Forecasting module (multiple models)
  - Optimization module (LP baseline, DRL agent)
  - Evaluation and comparison module

#### 3.2 Dữ Liệu, Tiền Xử Lý và Huấn Luyện Mô Hình Dự Báo
- **ETL Pipeline** (`etl_cloud_resource_forecasting.ipynb`):
  - Extract: Load CSV, convert timestamps
  - Transform: Create target variables, feature selection (|r| > 0.3), normalization
  - Load: Train/test split (80/20), sequential split
- **Feature Selection**:
  - Memory Usage: 9 features (load-15m, sys-context-switch-rate, ...)
  - CPU Usage: 10 features (cpu-user, cpu-system, ...)
  - System Load: 3 features (load-1m, load-5m, load-15m)
- **Model Training**:
  - ARIMAX: Statistical model với exogenous variables
  - SVR: Kernel-based regression
  - Random Forest: Ensemble method
  - Hybrid Prophet-LSTM: Deep learning hybrid

#### 3.3 Chuyển Đổi Dự Báo Thành Nhu Cầu Máy Ảo Theo Thời Gian
- **Forecast → Requirements**:
  - So sánh forecast với host capacity thresholds
  - Tính `cpu_overflow_cores = max(0, forecast_cpu - cpu_threshold)`
  - Tính `memory_overflow_gb = max(0, forecast_memory - memory_threshold)`
- **Requirements → VM Allocation**:
  - Input: CPU/Memory requirements, VM catalog
  - Output: Số lượng từng loại VM cần allocate

#### 3.4 Thiết Kế và Hiện Thực Thuật Toán Tối Ưu

##### 3.4.1 Linear Programming Implementation
- **File**: `vm_resource_planner.py`
- **Function**: `solve_vm_allocation()`
- **Objectives**:
  - `"cost"`: Minimize total VM cost
  - `"capacity"`: Minimize cost per capacity unit
- **Constraints**: CPU và Memory requirements
- **Solver**: scipy.optimize.linprog với method "highs"

##### 3.4.2 Deep Reinforcement Learning Implementation
- **Files**: `rl/` directory, `train_ppo.py`, `eval_ppo.py`
- **Environment**: `VMAllocationEnv` (Gymnasium)
- **Agent**: PPO (Stable-Baselines3)
- **Reward Function**:
  - SLA compliance (α)
  - Overflow penalty (β)
  - Switching cost (γ)
  - VM cost (δ)
- **Scenarios**:
  - `"overload"`: Prioritize SLA (α=1.0, β=0.7)
  - `"cost"`: Prioritize cost (δ=1.0, α=0.3)

#### 3.5 Kịch Bản Thực Nghiệm và Thiết Lập Tham Số
- **Dataset**: Test set (20% của data, ~17,150 timesteps)
- **VM Catalog**: 3 loại VM với specs khác nhau
- **Scenarios**:
  1. **LP Baseline**: Reactive mode, solve per timestep
  2. **DRL Overload**: Minimize resource overload
  3. **DRL Cost**: Optimize operational cost
- **Metrics**: Total cost, switching cost, SLA violations, utilization

#### 3.6 Đánh Giá Kết Quả Thực Nghiệm
- **Forecasting Performance**:
  - So sánh MAE, RMSE, MAPE, R² giữa các models
  - Best model cho từng target variable
- **Optimization Performance**:
  - So sánh LP vs DRL trên các metrics
  - Trade-off giữa cost và SLA compliance
  - Analysis của switching cost impact
- **Visualizations**:
  - Forecast vs actual plots
  - Allocation schedules
  - Cost and utilization trends

---

### Chương 4. Kết Luận và Hướng Phát Triển

#### 4.1 Kết Luận
- **Forecasting**: Model nào cho kết quả tốt nhất?
- **Optimization**: LP hay DRL hiệu quả hơn?
- **Trade-offs**: Cân bằng giữa cost và SLA
- **Contribution**: Đóng góp của nghiên cứu

#### 4.2 Hạn Chế
- Dataset: Chỉ một hệ thống, có thể không generalize
- Assumptions: Giả định về switching cost, VM specs
- Evaluation: Offline evaluation, chưa test trên production

#### 4.3 Hướng Phát Triển
- **Mở rộng models**: Thử các mô hình dự báo khác (Transformer, etc.)
- **Multi-objective optimization**: Pareto frontier analysis
- **Online learning**: Adaptive models với streaming data
- **Production deployment**: Test trên real cloud infrastructure
- **Cost models**: Chi tiết hóa cost models (network, storage)

---

## Tài Liệu Tham Khảo

### Sách và Bài Báo
1. Forecasting methods: ARIMA, LSTM, Prophet papers
2. Optimization: Linear Programming textbooks, RL papers
3. Cloud computing: Resource allocation surveys

### Tài Liệu Kỹ Thuật
1. scipy.optimize.linprog documentation
2. Stable-Baselines3 PPO documentation
3. Prophet, LSTM implementation guides

### Dataset
- Westermo Test System Performance Data Set

---

## Ghi Chú

- **File structure**: Tất cả code và data được tổ chức trong project directory
- **Reproducibility**: Các notebooks và scripts có thể chạy lại để reproduce kết quả
- **Documentation**: Chi tiết implementation trong code comments và docstrings

