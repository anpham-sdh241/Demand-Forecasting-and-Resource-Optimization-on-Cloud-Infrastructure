# Chương 3.2. Dữ Liệu, Tiền Xử Lý và Huấn Luyện Mô Hình Dự Báo

## 3.2.1. Mô Tả Bộ Dữ Liệu

### 3.2.1.1. Nguồn Dữ Liệu

Dự án sử dụng **Westermo Test System Performance Data Set**, một bộ dữ liệu công khai được tạo ra trong dự án AIDOaRt (AI-augmented Automation for DevOps) để nghiên cứu về anomaly detection trong hệ thống test tự động. [[1]](https://github.com/westermo/test-system-performance-dataset)

**Bối cảnh và mục đích của dataset**:
- Dataset được thu thập từ các hệ thống test chạy nightly testing của các cyber-physical systems tại Westermo Network Technologies AB, Västerås, Sweden
- Mục đích ban đầu: Phát hiện các trạng thái bất thường của hệ thống test để đảm bảo độ tin cậy của kết quả test (ví dụ: nightly testing không dừng, không khởi động, hệ thống không hoạt động, load bất thường)
- Dataset đã được sử dụng trong các hackathons, nghiên cứu sinh viên, và các dự án về sustainable software engineering

**Quy trình thu thập dữ liệu**:
- Các servers trong hệ thống test chạy **node exporter** - một công cụ để export performance data từ servers
- Dữ liệu được lưu trữ bằng **Grafana** và sau đó được export sang CSV bằng Python script
- Dữ liệu đã được anonymized và obfuscated để bảo vệ thông tin của Westermo

### 3.2.1.2. Đặc Điểm Dữ Liệu

**Cấu trúc dữ liệu**:
- **Format**: 19 CSV files, mỗi file đại diện cho một test system
- **Kích thước**: Mỗi file chứa khoảng 86,000 dòng dữ liệu (tổng cộng khoảng 360 MB)
- **Sampling frequency**: Dữ liệu được lấy mẫu khoảng 2 lần mỗi phút (tương đương mỗi 30 giây)
- **Time span**: 30 ngày liên tục
- **Số lượng metrics**: 23-24 time series metrics mỗi file

**Các loại metrics**:

- **System Load**: 
  - `load-1m`, `load-5m`, `load-15m`: System load trung bình trong 1, 5, và 15 phút

- **Memory Metrics** (đơn vị: bytes):
  - `sys-mem-total`: Tổng dung lượng memory (constant)
  - `sys-mem-available`: Memory khả dụng (free + cache)
  - `sys-mem-free`: Memory chưa sử dụng
  - `sys-mem-cache`: Memory dùng cho cache
  - `sys-mem-buffered`: Memory dùng cho kernel buffers
  - `sys-mem-swap-total`: Tổng dung lượng swap (constant)
  - `sys-mem-swap-free`: Swap khả dụng

- **CPU Metrics**:
  - `cpu-user`: Tỷ lệ thay đổi thời gian CPU dùng cho user space processes/threads
  - `cpu-system`: Tỷ lệ thay đổi thời gian CPU dùng cho kernel space threads
  - `cpu-iowait`: Tỷ lệ thay đổi thời gian CPU chờ I/O operations

- **Disk I/O Metrics**:
  - `disk-io-time`: Tỷ lệ thay đổi thời gian I/O operations
  - `disk-bytes-read`, `disk-bytes-written`: Tỷ lệ thay đổi bytes đọc/ghi
  - `disk-io-read`, `disk-io-write`: Tỷ lệ thay đổi số lượng read/write operations

- **System Metrics**:
  - `sys-fork-rate`: Tỷ lệ thay đổi số lượng forks
  - `sys-interrupt-rate`: Tỷ lệ thay đổi interrupts
  - `sys-context-switch-rate`: Tỷ lệ thay đổi context switches
  - `sys-thermal`: Nhiệt độ hệ thống trung bình (Celsius)
  - `server-up`: Trạng thái server (values > 0 = available)

**Dữ liệu sử dụng trong dự án**:
- **File**: `data/system-1.csv` (một trong 19 test systems)
- **Kích thước**: 85,749 dòng × 24 cột
- **Timestamp**: Số giây kể từ khi bắt đầu thu thập dữ liệu

### 3.2.1.3. Tính Phù Hợp Cho Bài Toán Dự Báo

Westermo Test System Performance Data Set là một lựa chọn phù hợp cho bài toán dự báo nhu cầu tài nguyên cloud do đáp ứng đầy đủ các yêu cầu của một dataset time-series chất lượng cao. Với frequency lấy mẫu cao (khoảng 2 lần mỗi phút) và time span đủ dài (30 ngày, tương đương ~86,000 timesteps), dataset này cung cấp độ phân giải thời gian cần thiết để phát hiện cả các patterns ngắn hạn (như daily cycles trong nightly testing) và dài hạn (như weekly trends). Đặc biệt, dữ liệu từ nightly testing systems thường thể hiện các seasonal patterns rõ ràng theo ngày và tuần, điều này rất quan trọng cho việc nghiên cứu seasonality trong các mô hình dự báo như Prophet và ARIMAX. 

Hơn nữa, đây là dữ liệu thực tế từ hệ thống công nghiệp (Westermo Network Technologies AB), không phải dữ liệu synthetic hay simulated, đảm bảo tính realistic và khả năng generalizability của các mô hình được train trên dataset này. Việc có nhiều metrics khác nhau (24 metrics bao gồm CPU, Memory, System Load, Disk I/O, System metrics) cho phép nghiên cứu mối quan hệ phức tạp giữa các thành phần hệ thống và tạo ra các features phong phú cho mô hình dự báo. Với khối lượng dữ liệu đủ lớn (~86,000 timesteps), dataset này đảm bảo có đủ dữ liệu để chia thành training set (80%) và test set (20%) một cách hợp lý, đồng thời vẫn giữ được sequential order để tránh data leakage - một yêu cầu quan trọng trong time series forecasting.

## 3.2.2. ETL Pipeline (Extract, Transform, Load)

### 3.2.2.0. Tổng Quan

ETL Pipeline được thiết kế theo kiến trúc trong Hình 3.2, chuyển đổi dữ liệu thô thành dữ liệu sẵn sàng cho machine learning thông qua ba giai đoạn chính: **Extract** (trích xuất và khám phá dữ liệu), **Transform** (preprocessing, EDA, feature selection, normalization), và **Load** (feature extraction, data splitting, export). Pipeline này đảm bảo chất lượng dữ liệu, loại bỏ noise và data leakage, đồng thời chuẩn bị dữ liệu phù hợp cho việc training các mô hình dự báo.

### 3.2.2.1. Extract - Trích Xuất Dữ Liệu

Giai đoạn này đọc và khám phá dữ liệu thô từ CSV file (`data/system-1.csv`), bao gồm load dữ liệu, khám phá cấu trúc (số dòng, số cột, kiểu dữ liệu), phân tích thống kê cơ bản, và chuyển đổi timestamp sang datetime format. Giai đoạn này giúp hiểu rõ về dữ liệu trước khi xử lý và phát hiện các vấn đề tiềm ẩn như missing values, outliers, hoặc data types không đúng.

### 3.2.2.2. Transform - Preprocessing & EDA

Giai đoạn này làm sạch và chuyển đổi dữ liệu thành format phù hợp cho machine learning. **Tạo Target Variables**: Ba biến mục tiêu được tạo từ các metrics gốc - `memory_usage_pct` (tỷ lệ memory sử dụng), `cpu_total_usage` (tổng CPU usage), và `system_load` (system load 1-minute). **Exploratory Data Analysis**: Phân tích phân phối, phát hiện outliers và missing values, phân tích correlation, và visualize time series để phát hiện trends và seasonality. **Data Cleaning**: Xử lý missing values, loại bỏ constant columns, và xử lý outliers. **Feature Selection**: Tính correlation matrix và chọn features có correlation cao với target (|r| > 0.3), giảm từ 24 features xuống còn 7-10 features cho mỗi target. **Normalization**: Sử dụng StandardScaler để chuẩn hóa features về cùng scale (mean=0, std=1), lưu statistics để sử dụng khi inference.

**Ý nghĩa**: Tạo target variables phù hợp với bài toán, feature selection giúp giảm dimensionality và loại bỏ noise, normalization đảm bảo các features có cùng scale, và data cleaning đảm bảo chất lượng dữ liệu.

### 3.2.2.3. Load - Feature Extraction và Data Splitting

Giai đoạn này chuẩn bị dữ liệu cuối cùng cho training và evaluation. **Feature Extraction**: Extract features phù hợp cho từng target variable dựa trên correlation analysis, loại bỏ component variables để tránh data leakage (ví dụ: loại `sys-mem-total`, `sys-mem-available` khỏi features cho `memory_usage_pct`). **Data Splitting**: Chia dữ liệu thành training set (80%, 68,599 timesteps) và test set (20%, 17,150 timesteps) với sequential split để giữ nguyên thứ tự thời gian, đảm bảo không có data leakage. **Export Processed Data**: Lưu processed datasets (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) và metadata (`feature_metadata.json`, `normalization_stats.json`) cho mỗi target variable.

**Ý nghĩa**: Feature extraction đảm bảo chỉ sử dụng features có ảnh hưởng mạnh, sequential split đảm bảo đánh giá realistic, và data leakage prevention đảm bảo mô hình học được patterns thực sự.

## 3.2.3. Huấn Luyện và Đánh Giá Mô Hình Dự Báo

### 3.2.3.0. Tổng Quan

Quy trình huấn luyện và đánh giá mô hình dự báo được thực hiện theo kiến trúc trong Hình 3.2, bao gồm: **Hyperparameter Tuning** để tìm cấu hình tối ưu, **Parameter Selection** dựa trên validation performance, **Fit Model** với training data, **Model Evaluation** trên test set, và **Model Selection & Export** để chọn mô hình tốt nhất. Hệ thống hỗ trợ 4 mô hình dự báo khác nhau để so sánh và chọn ra giải pháp tốt nhất cho từng target variable.

### 3.2.3.1. Các Mô Hình Dự Báo

Hệ thống hỗ trợ 4 mô hình dự báo: **ARIMAX** (statistical, xử lý trend và seasonality với exogenous variables, sử dụng `statsmodels`), **SVR** (kernel-based regression với RBF kernel cho non-linear patterns, sử dụng `scikit-learn`), **Random Forest** (ensemble method xử lý feature interactions, sử dụng `scikit-learn`), và **Hybrid Prophet + LSTM** (kết hợp Prophet để xử lý seasonality và LSTM để học long-term dependencies từ residuals, sử dụng `prophet` và `PyTorch`). Mỗi mô hình có strengths riêng và phù hợp với các loại patterns khác nhau trong dữ liệu.

### 3.2.3.2. Quy Trình Huấn Luyện

Quy trình huấn luyện theo kiến trúc trong Hình 3.2: **Hyperparameter Tuning** - Sử dụng grid search hoặc random search để tìm hyperparameters tối ưu cho mỗi mô hình (ARIMAX: orders và seasonal orders, SVR: C và gamma, Random Forest: n_estimators và max_depth, Hybrid: Prophet parameters và LSTM architecture), validation được thực hiện trên một phần của training set. **Parameter Selection** - Chọn bộ parameters tốt nhất dựa trên validation performance (RMSE hoặc MAE). **Fit Model** - Train mô hình với training data và selected parameters, mỗi mô hình được train riêng cho từng target variable. Với Hybrid Prophet + LSTM, Prophet được train trước để capture seasonality, sau đó LSTM được train trên residuals để capture patterns còn lại.

**Ý nghĩa**: Hyperparameter tuning đảm bảo performance tốt nhất, parameter selection giúp tránh overfitting, và training riêng cho từng target variable cho phép tối ưu hóa cho từng bài toán cụ thể.

### 3.2.3.3. Đánh Giá và Chọn Mô Hình

**Model Evaluation**: Sử dụng test set (unseen data) để đánh giá performance với các metrics: **MAE** (trung bình sai số tuyệt đối), **RMSE** (nhạy cảm với outliers), **MAPE** (phần trăm sai số, dễ interpret), và **R²** (độ phù hợp của mô hình). **So Sánh Mô Hình**: So sánh performance của 4 mô hình trên cùng test set cho cả 3 target variables, xem xét trade-off giữa accuracy và complexity (ARIMAX: đơn giản nhưng có thể không capture non-linear, SVR: tốt cho non-linear nhưng chậm, Random Forest: robust nhưng có thể overfit, Hybrid: phức tạp nhất nhưng có thể đạt accuracy cao nhất). **Model Selection & Export**: Chọn mô hình tốt nhất dựa trên RMSE/MAE thấp nhất, MAPE thấp, R² cao, và balance giữa accuracy và complexity. Trong dự án này, **Hybrid Prophet + LSTM** thường được chọn làm default model do có performance tốt nhất. Models được export vào `models/` với format phù hợp, evaluation results được lưu vào `results_*.json`, và normalization statistics cùng feature metadata được lưu để đảm bảo reproducibility.

**Ý nghĩa**: Evaluation trên test set đảm bảo đánh giá khách quan về generalization, so sánh nhiều mô hình cho phép chọn giải pháp tốt nhất, và model persistence đảm bảo reproducibility và consistency.

## 3.2.4. Kết Quả và Kết Luận

**Kết quả ETL Pipeline**:
- Dữ liệu đã được preprocessed và sẵn sàng cho training
- 3 target variables được tạo: Memory Usage %, CPU Total Usage, System Load
- Feature selection giảm số lượng features xuống còn 7-10 features cho mỗi target (từ 24 features ban đầu)
- Train/test split đảm bảo không có data leakage

**Kết quả Model Training**:
- 4 mô hình đã được train và đánh giá trên test set
- Hybrid Prophet + LSTM được chọn làm default model do có performance tốt nhất
- Models đã được export và sẵn sàng sử dụng cho forecasting

**Tầm quan trọng**:
- ETL Pipeline đảm bảo chất lượng dữ liệu và loại bỏ các vấn đề có thể ảnh hưởng đến mô hình
- Multiple models cho phép so sánh và chọn ra giải pháp tốt nhất
- Proper evaluation đảm bảo mô hình có khả năng generalization tốt trên unseen data
- Model persistence đảm bảo có thể sử dụng mô hình một cách nhất quán trong production

## Tài Liệu Tham Khảo

1. **Strandberg, P. E., & Marklund, Y. (2023).** The Westermo test system performance data set. Retrieved from https://github.com/westermo/test-system-performance-dataset

2. **Eramo, R., et al. (2021).** AIDOaRt: AI-augmented Automation for DevOps, a Model-based Framework for Continuous Development in Cyber-Physical Systems. In Euromicro Conference on Digital Systems Design.

3. **Salahshour Torshizi, S. (2022).** Software performance anomaly detection through analysis of test data by multivariate techniques. Master's thesis, Uppsala University.

4. **Strandberg, P. E. (2021).** Automated System-Level Software Testing of Industrial Networked Embedded Systems. PhD thesis, Mälardalen University. Online at: https://arxiv.org/abs/2111.08312

