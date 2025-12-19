# Chương 3.1. Kiến Trúc Tổng Quan

## 3.1.1. Tổng Quan Hệ Thống

Hệ thống dự báo nhu cầu và tối ưu hóa phân bổ tài nguyên trên Cloud Infrastructure được thiết kế theo kiến trúc ba tầng, hoạt động theo luồng xử lý tuần tự từ giám sát, dự báo đến tối ưu hóa. Hệ thống quản lý tài nguyên tự động trên cả môi trường Cloud (Azure, AWS, GCP) và On-premises infrastructure.

**Mục tiêu thiết kế**:
- Tự động hóa hoàn toàn quy trình từ thu thập dữ liệu đến phân bổ tài nguyên
- Hỗ trợ đa nền tảng (multi-cloud và hybrid cloud)
- Dự báo chính xác với nhiều mô hình khác nhau
- Tối ưu hóa đa mục tiêu (chi phí, SLA, switching cost)
- Kiến trúc modular dễ mở rộng

## 3.1.2. Các Thành Phần Chính

Hệ thống bao gồm ba thành phần chính hoạt động theo trình tự:

### 3.1.2.1. Resource System Monitoring (Giám Sát Hệ Thống Tài Nguyên)

**Vai trò và chức năng**: Thành phần này đóng vai trò là lớp thu thập dữ liệu đầu tiên của hệ thống, chịu trách nhiệm giám sát và thu thập các metrics hiệu suất từ tất cả các nguồn tài nguyên trong infrastructure (Cloud VMs: Azure, AWS, GCP và On-premises servers). Đây là nền tảng dữ liệu cho toàn bộ quy trình dự báo và tối ưu hóa sau này.

**Ý nghĩa và tầm quan trọng**: 
- **Nền tảng cho dự báo**: Dữ liệu lịch sử đầy đủ và chính xác là yêu cầu tiên quyết cho việc dự báo. Không có dữ liệu giám sát tốt, các mô hình dự báo sẽ không thể học được patterns và trends, dẫn đến dự báo không chính xác.
- **Hỗ trợ hybrid cloud**: Hỗ trợ đa nguồn cho phép hệ thống quản lý tài nguyên trên cả cloud và on-premises một cách thống nhất, phù hợp với kiến trúc hybrid cloud hiện đại và tránh vendor lock-in.
- **Phát hiện patterns**: Dữ liệu time-series với độ phân giải cao cho phép phát hiện các patterns ngắn hạn (daily cycles) và dài hạn (weekly/monthly trends), cần thiết cho việc dự báo chính xác và đưa ra quyết định phân bổ tài nguyên phù hợp.

**Output**: Dữ liệu thô được lưu trữ dưới dạng CSV với format time-series, sẵn sàng cho giai đoạn tiền xử lý và phân tích tiếp theo.

### 3.1.2.2. Workload & Resource Forecasting (Dự Báo Workload và Tài Nguyên)

**Vai trò và chức năng**: Thành phần này là trung tâm của hệ thống, chịu trách nhiệm phân tích dữ liệu lịch sử và tạo ra các dự báo về nhu cầu tài nguyên trong tương lai. Dự báo chính xác là cơ sở để đưa ra các quyết định phân bổ tài nguyên tối ưu.

**Các bước chính và ý nghĩa**:

**1. ETL Pipeline - Tiền xử lý dữ liệu**:

- **Extract, Transform, Load**: Chuyển đổi dữ liệu thô thành dữ liệu sẵn sàng cho machine learning thông qua các bước:
  - Tạo target variables từ các metrics gốc
  - Feature selection dựa trên correlation để loại bỏ noise và giảm dimensionality
  - Data cleaning để đảm bảo chất lượng dữ liệu
  - Normalization để chuẩn hóa scale
  - Train/test split với sequential split để tránh data leakage

**Ý nghĩa**: ETL pipeline đảm bảo dữ liệu chất lượng cao, giúp mô hình học được patterns thực sự thay vì memorizing dữ liệu. Feature selection giúp giảm overfitting và cải thiện generalization. Sequential split và data leakage prevention đảm bảo đánh giá mô hình một cách công bằng và realistic.

**2. Model Training và Forecasting**:

Hệ thống hỗ trợ 4 mô hình dự báo: **ARIMAX** (statistical, xử lý seasonality), **SVR** (non-linear patterns), **Random Forest** (feature interactions), và **Hybrid Prophet + LSTM** (kết hợp seasonality và long-term dependencies).

**Ý nghĩa**: Sử dụng nhiều mô hình cho phép so sánh và chọn ra mô hình tốt nhất, hoặc ensemble để tăng độ chính xác. Hybrid approach tận dụng ưu điểm của cả statistical methods và deep learning, đảm bảo dự báo chính xác cho cả patterns ngắn hạn và dài hạn.

**Output**: Các mô hình đã được huấn luyện và đánh giá, sẵn sàng tạo forecast cho nhu cầu tài nguyên trong tương lai. Forecast values này sẽ được sử dụng bởi Resource Allocation Optimizer để đưa ra quyết định phân bổ tài nguyên.

### 3.1.2.3. Resource Allocation Optimizer (Tối Ưu Hóa Phân Bổ Tài Nguyên)

**Vai trò và chức năng**: Đây là thành phần quyết định cuối cùng của hệ thống, chịu trách nhiệm chuyển đổi các dự báo thành các quyết định phân bổ tài nguyên cụ thể. Thành phần này phải cân bằng giữa nhiều mục tiêu mâu thuẫn: đảm bảo đủ tài nguyên (SLA), tối thiểu hóa chi phí, và giảm thiểu switching cost.

**Các bước chính và ý nghĩa**:

**1. Forecast → Resource Requirements**:

- **Host Capacity Thresholds**: Xác định ngưỡng capacity của host machine (ví dụ: 70% CPU, 75% Memory). Khi forecast vượt quá ngưỡng này, hệ thống sẽ tính toán phần overflow cần allocate từ VM.

- **Tính toán Overflow**: Chuyển đổi forecast thành resource requirements cụ thể (CPU cores và Memory GB) cần allocate từ VM.

**Ý nghĩa**: Forecast từ mô hình là giá trị dự đoán về utilization, không phải trực tiếp là yêu cầu VM. Bước chuyển đổi này là cần thiết để translate forecast thành actionable requirements. Threshold-based approach cho phép hệ thống có flexibility trong việc quyết định khi nào cần thêm tài nguyên, đảm bảo host luôn có buffer an toàn và tránh allocate quá sớm hoặc quá muộn.

**2. Requirements → VM Allocation**:

Hệ thống hỗ trợ hai phương pháp tối ưu hóa:

**a) Linear Programming (LP) - Baseline Method**:

- **Formulation**: Mô hình hóa bài toán như quy hoạch tuyến tính với objective minimize cost và constraints đảm bảo đáp ứng CPU/Memory requirements.

- **Reactive Mode**: Giải quyết độc lập tại mỗi timestep, không xem xét switching cost.

**Ý nghĩa**: LP đảm bảo tìm được nghiệm tối ưu toàn cục, giải nhanh, và dễ implement. Tuy nhiên, LP không xem xét switching cost và có thể tạo ra allocation schedule không smooth. LP đóng vai trò baseline để so sánh với các phương pháp khác và hữu ích khi cần giải nhanh.

**b) Deep Reinforcement Learning (DRL) với PPO**:

- **Environment**: Agent nhận state bao gồm current demand, forecast, current allocation, và time features. Action là số lượng từng loại VM cần allocate.

- **Reward Function**: Được thiết kế với 5 components: SLA compliance, overflow penalty, switching cost, VM cost, và efficiency bonus để guide agent học được behavior mong muốn.

- **Scenarios**: Hỗ trợ nhiều scenarios (Overload: ưu tiên SLA, Cost: ưu tiên chi phí) với reward weights khác nhau.

**Ý nghĩa**: DRL có thể học được patterns phức tạp từ dữ liệu, xem xét switching cost để tạo ra smooth allocation schedule, và có thể nhìn xa về tương lai nhờ forecast trong state. DRL phù hợp cho long-term optimization và tạo ra allocation schedule tốt hơn về mặt tổng thể. Tuy nhiên, DRL cần nhiều thời gian training và không đảm bảo optimality như LP.

**Tại sao cần cả hai phương pháp**: 
- **LP**: Baseline để so sánh, đảm bảo có giải pháp tối ưu toán học, hữu ích khi cần giải nhanh.
- **DRL**: Phù hợp khi cần xem xét switching cost và học patterns phức tạp, tạo ra allocation schedule tốt hơn về mặt tổng thể.

**Output**: Cả hai phương pháp đều tạo ra VM allocation schedules với đầy đủ thông tin về allocation, costs, utilization, và SLA violations. Các schedules này được so sánh để đánh giá hiệu quả của từng phương pháp.

## 3.1.3. Luồng Xử Lý Tổng Quan

```
┌─────────────────────────┐
│ Resource System         │
│ Monitoring              │
│ • Cloud VMs (Azure,    │
│   AWS, GCP)            │
│ • On-premises Server   │
└───────────┬─────────────┘
            │ Raw Data
            ▼
┌─────────────────────────┐
│ Workload & Resource     │
│ Forecasting            │
│ • ETL Pipeline         │
│ • Model Training       │
│ • Forecasting          │
└───────────┬─────────────┘
            │ Forecast
            ▼
┌─────────────────────────┐
│ Resource Allocation     │
│ Optimizer               │
│ • LP (Baseline)        │
│ • DRL (PPO)            │
└───────────┬─────────────┘
            │ Allocation
            ▼
┌─────────────────────────┐
│ Evaluation &            │
│ Comparison             │
└─────────────────────────┘
```

**Các giai đoạn**:
1. **Thu thập**: Giám sát thu thập metrics từ Cloud và On-premises
2. **Dự báo**: ETL và các mô hình dự báo tạo forecast values
3. **Tối ưu hóa**: Chuyển đổi forecast thành requirements và tối ưu hóa phân bổ VM
4. **Đánh giá**: So sánh hiệu quả giữa các phương pháp

## 3.1.4. Kiến Trúc Module

Hệ thống được tổ chức theo kiến trúc modular:

- **Data Processing Module** (`etl_cloud_resource_forecasting.ipynb`): ETL pipeline, feature engineering
- **Forecasting Module** (`model_*.ipynb`): Training và evaluation các mô hình dự báo
- **Optimization Module**:
  - LP: `vm_resource_planner.py`
  - DRL: `rl/` directory, `train_ppo.py`, `eval_ppo.py`
- **Evaluation Module**: So sánh và đánh giá kết quả

## 3.1.5. Tính Năng Nổi Bật

- **Hỗ trợ đa cloud**: Quản lý tài nguyên trên Azure, AWS, GCP đồng thời
- **Hybrid cloud**: Hỗ trợ cả Cloud và On-premises infrastructure
- **Multi-model forecasting**: Sử dụng nhiều mô hình để tăng độ chính xác
- **Multi-objective optimization**: Cân bằng chi phí, SLA, và switching cost
- **Flexible scenarios**: Hỗ trợ nhiều scenarios (Overload, Cost) với weights khác nhau

## 3.1.6. Kết Luận

Kiến trúc hệ thống được thiết kế với các nguyên tắc: **Modularity** (dễ bảo trì và mở rộng), **Scalability** (xử lý lượng dữ liệu lớn), **Flexibility** (hỗ trợ nhiều mô hình và thuật toán), và **Automation** (tự động hóa toàn bộ quy trình). Kiến trúc này cho phép hệ thống đáp ứng các yêu cầu của bài toán dự báo và tối ưu hóa tài nguyên một cách hiệu quả.
