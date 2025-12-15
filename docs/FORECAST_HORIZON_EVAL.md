# Forecast Horizon Evaluation Guide

Mục tiêu: tìm horizon H tốt nhất cho từng model (SVR, Random Forest, ARIMAX, Hybrid) trên 3 target (`cpu_total_usage`, `memory_usage_pct`, `system_load`), và so sánh giữa các model để chọn H chung cho PPO.

## 1. Chuẩn bị
- Data: `processed_data/[target]/X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv` (đã có, time-ordered).
- Models: lấy model mới nhất từ `models/` cho từng target & model type.
- Horizon cần thử (gợi ý, bước 30s): `HORIZONS = [6, 12, 18, 24, 30, 36]`  → 3, 6, 9, 12, 15, 18 phút.

## 2. Ý tưởng đánh giá H
### 2.1 Rolling-origin (walk-forward) đúng nghĩa cho horizon H
Mục tiêu của “forecast H bước tới” là:
- Đứng tại thời điểm **t** (start), chỉ dùng thông tin “có sẵn tại t” (và lịch sử trước đó).
- Dự báo giá trị mục tiêu tại **t+H** (hoặc **t+H-1** nếu bạn đếm H là số bước, bắt đầu từ t+1).
- Lặp t = t0, t0+1, t0+2… cho đến khi **không còn đủ H bước** trong tập test.

Bạn mô tả như sau là **đúng** về mặt ý tưởng:
- Ở t, generate một dải dự báo \(t+1, t+2, \dots, t+H\) (multi-step path).
- Sau đó dịch sang t+1, chỉ cần quan tâm giá trị ở **điểm cuối** mới \(t+H+1\) (nếu mục tiêu là đánh giá “H-step ahead” tại mỗi start).
- Các điểm forecast “lố” qua cuối tập test thì **không tính vào metric**.

### 2.2 Hai cách chấm điểm phổ biến (đừng nhầm)
**A) Chấm đúng “H-step ahead” (chỉ lấy điểm cuối):**
- Với mỗi start t, bạn forecast H bước, nhưng **chỉ lấy dự báo ở bước H**:
  - \( \hat{y}_{t+H} \) (hoặc \( \hat{y}_{t+H-1} \) tùy quy ước chỉ số)
- So sánh với \( y_{t+H} \) tương ứng.
- Đây là cách dùng để trả lời câu hỏi: “dự báo xa H bước thì sai số thế nào?”

**B) Chấm toàn bộ đường dự báo (multi-step path):**
- Với mỗi start t, bạn lấy cả vector dự báo \(\hat{y}_{t+1..t+H}\)
- Và có thể tính metric theo từng step k=1..H, hoặc gộp tất cả.
- Cách này trả lời câu hỏi: “sai số tăng dần theo từng bước trong 1 rollout ra sao?”

Notebook hiện tại đang hướng tới **(A)** (đánh giá riêng từng H).

### 2.3 Pseudocode (chuẩn, dễ đối chiếu code)
Giả sử test có N điểm, chỉ số 0..N-1, và H là “số bước ahead”, thì:

```python
# start t chạy tới khi t+H không vượt N-1
for t in range(0, N - H):
    # tạo dự báo H bước: y_hat[t+1], ..., y_hat[t+H]
    path = forecast_from_t(t, steps=H)
    y_pred.append(path[-1])        # chỉ lấy bước cuối (H-step)
    y_true.append(y_test[t + H])   # so với ground truth cùng thời điểm
```

Nếu bạn dùng quy ước “bước cuối là t+H-1” (như notebook hiện tại đang dùng `target_idx = t + H - 1`) thì thay `t+H` bằng `t+H-1` cho nhất quán.

## 3. Hàm rolling forecast (khái niệm)
- **ARIMAX**: dùng `rolling_forecast` / `forecast_with_horizon` trong `model_utils.py` (đã có).
### 3.1 ARIMAX (multi-step path → lấy điểm cuối)
- Với mỗi start t: gọi `model.forecast(steps=H, exog=...)` để ra vector dự báo H bước.
- Lấy phần tử cuối làm dự báo “H-step ahead”.

### 3.2 SVR / Random Forest: cần phân biệt “eval đúng” và “mô hình có support multi-step hay không”
**Trường hợp bạn muốn đúng như mô tả (generate t+1..t+H, rồi trượt cửa sổ):**
- Bạn cần feature set có **lag của y** (và/hoặc time features) để sau mỗi bước có thể “cập nhật state” và dự báo bước tiếp theo (recursive).
- Pseudocode (recursive) sẽ giống:
  - lấy feature tại t,
  - dự đoán \(\hat{y}_{t+1}\),
  - cập nhật các cột lag bằng \(\hat{y}_{t+1}\),
  - dự đoán \(\hat{y}_{t+2}\), … đến \(\hat{y}_{t+H}\).

**Nếu feature set KHÔNG có lag của y** (như dataset hiện tại), thì SVR/RF *không thể* tự “roll” ra t+1..t+H một cách hợp lệ chỉ từ X[t].
Trong trường hợp này bạn có 2 lựa chọn:
- **(i) Re-train theo từng H (direct model):** train mapping `X[t] -> y[t+H]` cho mỗi H, rồi đánh giá.
- **(ii) Chấp nhận “oracle future features” (không khuyến nghị cho bài toán vận hành):**
  dùng `X[t+H]` để dự đoán `y[t+H]`. Cách này thường làm metric “đẹp” vì vô tình dùng thông tin tương lai (nhiều feature thực tế không biết trước).

### 3.3 Hybrid Prophet+LSTM
- Prophet dự báo trend/seasonality theo thời gian.
- LSTM dự báo residual theo kiểu recursive.
- Lấy điểm cuối trong path H bước để chấm “H-step ahead”, hoặc chấm toàn bộ path nếu cần.

## 4. Quy trình chạy trong notebook mới
1) Load data + models (dùng `get_latest_model`).
2) Định nghĩa `HORIZONS`.
3) Viết hàm `rolling_forecast_<model_type>(..., horizon=H)` cho SVR/RF (ARIMAX đã có).
4) Vòng lặp đánh giá:
   ```python
   records = []
   for model in ['svr','random_forest','arimax','hybrid_prophet_lstm']:
       for target in ['cpu_total_usage','memory_usage_pct','system_load']:
           for H in HORIZONS:
               y_pred, y_true = rolling_forecast_<model>(..., horizon=H)
               m = calculate_metrics(y_true, y_pred)
               records.append({'model':model,'target':target,'H':H,
                               'mae':m['mae'],'rmse':m['rmse'],'r2':m['r2']})
   df = pd.DataFrame(records)
   df.to_csv('forecast_result/horizon_eval.csv', index=False)
   ```
5) Visualization:
   - Line plot MAE vs H cho từng target, nhiều model.
   - Pivot nhanh: `df.pivot_table(index='H', columns=['target','model'], values='mae')`.

## 5. Cách chọn H tốt nhất
- Ngưỡng lỗi: chọn H lớn nhất sao cho (ví dụ) CPU MAE < 0.5 core, RAM MAE < 5%, R² > 0.
- Điểm gãy: chọn H ngay trước khi MAE tăng vọt.
- Horizon chung cho PPO: `H_final = min(H_cpu*, H_mem*, H_load*)` để không dựa vào biến dự báo kém.

## 6. Kỳ vọng đầu ra
- File kết quả: `forecast_result/horizon_eval.csv` với cột `model, target, H, mae, rmse, r2`.
- Biểu đồ MAE vs H (per target).
- Quyết định: model nào cho từng target (RAM: SVR, CPU: RF, Load: RF theo kết quả hiện tại) và horizon chung (thường 12–18 bước).

## 7. Lưu ý
- Không cần re-ETL. Dùng đúng train/test đã có; test là chuỗi nối tiếp train.
- Với SVR/RF, muốn forecast nhiều bước đúng nghĩa (recursive) thì **bắt buộc** có lag của y (hoặc state tương đương). Nếu feature set không có lag, hãy:
  - bổ sung lag trong ETL, hoặc
  - train direct model riêng cho từng H.
- Thời gian chạy: tăng theo số H và số điểm test; cân nhắc giảm `HORIZONS` khi thử nhanh.

