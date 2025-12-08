# Forecast Horizon Evaluation Guide

Mục tiêu: tìm horizon H tốt nhất cho từng model (SVR, Random Forest, ARIMAX, Hybrid) trên 3 target (`cpu_total_usage`, `memory_usage_pct`, `system_load`), và so sánh giữa các model để chọn H chung cho PPO.

## 1. Chuẩn bị
- Data: `processed_data/[target]/X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv` (đã có, time-ordered).
- Models: lấy model mới nhất từ `models/` cho từng target & model type.
- Horizon cần thử (gợi ý, bước 30s): `HORIZONS = [6, 12, 18, 24, 30, 36]`  → 3, 6, 9, 12, 15, 18 phút.

## 2. Ý tưởng đánh giá H
Tại mỗi vị trí start `s` trên tập test:
- Giả sử đang ở thời điểm `s`, biết toàn bộ lịch sử trước đó (train + phần test trước `s`).
- Forecast H bước tới → lấy giá trị ở bước H: `y_hat(s+H)`.
- So sánh với `y_true(s+H)`.
Lặp cho nhiều `s` (đến hết test trừ H) → tính MAE/RMSE/R² riêng cho horizon H. Lặp cho mọi H trong `HORIZONS`.

## 3. Hàm rolling forecast (khái niệm)
- **ARIMAX**: dùng `rolling_forecast` / `forecast_with_horizon` trong `model_utils.py` (đã có).
- **SVR / Random Forest** (recursive):
  - Giữ một bản sao feature tại thời điểm `s` (có chứa lag/time features).
  - Dự đoán liên tiếp từng bước, sau mỗi bước cập nhật các cột lag bằng giá trị vừa dự đoán.
  - Sau H bước, lấy giá trị cuối cùng làm `y_hat(s+H)`.
- **Hybrid Prophet+LSTM**: tương tự ARIMAX, giữ state residual và dự báo nối tiếp H bước.

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
- Với SVR/RF, cần đảm bảo cập nhật đúng các cột lag khi forecast nhiều bước (recursive). Nếu feature set không có lag, xem xét bổ sung lag trong ETL cho phiên bản đánh giá H.
- Thời gian chạy: tăng theo số H và số điểm test; cân nhắc giảm `HORIZONS` khi thử nhanh.

