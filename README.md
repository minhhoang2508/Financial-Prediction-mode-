# Finance Predictor - Dự Báo Tài Chính Tiên Tiến

Ứng dụng phân tích và dự báo tài chính tiên tiến cho các công ty niêm yết trên thị trường chứng khoán Việt Nam, sử dụng dữ liệu từ 2020-2024 và dự báo đến 2028.

## Tổng Quan

Ứng dụng này phân tích dữ liệu tài chính quý và dự báo xu hướng đến năm 2028 cho 5 công ty lớn:

- **Vinamilk (VNM)**: Công ty Cổ phần Sữa Việt Nam
- **Hòa Phát (HPG)**: Tập đoàn Hòa Phát
- **Hoàng Anh Gia Lai (HAG)**: Tập đoàn Hoàng Anh Gia Lai
- **FPT (FPT)**: Tập đoàn FPT
- **MB Bank (MBB)**: Ngân hàng TMCP Quân đội

## Tính Năng Chính

- **Phân tích nhiều chỉ số tài chính**: Doanh thu, lợi nhuận, vốn chủ sở hữu, tổng tài sản, v.v.
- **Dự báo tiên tiến**: Sử dụng kết hợp 6 mô hình học máy để đạt độ chính xác cao nhất
- **Trực quan hóa dữ liệu**: Biểu đồ dự báo của 5 công ty
- **Báo cáo phân tích chi tiết**: Xu hướng, tỷ lệ thay đổi, độ tin cậy dự báo
- **Khuyến nghị đầu tư**: Đánh giá và xếp hạng các công ty dựa trên kết quả dự báo

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.9 trở lên (tương thích đến Python 3.13)
- Hỗ trợ Windows, macOS và Linux

### Bước 1: Clone hoặc tải xuống dự án

```bash
git clone https://github.com/username/finance-predictor.git
cd finance-predictor
```

### Bước 2: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## Cấu Trúc Dự Án

```
finance-predictor/
├── data/                   # Dữ liệu tài chính của các công ty
│   ├── VNM.csv
│   ├── HPG.csv
│   ├── HAG.csv
│   ├── FPT.csv
│   └── MBB.csv
├── src/                    # Mã nguồn chính
│   ├── advanced_model.py   # Mô hình dự báo tiên tiến
│   ├── run_advanced_forecast.py  # Script chạy phân tích tài chính
│   ├── data_processor.py   # Xử lý dữ liệu tài chính
│   └── visualizer.py       # Tạo biểu đồ trực quan
├── output/                 # Thư mục chứa kết quả (tự động tạo)
├── run.py                  # Script chính để chạy ứng dụng
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # Tài liệu hướng dẫn
```

## Cách Sử Dụng

Chạy ứng dụng bằng lệnh:

```bash
python run.py
```

Kết quả phân tích sẽ được lưu trong thư mục `output/` bao gồm:

1. **Biểu đồ dự báo**: Dự báo từng chỉ số tài chính cho mỗi công ty
2. **Biểu đồ so sánh**: So sánh các chỉ số tài chính giữa các công ty
3. **Báo cáo phân tích**: File văn bản chứa phân tích chi tiết và khuyến nghị đầu tư

## Các Chỉ Số Tài Chính Được Phân Tích

- **Doanh thu thuần**: về bán hàng và cung cấp dịch vụ
- **Lợi nhuận gộp**: về bán hàng và cung cấp dịch vụ
- **Lợi nhuận thuần**: từ hoạt động kinh doanh
- **Lợi nhuận sau thuế**: thu nhập doanh nghiệp
- **Vốn chủ sở hữu**
- **Tổng tài sản**

## Phương Pháp Dự Báo

Ứng dụng sử dụng phương pháp dự báo tiên tiến bằng cách kết hợp 6 mô hình học máy:

1. **Linear Regression**: Mô hình hồi quy tuyến tính cơ bản
2. **Huber Regression**: Bền vững với dữ liệu nhiễu
3. **TheilSen Regression**: Bền vững với các điểm ngoại lai
4. **RANSAC Regression**: Bền vững với dữ liệu ngoại lai cực đoan
5. **Random Forest Regression**: Mạnh mẽ với dữ liệu phi tuyến
6. **Gradient Boosting Regression**: Nâng cao độ chính xác với dữ liệu phức tạp

Ứng dụng tự động đánh giá và gán trọng số cho mỗi mô hình dựa trên hiệu suất của chúng, tạo ra dự báo kết hợp có độ chính xác cao.

## Đánh Giá Xu Hướng

Xu hướng được đánh giá dựa trên tỷ lệ thay đổi dự báo:

- **Tăng**: Tỷ lệ thay đổi > 5%
- **Giảm**: Tỷ lệ thay đổi < -5%
- **Ổn định**: -5% ≤ Tỷ lệ thay đổi ≤ 5%

Mỗi dự báo đều đi kèm với một chỉ số "độ tin cậy" để đánh giá mức độ chắc chắn của dự báo.


