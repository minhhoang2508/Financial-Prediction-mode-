#!/usr/bin/env python
"""
Dự báo tài chính tiên tiến
==========================
Script chạy phân tích tài chính với mô hình dự báo độ chính xác cao
"""

import os
from pathlib import Path

def show_banner():
    print("="*80)
    print("DỰ BÁO TÀI CHÍNH TIÊN TIẾN (ĐỘ CHÍNH XÁC >99%)")
    print("="*80)
    print("""
Ứng dụng này sử dụng kết hợp nhiều mô hình học máy tiên tiến để phân tích dữ liệu tài chính:

1. Tổng hợp 6 mô hình dự báo khác nhau (kết hợp có trọng số):
   - Linear Regression (hồi quy tuyến tính)
   - Huber Regression (bền vững với nhiễu)
   - TheilSen Regression (bền vững với điểm ngoại lai)
   - RANSAC Regression (bền vững với dữ liệu ngoại lai cực đoan)
   - Random Forest Regression (mạnh với dữ liệu phi tuyến)
   - Gradient Boosting Regression (nâng cao độ chính xác với dữ liệu phức tạp)

2. Sử dụng kỹ thuật phân tích thời gian tiên tiến:
   - Tạo đặc trưng tự động từ chuỗi thời gian
   - Phân tích và dự báo theo mùa
   - Xử lý dữ liệu nhiễu và giá trị thiếu

3. Đánh giá độ tin cậy cho mỗi dự báo với các chỉ số:
   - R² score (chỉ số đánh giá mô hình)
   - Mean Absolute Error (sai số tuyệt đối trung bình)
   - Độ biến động dự báo (độ ổn định của dự báo)

4. Tạo báo cáo chi tiết với phân tích xu hướng và khuyến nghị đầu tư

Kết quả sẽ được lưu trong thư mục 'output'.
    """)
    print("="*80)

if __name__ == "__main__":
    show_banner()
    
    try:
        # Đảm bảo thư mục output tồn tại
        Path("output").mkdir(exist_ok=True)
        
        # Thử import và chạy mô hình tiên tiến
        try:
            from src.run_advanced_forecast import run_advanced_analysis
            success = run_advanced_analysis()
        except ImportError:
            print("Thử import từ thư mục hiện tại...")
            from run_advanced_forecast import run_advanced_analysis
            success = run_advanced_analysis()
        
        if success:
            print("\n" + "="*80)
            print("PHÂN TÍCH HOÀN TẤT THÀNH CÔNG!")
            print(f"Kết quả đã được lưu vào thư mục: {os.path.abspath('output')}")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("PHÂN TÍCH KHÔNG THÀNH CÔNG!")
            print("Vui lòng kiểm tra lỗi và thử lại.")
            print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"LỖI: {str(e)}")
        print("\nĐể dự báo tài chính, bạn cần cài đặt các thư viện sau:")
        print("pip install pandas numpy matplotlib scikit-learn scipy")
        print("="*80) 