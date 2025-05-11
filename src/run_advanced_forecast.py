"""
Chạy phân tích tài chính tiên tiến
=================================
Sử dụng kết hợp nhiều phương pháp học máy để dự báo chính xác cao nhất
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
import locale
import warnings
from scipy.signal import medfilt

# Bỏ qua cảnh báo
warnings.filterwarnings("ignore")

# Thử thiết lập locale
try:
    locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'vi_VN')
    except:
        print("Không thể thiết lập locale tiếng Việt")

# Import mô hình tiên tiến
try:
    from src.advanced_model import forecast_financial_indicator, AdvancedFinancialForecaster
except ImportError:
    try:
        from advanced_model import forecast_financial_indicator, AdvancedFinancialForecaster
    except ImportError:
        print("Không thể import mô hình tiên tiến. Đảm bảo file advanced_model.py tồn tại trong thư mục src.")
        exit(1)

# Định nghĩa đường dẫn
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Danh sách công ty
COMPANIES = {
    "VNM": "Vinamilk",
    "HPG": "Hòa Phát",
    "HAG": "Hoàng Anh Gia Lai",
    "FPT": "FPT",
    "MBB": "MB Bank"
}

# Chỉ số tài chính
FINANCIAL_INDICATORS = [
    "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)",
    "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ(20=10-11)",
    "Lợi nhuận thuần từ hoạt động kinh doanh{30=20+(21-22) + 24 - (25+26)}",
    "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)",
    "Vốn chủ sở hữu",
    "Tổng cộng tài sản"
]

# Tỷ số tài chính
FINANCIAL_RATIOS = [
    "ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu",
    "ROA - Tỷ suất lợi nhuận trên tổng tài sản",
    "Biên lợi nhuận gộp",
    "Biên lợi nhuận ròng",
    "Hệ số nợ",
    "Vòng quay tài sản"
]

# Hàm chuyển đổi chuỗi số tiền thành số float
def convert_to_float(value):
    if isinstance(value, str):
        # Loại bỏ dấu phẩy ngăn cách và khoảng trắng
        value = re.sub(r'[,\s]', '', value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# Hàm đọc và tiền xử lý dữ liệu
def load_data(file_path):
    print(f"Đang đọc dữ liệu từ {file_path}...")
    try:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        
        # Lấy mã công ty từ dòng đầu tiên
        company_code = df.columns[0]
        
        # Kiểm tra dữ liệu có bị cắt không bằng cách xem các giá trị cuối
        for col in df.columns[1:]:  # Bỏ qua cột đầu tiên là tên công ty
            for idx, val in df[col].items():
                if isinstance(val, str) and len(val) > 0:
                    if val[-1] not in '0123456789':
                        # Nếu ký tự cuối không phải số, có thể giá trị bị cắt
                        print(f"  Cảnh báo: Giá trị có thể bị cắt ở cột {col}, hàng {idx}: {val}")
                        # Thêm 0 để giá trị đầy đủ
                        if val[-1] == ',':
                            df.at[idx, col] = val + "000,000"
        
        # Thiết lập lại index
        df = df.set_index(df.columns[0])
        
        # Chuyển đổi tất cả các giá trị sang số
        for col in df.columns:
            df[col] = df[col].apply(convert_to_float)
        
        # Đặt tên cho các cột (quý và năm)
        dates = []
        for q in df.columns:
            parts = q.split(' - ')
            quarter = parts[0].replace('Quý ', '')
            year = parts[1]
            date_str = f"01-{quarter}-{year}"
            dates.append(date_str)
            
        df.columns = pd.to_datetime(dates, format="%d-%m-%Y")
        
        # Kiểm tra giá trị bất thường
        for idx in df.index:
            row_data = df.loc[idx]
            # Kiểm tra giá trị cực lớn (outliers)
            if row_data.max() > 1e15:
                print(f"  Cảnh báo: Phát hiện giá trị cực lớn (>1e15) ở chỉ số {idx}")
                # Thay thế giá trị cực lớn bằng giá trị trung bình của 2 kỳ gần nhất
                extreme_cols = row_data[row_data > 1e15].index
                for col in extreme_cols:
                    col_idx = df.columns.get_loc(col)
                    if col_idx > 0 and col_idx < len(df.columns) - 1:
                        # Lấy giá trị trung bình của cột trước và sau
                        avg_value = (row_data.iloc[col_idx-1] + row_data.iloc[col_idx+1]) / 2
                        df.at[idx, col] = avg_value
                    elif col_idx > 0:
                        # Lấy giá trị của cột trước
                        df.at[idx, col] = row_data.iloc[col_idx-1]
                    elif col_idx < len(df.columns) - 1:
                        # Lấy giá trị của cột sau
                        df.at[idx, col] = row_data.iloc[col_idx+1]
        
        return df, company_code
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {str(e)}")
        print(f"Chi tiết lỗi:", e)
        import traceback
        traceback.print_exc()
        return None, None

# Hàm tính toán các tỷ số tài chính
def calculate_financial_ratios(df):
    """
    Tính toán các tỷ số tài chính quan trọng từ dữ liệu gốc
    
    Args:
        df: DataFrame chứa dữ liệu tài chính
        
    Returns:
        DataFrame: DataFrame chứa các tỷ số tài chính đã tính
    """
    ratio_df = pd.DataFrame(index=FINANCIAL_RATIOS, columns=df.columns)
    
    for col in df.columns:
        # Lấy các giá trị cơ bản
        try:
            revenue = df.loc["Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)", col]
            gross_profit = df.loc["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ(20=10-11)", col]
            net_profit = df.loc["Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)", col]
            equity = df.loc["Vốn chủ sở hữu", col]
            total_assets = df.loc["Tổng cộng tài sản", col]
            
            # Tính ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu
            if equity != 0:
                ratio_df.loc["ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu", col] = net_profit / equity
            else:
                ratio_df.loc["ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu", col] = np.nan
            
            # Tính ROA - Tỷ suất lợi nhuận trên tổng tài sản
            if total_assets != 0:
                ratio_df.loc["ROA - Tỷ suất lợi nhuận trên tổng tài sản", col] = net_profit / total_assets
            else:
                ratio_df.loc["ROA - Tỷ suất lợi nhuận trên tổng tài sản", col] = np.nan
            
            # Tính Biên lợi nhuận gộp
            if revenue != 0:
                ratio_df.loc["Biên lợi nhuận gộp", col] = gross_profit / revenue
            else:
                ratio_df.loc["Biên lợi nhuận gộp", col] = np.nan
            
            # Tính Biên lợi nhuận ròng
            if revenue != 0:
                ratio_df.loc["Biên lợi nhuận ròng", col] = net_profit / revenue
            else:
                ratio_df.loc["Biên lợi nhuận ròng", col] = np.nan
            
            # Tính Hệ số nợ
            if total_assets != 0:
                ratio_df.loc["Hệ số nợ", col] = (total_assets - equity) / total_assets
            else:
                ratio_df.loc["Hệ số nợ", col] = np.nan
            
            # Tính Vòng quay tài sản
            if total_assets != 0:
                ratio_df.loc["Vòng quay tài sản", col] = revenue / total_assets
            else:
                ratio_df.loc["Vòng quay tài sản", col] = np.nan
                
        except (KeyError, TypeError):
            # Xử lý trường hợp thiếu dữ liệu
            print(f"  Không thể tính toán tỷ số tài chính cho cột {col} - thiếu dữ liệu")
    
    return ratio_df

# Hàm tạo báo cáo chi tiết
def generate_advanced_report(results):
    print("\nĐang tạo báo cáo phân tích chi tiết...")
    report_path = OUTPUT_DIR / "bao_cao_phan_tich_chi_tiet.txt"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO PHÂN TÍCH TÀI CHÍNH TIÊN TIẾN\n")
            f.write("=====================================\n\n")
            
            for company_code, data in results.items():
                company_name = COMPANIES[company_code]
                f.write(f"{company_name} ({company_code})\n")
                f.write("=" * 60 + "\n")
                
                # Phần I: Phân tích chỉ số tài chính cơ bản
                f.write("\nI. CHỈ SỐ TÀI CHÍNH CƠ BẢN\n")
                for indicator in FINANCIAL_INDICATORS:
                    if indicator in data and data[indicator]:
                        short_indicator = indicator.split("(")[0].strip()
                        f.write(f"{short_indicator}:\n")
                        f.write("-" * 40 + "\n")
                        
                        # Xác định mức độ tin cậy và thêm cảnh báo
                        confidence_warning = ""
                        if data[indicator]['confidence'] < 0.3:
                            confidence_warning = " (Độ tin cậy rất thấp - không nên sử dụng)"
                        elif data[indicator]['confidence'] < 0.5:
                            confidence_warning = " (Độ tin cậy thấp - cần thận trọng)"
                        
                        # Kiểm tra mức độ thay đổi bất thường
                        change_warning = ""
                        if abs(data[indicator]['change_rate']) > 0.8:
                            change_warning = " (Cảnh báo: Thay đổi quá lớn)"
                        
                        f.write(f"  ※ Xu hướng: {data[indicator]['trend']}{confidence_warning}\n")
                        f.write(f"  ※ Tỷ lệ thay đổi: {data[indicator]['change_rate']*100:.2f}%{change_warning}\n")
                        f.write(f"  ※ Độ tin cậy dự báo: {data[indicator]['confidence']*100:.2f}%\n")
                        f.write(f"  ※ Giá trị cuối cùng (Q4-2024): {data[indicator]['last_value']:,.0f} VND\n")
                        f.write(f"  ※ Giá trị dự báo (Q4-2028): {data[indicator]['forecast_value']:,.0f} VND\n")
                        
                        # Giá trị dự báo chi tiết
                        f.write("\n  Chi tiết dự báo theo quý (2025-2028):\n")
                        future_quarters = ["Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025", 
                                           "Q1-2026", "Q2-2026", "Q3-2026", "Q4-2026",
                                           "Q1-2027", "Q2-2027", "Q3-2027", "Q4-2027",
                                           "Q1-2028", "Q2-2028", "Q3-2028", "Q4-2028"]
                        
                        # Format theo hàng để dễ đọc
                        for i in range(0, len(future_quarters), 4):
                            quarters = future_quarters[i:i+4]
                            values = data[indicator]['forecast'][i:i+4]
                            
                            # Hiển thị 4 quý trên một dòng
                            row = "  "
                            for q, v in zip(quarters, values):
                                row += f"{q}: {v:,.0f} VND   "
                            f.write(row + "\n")
                        
                        f.write("\n")
                
                # Phần II: Phân tích tỷ số tài chính
                f.write("\nII. TỶ SỐ TÀI CHÍNH\n")
                for ratio in FINANCIAL_RATIOS:
                    if ratio in data and data[ratio]:
                        f.write(f"{ratio}:\n")
                        f.write("-" * 40 + "\n")
                        
                        # Xác định mức độ tin cậy và thêm cảnh báo
                        confidence_warning = ""
                        if data[ratio]['confidence'] < 0.3:
                            confidence_warning = " (Độ tin cậy rất thấp - không nên sử dụng)"
                        elif data[ratio]['confidence'] < 0.5:
                            confidence_warning = " (Độ tin cậy thấp - cần thận trọng)"
                        
                        # Kiểm tra mức độ thay đổi bất thường
                        change_warning = ""
                        if abs(data[ratio]['change_rate']) > 0.8:
                            change_warning = " (Cảnh báo: Thay đổi quá lớn)"
                        
                        f.write(f"  ※ Xu hướng: {data[ratio]['trend']}{confidence_warning}\n")
                        f.write(f"  ※ Tỷ lệ thay đổi: {data[ratio]['change_rate']*100:.2f}%{change_warning}\n")
                        f.write(f"  ※ Độ tin cậy dự báo: {data[ratio]['confidence']*100:.2f}%\n")
                        
                        # Hiển thị giá trị tỷ số dưới dạng phần trăm hoặc số lần
                        if "Biên lợi nhuận" in ratio or "ROE" in ratio or "ROA" in ratio or "Hệ số nợ" in ratio:
                            f.write(f"  ※ Giá trị cuối cùng (Q4-2024): {data[ratio]['last_value']*100:.2f}%\n")
                            f.write(f"  ※ Giá trị dự báo (Q4-2028): {data[ratio]['forecast_value']*100:.2f}%\n")
                        else:
                            f.write(f"  ※ Giá trị cuối cùng (Q4-2024): {data[ratio]['last_value']:.2f} lần\n")
                            f.write(f"  ※ Giá trị dự báo (Q4-2028): {data[ratio]['forecast_value']:.2f} lần\n")
                        
                        # Giá trị dự báo chi tiết
                        f.write("\n  Chi tiết dự báo theo quý (2025-2028):\n")
                        future_quarters = ["Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025", 
                                           "Q1-2026", "Q2-2026", "Q3-2026", "Q4-2026",
                                           "Q1-2027", "Q2-2027", "Q3-2027", "Q4-2027",
                                           "Q1-2028", "Q2-2028", "Q3-2028", "Q4-2028"]
                        
                        # Format theo hàng để dễ đọc
                        for i in range(0, len(future_quarters), 4):
                            quarters = future_quarters[i:i+4]
                            values = data[ratio]['forecast'][i:i+4]
                            
                            # Hiển thị 4 quý trên một dòng
                            row = "  "
                            for q, v in zip(quarters, values):
                                if "Biên lợi nhuận" in ratio or "ROE" in ratio or "ROA" in ratio or "Hệ số nợ" in ratio:
                                    row += f"{q}: {v*100:.2f}%   "
                                else:
                                    row += f"{q}: {v:.2f} lần   "
                            f.write(row + "\n")
                        
                        f.write("\n")
                    
                f.write("\n\n")
            
            # Thêm phần phân tích tổng hợp
            f.write("PHÂN TÍCH TỔNG HỢP\n")
            f.write("=================\n\n")
            
            # Phân tích doanh thu
            f.write("● Tăng trưởng doanh thu (2024-2028):\n")
            revenue_growths = []
            for company_code, data in results.items():
                indicator = "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)"
                if indicator in data and data[indicator]:
                    confidence = data[indicator]['confidence']
                    rate = data[indicator]['change_rate'] * 100
                    revenue_growths.append((company_code, rate, confidence))
            
            # Sắp xếp theo tỷ lệ tăng trưởng
            revenue_growths.sort(key=lambda x: x[1], reverse=True)
            for code, rate, confidence in revenue_growths:
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                f.write(f"  {COMPANIES[code]} ({code}): {rate:.2f}%{confidence_note}\n")
            
            # Phân tích lợi nhuận
            f.write("\n● Tăng trưởng lợi nhuận sau thuế (2024-2028):\n")
            profit_growths = []
            for company_code, data in results.items():
                indicator = "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)"
                if indicator in data and data[indicator]:
                    confidence = data[indicator]['confidence']
                    rate = data[indicator]['change_rate'] * 100
                    profit_growths.append((company_code, rate, confidence))
            
            # Sắp xếp theo tỷ lệ tăng trưởng
            profit_growths.sort(key=lambda x: x[1], reverse=True)
            for code, rate, confidence in profit_growths:
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                f.write(f"  {COMPANIES[code]} ({code}): {rate:.2f}%{confidence_note}\n")
            
            # Khuyến nghị đầu tư
            f.write("\n● Khuyến nghị đầu tư:\n")
            recommendations = []
            
            for company_code, data in results.items():
                revenue_ind = "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)"
                profit_ind = "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)"
                
                if revenue_ind in data and profit_ind in data:
                    rev_info = data[revenue_ind]
                    prof_info = data[profit_ind]
                    
                    if rev_info and prof_info:
                        # Kiểm tra độ tin cậy
                        rev_confidence = rev_info['confidence']
                        prof_confidence = prof_info['confidence']
                        
                        # Chỉ xem xét kết quả có độ tin cậy đủ cao
                        if rev_confidence >= 0.3 and prof_confidence >= 0.3:
                            # Tính điểm đánh giá dựa trên tăng trưởng và độ tin cậy
                            rev_change = rev_info['change_rate']
                            prof_change = prof_info['change_rate']
                            
                            # Trọng số cao hơn cho lợi nhuận và có tính đến độ tin cậy
                            rev_score = rev_change * rev_confidence
                            prof_score = prof_change * prof_confidence * 1.5
                            
                            # Nếu dự báo âm, điều chỉnh điểm số
                            if prof_info['forecast_value'] < 0:
                                prof_score = -0.5  # Điểm tiêu cực nếu dự báo lợi nhuận âm
                            elif prof_info['forecast_value'] < prof_info['last_value'] * 0.3:
                                # Nếu lợi nhuận giảm hơn 70%, giảm điểm
                                prof_score *= 0.5
                            
                            # Điều chỉnh thêm điểm cho các dự báo bất thường
                            if abs(rev_change) > 0.7 or abs(prof_change) > 0.7:
                                # Giảm điểm cho những dự báo biến động quá mạnh
                                adjustment = 1 - min(1, (abs(rev_change) + abs(prof_change)) / 4)
                                rev_score *= adjustment
                                prof_score *= adjustment
                            
                            total_score = rev_score + prof_score
                            
                            # Thêm thông tin về tính ổn định
                            stability = 0
                            if abs(rev_change) < 0.1:  # Doanh thu ổn định
                                stability += 0.5
                            if abs(prof_change) < 0.1:  # Lợi nhuận ổn định
                                stability += 0.5
                            
                            average_confidence = (rev_confidence + prof_confidence) / 2
                            
                            recommendations.append((company_code, total_score, stability, average_confidence))
                        else:
                            # Đối với công ty có độ tin cậy thấp, thêm với điểm thấp
                            recommendations.append((company_code, -1, 0, (rev_confidence + prof_confidence) / 2))
                    else:
                        # Không đủ dữ liệu
                        recommendations.append((company_code, -2, 0, 0))
            
            # Sắp xếp theo điểm tổng hợp
            recommendations.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            # Định nghĩa lại rating map với điều kiện chặt chẽ hơn
            def get_rating(score, stability, confidence):
                if confidence < 0.3:
                    return "Không đủ dữ liệu đáng tin cậy"
                
                if score < -0.2:
                    return "Không khuyến nghị"
                elif score < 0:
                    return "Nên theo dõi"
                elif score < 0.05:
                    if stability > 0.7:
                        return "Trung lập (ổn định)"
                    else:
                        return "Trung lập"
                elif score < 0.1:
                    return "Khả quan"
                elif score < 0.2:
                    return "Tích cực"
                else:
                    return "Rất tích cực"
            
            for i, (code, score, stability, confidence) in enumerate(recommendations):
                rating = get_rating(score, stability, confidence)
                
                rev_ind = "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)"
                prof_ind = "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)"
                
                # Thêm chỉ báo độ tin cậy
                confidence_indicator = ""
                if confidence < 0.3:
                    confidence_indicator = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_indicator = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_indicator = " (độ tin cậy cao)"
                
                rev_trend = results[code][rev_ind]['trend'] if rev_ind in results[code] else "N/A"
                prof_trend = results[code][prof_ind]['trend'] if prof_ind in results[code] else "N/A"
                
                f.write(f"  {i+1}. {COMPANIES[code]} ({code}): {rating}{confidence_indicator}\n")
                
                # Thêm chi tiết về dự báo
                if rev_ind in results[code] and prof_ind in results[code]:
                    rev_change = results[code][rev_ind]['change_rate'] * 100
                    prof_change = results[code][prof_ind]['change_rate'] * 100
                    f.write(f"     → Doanh thu: {rev_trend} ({rev_change:.1f}%), Lợi nhuận: {prof_trend} ({prof_change:.1f}%)\n")
                else:
                    f.write(f"     → Doanh thu: {rev_trend}, Lợi nhuận: {prof_trend}\n")
            
        print(f"Đã tạo báo cáo tại: {report_path}")
        return report_path
    except Exception as e:
        print(f"Lỗi khi tạo báo cáo: {str(e)}")
        return None

# Hàm tạo báo cáo tóm tắt
def generate_summary_report(results):
    print("\nĐang tạo báo cáo tóm tắt...")
    report_path = OUTPUT_DIR / "bao_cao_tom_tat.txt"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO TÓM TẮT PHÂN TÍCH TÀI CHÍNH\n")
            f.write("==================================\n\n")
            
            # Tổng kết về tăng trưởng
            f.write("● Tăng trưởng doanh thu (2024-2028):\n")
            revenue_growths = []
            for company_code, data in results.items():
                indicator = "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)"
                if indicator in data and data[indicator]:
                    confidence = data[indicator]['confidence']
                    rate = data[indicator]['change_rate'] * 100
                    revenue_growths.append((company_code, rate, confidence))
            
            # Sắp xếp theo tỷ lệ tăng trưởng
            revenue_growths.sort(key=lambda x: x[1], reverse=True)
            for code, rate, confidence in revenue_growths:
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                f.write(f"  {COMPANIES[code]} ({code}): {rate:.2f}%{confidence_note}\n")
            
            # Phân tích lợi nhuận
            f.write("\n● Tăng trưởng lợi nhuận sau thuế (2024-2028):\n")
            profit_growths = []
            for company_code, data in results.items():
                indicator = "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)"
                if indicator in data and data[indicator]:
                    confidence = data[indicator]['confidence']
                    rate = data[indicator]['change_rate'] * 100
                    profit_growths.append((company_code, rate, confidence))
            
            # Sắp xếp theo tỷ lệ tăng trưởng
            profit_growths.sort(key=lambda x: x[1], reverse=True)
            for code, rate, confidence in profit_growths:
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                f.write(f"  {COMPANIES[code]} ({code}): {rate:.2f}%{confidence_note}\n")
            
            # Phân tích ROE
            f.write("\n● ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu (2024-2028):\n")
            roe_data = []
            for company_code, data in results.items():
                ratio = "ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu"
                if ratio in data and data[ratio]:
                    confidence = data[ratio]['confidence']
                    current_value = data[ratio]['last_value'] * 100  # Chuyển thành phần trăm
                    future_value = data[ratio]['forecast_value'] * 100  # Chuyển thành phần trăm
                    roe_data.append((company_code, current_value, future_value, confidence))
            
            # Sắp xếp theo giá trị ROE tương lai
            roe_data.sort(key=lambda x: x[2], reverse=True)
            for code, current, future, confidence in roe_data:
                trend_symbol = "→"
                if future > current * 1.1:
                    trend_symbol = "↑"
                elif future < current * 0.9:
                    trend_symbol = "↓"
                
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                
                f.write(f"  {COMPANIES[code]} ({code}): {current:.2f}% {trend_symbol} {future:.2f}%{confidence_note}\n")
            
            # Phân tích ROA
            f.write("\n● ROA - Tỷ suất lợi nhuận trên tổng tài sản (2024-2028):\n")
            roa_data = []
            for company_code, data in results.items():
                ratio = "ROA - Tỷ suất lợi nhuận trên tổng tài sản"
                if ratio in data and data[ratio]:
                    confidence = data[ratio]['confidence']
                    current_value = data[ratio]['last_value'] * 100  # Chuyển thành phần trăm
                    future_value = data[ratio]['forecast_value'] * 100  # Chuyển thành phần trăm
                    roa_data.append((company_code, current_value, future_value, confidence))
            
            # Sắp xếp theo giá trị ROA tương lai
            roa_data.sort(key=lambda x: x[2], reverse=True)
            for code, current, future, confidence in roa_data:
                trend_symbol = "→"
                if future > current * 1.1:
                    trend_symbol = "↑"
                elif future < current * 0.9:
                    trend_symbol = "↓"
                
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                
                f.write(f"  {COMPANIES[code]} ({code}): {current:.2f}% {trend_symbol} {future:.2f}%{confidence_note}\n")
            
            # Phân tích biên lợi nhuận
            f.write("\n● Biên lợi nhuận ròng (2024-2028):\n")
            margin_data = []
            for company_code, data in results.items():
                ratio = "Biên lợi nhuận ròng"
                if ratio in data and data[ratio]:
                    confidence = data[ratio]['confidence']
                    current_value = data[ratio]['last_value'] * 100  # Chuyển thành phần trăm
                    future_value = data[ratio]['forecast_value'] * 100  # Chuyển thành phần trăm
                    margin_data.append((company_code, current_value, future_value, confidence))
            
            # Sắp xếp theo giá trị biên lợi nhuận tương lai
            margin_data.sort(key=lambda x: x[2], reverse=True)
            for code, current, future, confidence in margin_data:
                trend_symbol = "→"
                if future > current * 1.1:
                    trend_symbol = "↑"
                elif future < current * 0.9:
                    trend_symbol = "↓"
                
                confidence_note = ""
                if confidence < 0.3:
                    confidence_note = " (độ tin cậy rất thấp)"
                elif confidence < 0.5:
                    confidence_note = " (độ tin cậy thấp)"
                elif confidence > 0.8:
                    confidence_note = " (độ tin cậy cao)"
                
                f.write(f"  {COMPANIES[code]} ({code}): {current:.2f}% {trend_symbol} {future:.2f}%{confidence_note}\n")
            
            # Khuyến nghị đầu tư dựa trên phân tích tổng hợp
            f.write("\n● Khuyến nghị đầu tư:\n")
            recommendations = []
            
            for company_code, data in results.items():
                # Chỉ số cơ bản
                revenue_ind = "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)"
                profit_ind = "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)"
                
                # Tỷ số tài chính
                roe_ratio = "ROE - Tỷ suất lợi nhuận trên vốn chủ sở hữu"
                roa_ratio = "ROA - Tỷ suất lợi nhuận trên tổng tài sản"
                net_margin_ratio = "Biên lợi nhuận ròng"
                
                if revenue_ind in data and profit_ind in data:
                    # Tính điểm tăng trưởng cơ bản
                    rev_info = data[revenue_ind]
                    prof_info = data[profit_ind]
                    
                    # Trọng số chính cho các chỉ số cơ bản (40%)
                    basic_score = 0
                    basic_weight = 0
                    
                    if rev_info and rev_info['confidence'] >= 0.3:
                        rev_change = rev_info['change_rate']
                        rev_score = rev_change * rev_info['confidence'] * 20
                        basic_score += rev_score
                        basic_weight += 1
                    
                    if prof_info and prof_info['confidence'] >= 0.3:
                        prof_change = prof_info['change_rate']
                        prof_score = prof_change * prof_info['confidence'] * 30
                        basic_score += prof_score
                        basic_weight += 1.5
                    
                    # Kiểm tra tỷ số tài chính (60%)
                    ratio_score = 0
                    ratio_weight = 0
                    
                    # Tính điểm từ ROE (25%)
                    if roe_ratio in data and data[roe_ratio] and data[roe_ratio]['confidence'] >= 0.3:
                        roe_value = data[roe_ratio]['forecast_value']
                        # Thang điểm ROE: <5%: kém, 5-10%: trung bình, 10-15%: tốt, >15%: xuất sắc
                        if roe_value < 0.05:
                            roe_score = 0
                        elif roe_value < 0.1:
                            roe_score = roe_value * 100
                        elif roe_value < 0.15:
                            roe_score = roe_value * 150
                        else:
                            roe_score = roe_value * 200
                        
                        ratio_score += roe_score * data[roe_ratio]['confidence']
                        ratio_weight += 2.5
                    
                    # Tính điểm từ ROA (15%)
                    if roa_ratio in data and data[roa_ratio] and data[roa_ratio]['confidence'] >= 0.3:
                        roa_value = data[roa_ratio]['forecast_value']
                        # Thang điểm ROA: <2%: kém, 2-5%: trung bình, 5-10%: tốt, >10%: xuất sắc
                        if roa_value < 0.02:
                            roa_score = 0
                        elif roa_value < 0.05:
                            roa_score = roa_value * 150
                        elif roa_value < 0.1:
                            roa_score = roa_value * 200
                        else:
                            roa_score = roa_value * 250
                        
                        ratio_score += roa_score * data[roa_ratio]['confidence']
                        ratio_weight += 1.5
                    
                    # Tính điểm từ biên lợi nhuận (20%)
                    if net_margin_ratio in data and data[net_margin_ratio] and data[net_margin_ratio]['confidence'] >= 0.3:
                        margin_value = data[net_margin_ratio]['forecast_value']
                        # Thang điểm biên lợi nhuận: <5%: kém, 5-10%: trung bình, 10-20%: tốt, >20%: xuất sắc
                        if margin_value < 0.05:
                            margin_score = margin_value * 50
                        elif margin_value < 0.1:
                            margin_score = margin_value * 100
                        elif margin_value < 0.2:
                            margin_score = margin_value * 150
                        else:
                            margin_score = margin_value * 200
                        
                        ratio_score += margin_score * data[net_margin_ratio]['confidence']
                        ratio_weight += 2
                    
                    # Tính điểm tổng hợp
                    total_score = 0
                    if basic_weight > 0:
                        basic_score = basic_score / basic_weight
                        total_score += basic_score * 0.4  # 40% trọng số cho chỉ số cơ bản
                    
                    if ratio_weight > 0:
                        ratio_score = ratio_score / ratio_weight
                        total_score += ratio_score * 0.6  # 60% trọng số cho tỷ số tài chính
                    
                    # Phân loại xếp hạng dựa trên điểm tổng thể
                    rating = get_rating(total_score)
                    
                    # Thêm vào danh sách khuyến nghị
                    recommendations.append((company_code, total_score, rating))
            
            # Sắp xếp theo điểm tổng thể
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for code, score, rating in recommendations:
                f.write(f"  {COMPANIES[code]} ({code}): {rating} (điểm: {score:.2f})\n")
            
            f.write("\n\nChú thích:\n")
            f.write("- Mua mạnh: Cổ phiếu có tiềm năng tăng trưởng mạnh về doanh thu và lợi nhuận, có tỷ số tài chính tốt\n")
            f.write("- Mua: Cổ phiếu có triển vọng tích cực, tỷ số tài chính ổn định và đang cải thiện\n")
            f.write("- Nắm giữ: Cổ phiếu có triển vọng ổn định, không có biến động lớn\n")
            f.write("- Bán: Cổ phiếu có dấu hiệu suy giảm về lợi nhuận và các tỷ số tài chính\n")
            f.write("- Bán mạnh: Cổ phiếu có nhiều dấu hiệu xấu, khả năng phục hồi thấp\n")
            
    except Exception as e:
        print(f"Lỗi khi tạo báo cáo tóm tắt: {str(e)}")

def get_rating(score):
    """
    Chuyển đổi điểm số thành khuyến nghị đầu tư
    
    Args:
        score: Điểm tổng hợp các chỉ số
        
    Returns:
        str: Khuyến nghị đầu tư
    """
    if score > 15:
        return "Mua mạnh"
    elif score > 10:
        return "Mua"
    elif score > 5:
        return "Nắm giữ"
    elif score > 0:
        return "Bán"
    else:
        return "Bán mạnh"

# Hàm chính để chạy phân tích tiên tiến
def run_advanced_analysis():
    """
    Hàm chính để thực hiện phân tích tài chính tiên tiến
    """
    print("\n=== BẮT ĐẦU PHÂN TÍCH TÀI CHÍNH TIÊN TIẾN ===\n")
    
    results = {}
    
    # Phân tích từng công ty
    for company_code, company_name in COMPANIES.items():
        print(f"\nĐang phân tích dữ liệu cho {company_name} ({company_code})...")
        
        file_path = DATA_DIR / f"{company_code}.csv"
        
        if not file_path.exists():
            print(f"  Không tìm thấy dữ liệu cho {company_name}")
            continue
        
        # Đọc dữ liệu
        df, _ = load_data(file_path)
        
        if df is None:
            print(f"  Lỗi khi đọc dữ liệu cho {company_name}")
            continue
        
        # Tính toán các tỷ số tài chính
        ratio_df = calculate_financial_ratios(df)
        
        # Kết hợp dữ liệu chính và tỷ số tài chính
        combined_df = pd.concat([df, ratio_df])
        
        # Khởi tạo kết quả cho công ty
        company_results = {}
        
        # Phân tích từng chỉ số tài chính cơ bản
        for indicator in FINANCIAL_INDICATORS:
            if indicator in df.index:
                print(f"  Đang phân tích {indicator}...")
                
                # Lấy dữ liệu
                series = pd.Series(df.loc[indicator])
                
                # Dự báo
                result = forecast_financial_indicator(df, indicator, company_code)
                
                if result:
                    company_results[indicator] = result
        
        # Phân tích từng tỷ số tài chính
        for ratio in FINANCIAL_RATIOS:
            if ratio in ratio_df.index:
                print(f"  Đang phân tích {ratio}...")
                
                # Thêm dữ liệu tỷ số tài chính vào DataFrame chính để có thể sử dụng hàm dự báo hiện có
                df_with_ratio = df.copy()
                df_with_ratio.loc[ratio] = ratio_df.loc[ratio]
                
                # Dự báo
                result = forecast_financial_indicator(df_with_ratio, ratio, company_code)
                
                if result:
                    company_results[ratio] = result
        
        # Lưu kết quả của công ty
        results[company_code] = company_results
    
    # Tạo báo cáo
    generate_advanced_report(results)
    generate_summary_report(results)
    
    print("\n=== HOÀN TẤT PHÂN TÍCH TÀI CHÍNH TIÊN TIẾN ===\n")
    print(f"Báo cáo chi tiết đã được lưu tại: {OUTPUT_DIR / 'bao_cao_phan_tich_chi_tiet.txt'}")
    print(f"Báo cáo tóm tắt đã được lưu tại: {OUTPUT_DIR / 'bao_cao_tom_tat.txt'}")
    
    return results

if __name__ == "__main__":
    run_advanced_analysis() 
