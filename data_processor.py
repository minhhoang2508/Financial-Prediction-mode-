import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    from statsmodels.tsa.arima_model import ARIMA
import os
import locale
import re
from pathlib import Path

# Đặt ngôn ngữ hiển thị là Tiếng Việt
try:
    locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'vi_VN')
    except:
        print("Không thể đặt locale tiếng Việt, sử dụng locale mặc định")

# Định nghĩa đường dẫn dữ liệu
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Định nghĩa các chỉ số tài chính cần phân tích
FINANCIAL_INDICATORS = [
    "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)",
    "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ(20=10-11)",
    "Lợi nhuận thuần từ hoạt động kinh doanh{30=20+(21-22) + 24 - (25+26)}",
    "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)",
    "Vốn chủ sở hữu",
    "Tổng cộng tài sản"
]

# Danh sách công ty
COMPANIES = {
    "VNM": "Vinamilk",
    "HPG": "Hòa Phát",
    "HAG": "Hoàng Anh Gia Lai",
    "FPT": "FPT",
    "MBB": "MB Bank"
}

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
def load_and_preprocess_data(file_path):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    
    # Lấy mã công ty từ dòng đầu tiên
    company_code = df.columns[0]
    
    # Thiết lập lại index
    df = df.set_index(df.columns[0])
    
    # Chuyển đổi tất cả các giá trị sang số
    for col in df.columns:
        df[col] = df[col].apply(convert_to_float)
    
    # Đặt tên cho các cột (quý và năm)
    df.columns = pd.to_datetime([f"01-{q.split(' - ')[0].replace('Quý ', '')}-{q.split(' - ')[1]}" for q in df.columns], format="%d-%m-%Y")
    
    return df, company_code

# Hàm vẽ biểu đồ xu hướng
def plot_trend(df, company_code, company_name, indicator):
    plt.figure(figsize=(12, 6))
    
    # Vẽ dữ liệu lịch sử
    plt.plot(df.columns, df.loc[indicator], marker='o', label='Dữ liệu lịch sử')
    
    # Định dạng trục x
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Đặt tên cho biểu đồ
    plt.title(f"Xu hướng {indicator} - {company_name} ({company_code})")
    plt.ylabel('Giá trị (VND)')
    plt.xlabel('Thời gian')
    
    # Lưu biểu đồ
    chart_path = OUTPUT_DIR / f"{company_code}_{indicator.split(' ')[0]}_trend.png"
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Hàm dự báo giá trị tương lai bằng ARIMA hoặc LinearRegression
def forecast_arima(series, company_code, indicator_name, periods=16):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    values = series.values.reshape(-1, 1)
    
    # Kiểm tra giá trị NaN
    if np.isnan(values).any():
        # Xử lý giá trị NaN bằng cách thay thế bằng giá trị trung bình
        mean_value = np.nanmean(values)
        values = np.nan_to_num(values, nan=mean_value)
    
    scaled_data = scaler.fit_transform(values).flatten()
    
    try:
        # Thử sử dụng ARIMA
        model = ARIMA(scaled_data, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Dự báo
        forecast = model_fit.forecast(steps=periods)
        
        # Chuyển về giá trị gốc
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        
    except Exception as e:
        print(f"Không thể sử dụng ARIMA cho {company_code} - {indicator_name}: {str(e)}")
        print("Chuyển sang sử dụng Linear Regression")
        
        # Sử dụng Linear Regression
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        # Xử lý giá trị NaN trong y
        if np.isnan(y).any():
            mean_value = np.nanmean(y)
            y = np.nan_to_num(y, nan=mean_value)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Dự báo
        X_future = np.arange(len(series), len(series) + periods).reshape(-1, 1)
        forecast = model.predict(X_future)
    
    return forecast

# Hàm đánh giá xu hướng
def evaluate_trend(historical_data, predicted_data):
    # Tính tỷ lệ thay đổi trung bình
    last_historical = historical_data.iloc[-1]
    last_predicted = predicted_data[-1]
    
    change_rate = (last_predicted - last_historical) / last_historical
    
    # Đánh giá xu hướng
    if change_rate > 0.05:
        return "Tăng", change_rate
    elif change_rate < -0.05:
        return "Giảm", change_rate
    else:
        return "Ổn định", change_rate

# Hàm xử lý và phân tích dữ liệu cho mỗi công ty
def process_company_data(file_path, company_code, company_name):
    # Đọc và tiền xử lý dữ liệu
    df, _ = load_and_preprocess_data(file_path)
    
    # Tạo DataFrame cho kết quả dự báo
    forecast_results = pd.DataFrame()
    
    # Dự báo cho các quý từ 2025-2028
    future_quarters = pd.date_range(start='2025-01-01', end='2028-12-31', freq='Q')
    
    # Phân tích từng chỉ số
    analysis_results = {}
    
    for indicator in FINANCIAL_INDICATORS:
        if indicator in df.index:
            # Vẽ biểu đồ xu hướng
            chart_path = plot_trend(df, company_code, company_name, indicator)
            
            # Dự báo giá trị tương lai
            forecast = forecast_arima(pd.Series(df.loc[indicator]), company_code, indicator)
            
            # Đánh giá xu hướng
            trend, change_rate = evaluate_trend(pd.Series(df.loc[indicator]), forecast)
            
            # Lưu kết quả
            analysis_results[indicator] = {
                "chart_path": str(chart_path),
                "trend": trend,
                "change_rate": change_rate,
                "last_value": df.loc[indicator].iloc[-1],
                "predicted_value": forecast[-1]
            }
            
            # Thêm vào DataFrame dự báo
            forecast_series = pd.Series(forecast, index=future_quarters)
            forecast_results[indicator] = forecast_series
    
    # Lưu kết quả dự báo
    forecast_file = OUTPUT_DIR / f"{company_code}_forecast.csv"
    forecast_results.to_csv(forecast_file)
    
    return analysis_results

# Hàm chính để thực hiện toàn bộ phân tích
def main():
    all_results = {}
    
    for company_code, company_name in COMPANIES.items():
        print(f"Đang xử lý dữ liệu cho {company_name} ({company_code})...")
        file_path = DATA_DIR / f"{company_code}.csv"
        
        if file_path.exists():
            results = process_company_data(file_path, company_code, company_name)
            all_results[company_code] = {
                "name": company_name,
                "results": results
            }
        else:
            print(f"Không tìm thấy dữ liệu cho {company_name}")
    
    # Tạo báo cáo tổng hợp
    generate_report(all_results)
    
    print("Hoàn tất phân tích!")

# Hàm tạo báo cáo tổng hợp
def generate_report(all_results):
    report_path = OUTPUT_DIR / "financial_analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO PHÂN TÍCH TÀI CHÍNH\n")
        f.write("===========================\n\n")
        
        for company_code, data in all_results.items():
            company_name = data["name"]
            results = data["results"]
            
            f.write(f"{company_name} ({company_code})\n")
            f.write("-" * 50 + "\n")
            
            for indicator, info in results.items():
                short_indicator = indicator.split("(")[0].strip()
                f.write(f"{short_indicator}:\n")
                f.write(f"  - Xu hướng: {info['trend']}\n")
                f.write(f"  - Tỷ lệ thay đổi: {info['change_rate']*100:.2f}%\n")
                f.write(f"  - Giá trị cuối cùng: {info['last_value']:,.0f} VND\n")
                f.write(f"  - Giá trị dự báo (Q4-2028): {info['predicted_value']:,.0f} VND\n")
                f.write("\n")
            
            f.write("\n")
    
    print(f"Đã tạo báo cáo tại: {report_path}")

if __name__ == "__main__":
    main() 