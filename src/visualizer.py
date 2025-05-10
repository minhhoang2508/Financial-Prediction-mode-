import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import locale
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Đặt ngôn ngữ hiển thị là Tiếng Việt
locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')

# Định nghĩa đường dẫn
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Định nghĩa các màu cho biểu đồ
COLORS = {
    "VNM": "#F8766D",
    "HPG": "#00BA38",
    "HAG": "#619CFF",
    "FPT": "#F564E3",
    "MBB": "#FF9E4A"
}

# Hàm định dạng số tiền
def format_money(x, pos):
    return f'{x/1e9:.0f} tỷ'

# Hàm vẽ biểu đồ so sánh nhiều công ty
def plot_comparison(data_dict, indicator, title):
    plt.figure(figsize=(14, 8))
    
    # Định dạng tiền tệ
    formatter = FuncFormatter(format_money)
    
    # Vẽ dữ liệu cho mỗi công ty
    for company_code, data in data_dict.items():
        if indicator in data.index:
            plt.plot(data.columns, data.loc[indicator], marker='o', 
                     label=f"{company_code}", color=COLORS.get(company_code, None), 
                     linewidth=2)
    
    # Định dạng trục x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Định dạng trục y
    plt.gca().yaxis.set_major_formatter(formatter)
    
    # Đặt tên cho biểu đồ
    plt.title(f"So sánh {title} giữa các công ty")
    plt.ylabel('Giá trị (tỷ đồng)')
    plt.xlabel('Thời gian')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm legend
    plt.legend()
    
    # Lưu biểu đồ
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"comparison_{indicator.split(' ')[0]}.png"
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Hàm vẽ biểu đồ dự báo
def plot_forecast(historical_data, forecast_data, company_code, indicator, company_name):
    plt.figure(figsize=(14, 8))
    
    # Định dạng tiền tệ
    formatter = FuncFormatter(format_money)
    
    # Vẽ dữ liệu lịch sử
    plt.plot(historical_data.index, historical_data.values, 
             marker='o', label='Dữ liệu lịch sử', color='blue', linewidth=2)
    
    # Vẽ dữ liệu dự báo
    plt.plot(forecast_data.index, forecast_data.values, 
             marker='s', label='Dự báo', color='red', linestyle='--', linewidth=2)
    
    # Thêm vùng dự báo
    plt.axvspan(forecast_data.index[0], forecast_data.index[-1], 
                alpha=0.2, color='gray', label='Khoảng dự báo')
    
    # Định dạng trục x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Định dạng trục y
    plt.gca().yaxis.set_major_formatter(formatter)
    
    # Đặt tên cho biểu đồ
    plt.title(f"Dự báo {indicator} - {company_name} ({company_code})")
    plt.ylabel('Giá trị (tỷ đồng)')
    plt.xlabel('Thời gian')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm legend
    plt.legend()
    
    # Lưu biểu đồ
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"{company_code}_{indicator.split(' ')[0]}_forecast.png"
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Hàm vẽ biểu đồ radar cho đánh giá tổng thể
def plot_radar_chart(data_dict, companies):
    # Các chỉ số cần đánh giá
    indicators = [
        "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)",
        "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)",
        "Vốn chủ sở hữu",
    ]
    
    # Tạo nhãn ngắn gọn
    labels = ["Doanh thu", "Lợi nhuận", "Vốn CSH"]
    
    # Số lượng chỉ số
    N = len(labels)
    
    # Vị trí góc cho mỗi chỉ số
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Khép kín đồ thị
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Chuẩn hóa dữ liệu cho mỗi chỉ số
    max_values = {}
    for indicator in indicators:
        max_values[indicator] = max([data.loc[indicator].max() if indicator in data.index else 0 
                               for company_code, data in data_dict.items()])
    
    # Vẽ cho từng công ty
    for company_code, company_name in companies.items():
        if company_code in data_dict:
            data = data_dict[company_code]
            
            # Lấy giá trị mới nhất cho mỗi chỉ số và chuẩn hóa
            values = []
            for indicator in indicators:
                if indicator in data.index:
                    value = data.loc[indicator].iloc[-1] / max_values[indicator]
                else:
                    value = 0
                values.append(value)
            
            # Khép kín đồ thị
            values += values[:1]
            
            # Vẽ
            ax.plot(angles, values, linewidth=2, label=company_code, color=COLORS.get(company_code, None))
            ax.fill(angles, values, alpha=0.1, color=COLORS.get(company_code, None))
    
    # Thiết lập trục
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Thêm legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Tiêu đề
    plt.title("So sánh chỉ số tài chính các công ty", size=15, y=1.1)
    
    # Lưu biểu đồ
    chart_path = OUTPUT_DIR / "radar_comparison.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Hàm vẽ biểu đồ tăng trưởng dự báo
def plot_growth_forecast(all_results, companies):
    plt.figure(figsize=(12, 8))
    
    # Chuẩn bị dữ liệu
    company_codes = []
    revenue_growth = []
    profit_growth = []
    
    for company_code, data in all_results.items():
        results = data["results"]
        
        # Chỉ số doanh thu và lợi nhuận
        revenue_indicator = "Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)"
        profit_indicator = "Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)"
        
        if revenue_indicator in results and profit_indicator in results:
            company_codes.append(company_code)
            revenue_growth.append(results[revenue_indicator]["change_rate"] * 100)
            profit_growth.append(results[profit_indicator]["change_rate"] * 100)
    
    # Tạo biểu đồ cột
    x = np.arange(len(company_codes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, revenue_growth, width, label='Tăng trưởng doanh thu', color='skyblue')
    rects2 = ax.bar(x + width/2, profit_growth, width, label='Tăng trưởng lợi nhuận', color='salmon')
    
    # Thêm giá trị lên cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Thiết lập trục
    ax.set_ylabel('Tỷ lệ tăng trưởng (%)')
    ax.set_title('Dự báo tăng trưởng đến năm 2028')
    ax.set_xticks(x)
    ax.set_xticklabels(company_codes)
    ax.legend()
    
    # Thêm đường tham chiếu
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    chart_path = OUTPUT_DIR / "growth_forecast.png"
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path 