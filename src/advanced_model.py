"""
Mô hình dự báo tài chính tiên tiến
==================================
Sử dụng kết hợp nhiều phương pháp tiên tiến để đạt độ chính xác cao nhất
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# Đường dẫn đầu ra
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

class AdvancedFinancialForecaster:
    """
    Lớp dự báo tài chính tiên tiến kết hợp nhiều phương pháp
    """
    
    def __init__(self, seasonality=4):
        """
        Khởi tạo mô hình dự báo
        
        Args:
            seasonality: Độ dài chu kỳ mùa vụ (4 quý cho dữ liệu quý)
        """
        self.seasonality = seasonality
        self.models = {
            "linear": LinearRegression(),
            "huber": HuberRegressor(epsilon=1.35, max_iter=200),
            "theil_sen": TheilSenRegressor(random_state=42, max_iter=300),
            "ransac": RANSACRegressor(random_state=42),
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "gbr": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Các trọng số cho mô hình kết hợp (sẽ được cập nhật dựa trên hiệu suất)
        self.weights = {model: 1/len(self.models) for model in self.models}
        
        # Bộ chuẩn hóa dữ liệu
        self.scaler = RobustScaler()
        
        # Lưu các mô hình đã huấn luyện
        self.trained_models = {}
        
    def _create_features(self, data):
        """
        Tạo đặc trưng từ dữ liệu chuỗi thời gian
        
        Args:
            data: Series dữ liệu chuỗi thời gian
            
        Returns:
            X: Đặc trưng
            y: Giá trị mục tiêu
        """
        # Xử lý giá trị NaN
        data = data.copy()
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        df = pd.DataFrame()
        
        # Tạo đặc trưng lag (giá trị trễ)
        for i in range(1, min(5, len(data)//2)):
            df[f'lag_{i}'] = data.shift(i)
        
        # Tạo đặc trưng xu hướng
        df['trend'] = np.arange(len(data))
        
        # Tạo đặc trưng mùa vụ
        for i in range(1, self.seasonality):
            df[f'season_{i}'] = (np.arange(len(data)) % self.seasonality == i).astype(int)
        
        # Tạo đặc trưng biến động
        if len(data) >= 3:
            df['volatility'] = data.rolling(3).std().fillna(method='bfill')
        
        # Tạo đặc trưng tăng trưởng
        if len(data) >= 4:
            df['growth_rate'] = data.pct_change(4).fillna(0)
        
        # Xóa các dòng có giá trị NaN
        df['target'] = data
        df = df.dropna()
        
        # Tách đặc trưng và mục tiêu
        X = df.drop('target', axis=1)
        y = df['target']
        
        return X, y
    
    def _evaluate_models(self, X, y):
        """
        Đánh giá và tính toán trọng số cho các mô hình
        
        Args:
            X: Đặc trưng
            y: Giá trị mục tiêu
            
        Returns:
            weights: Trọng số cho mỗi mô hình
        """
        if len(X) < 8:  # Không đủ dữ liệu để thực hiện kiểm tra chéo
            return self.weights
        
        # Phân chia dữ liệu theo thời gian
        tscv = TimeSeriesSplit(n_splits=min(3, len(X)//3))
        
        scores = {model: [] for model in self.models}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Chuẩn hóa dữ liệu
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Tính điểm R² điều chỉnh
                    r2 = r2_score(y_test, y_pred)
                    if r2 < 0:  # Nếu mô hình tệ hơn dự đoán trung bình
                        r2 = 0
                    
                    # Sử dụng MAE điều chỉnh
                    mae = mean_absolute_error(y_test, y_pred)
                    max_val = max(abs(y_test.max()), abs(y_test.min()))
                    if max_val > 0:
                        norm_mae = 1 - (mae / max_val)  # Điểm số càng cao càng tốt
                    else:
                        norm_mae = 0
                    
                    # Kết hợp các độ đo
                    score = (r2 + norm_mae) / 2
                    scores[name].append(score)
                except Exception as e:
                    scores[name].append(0)
        
        # Tính trung bình
        avg_scores = {model: np.mean(s) if len(s) > 0 else 0 for model, s in scores.items()}
        
        # Chuẩn hóa để tạo trọng số (tổng = 1)
        total = sum(avg_scores.values())
        if total > 0:
            weights = {model: score/total for model, score in avg_scores.items()}
        else:
            weights = {model: 1/len(self.models) for model in self.models}
        
        return weights
    
    def fit(self, series):
        """
        Huấn luyện mô hình với dữ liệu chuỗi thời gian
        
        Args:
            series: Series dữ liệu chuỗi thời gian
            
        Returns:
            self: Đối tượng đã được huấn luyện
        """
        # Tạo đặc trưng
        X, y = self._create_features(series)
        
        if len(X) <= 1:
            # Không đủ dữ liệu để huấn luyện
            return self
        
        # Đánh giá và cập nhật trọng số
        self.weights = self._evaluate_models(X, y)
        
        # Huấn luyện mô hình với toàn bộ dữ liệu
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                self.trained_models[name] = model
            except Exception as e:
                print(f"Lỗi khi huấn luyện mô hình {name}: {e}")
                # Sử dụng mô hình dự phòng (linear)
                if name != "linear":
                    try:
                        self.trained_models[name] = LinearRegression().fit(X_scaled, y)
                    except:
                        pass
        
        return self
    
    def predict(self, series, periods=16):
        """
        Dự báo giá trị tương lai
        
        Args:
            series: Series dữ liệu chuỗi thời gian
            periods: Số kỳ cần dự báo
            
        Returns:
            forecast: Mảng các giá trị dự báo
        """
        if not self.trained_models:
            # Nếu chưa huấn luyện, thực hiện huấn luyện
            self.fit(series)
        
        # Tạo đặc trưng từ dữ liệu hiện có
        X, _ = self._create_features(series)
        
        if len(X) == 0:
            # Không đủ dữ liệu để dự báo
            # Trả về một dự báo đơn giản (tăng tuyến tính)
            last_value = series.iloc[-1] if not pd.isna(series.iloc[-1]) else 0
            return last_value + np.arange(1, periods+1) * 0.01 * last_value
        
        # Xác định ngưỡng giảm tối đa dựa trên đặc điểm chuỗi thời gian
        non_zero_values = series[series > 0]
        if len(non_zero_values) > 0:
            min_historical = non_zero_values.min()
            historical_min_ratio = min_historical / non_zero_values.max()
            # Ngưỡng giảm tối đa: không thấp hơn 30% giá trị thấp nhất trong lịch sử
            min_threshold = min_historical * 0.3
        else:
            min_threshold = 0
            historical_min_ratio = 0.1
        
        # Dự báo từng bước
        forecasts = []
        current_data = series.copy()
        
        # Lưu giá trị cuối cùng để đảm bảo tính liên tục
        last_hist_value = current_data.iloc[-1]
        
        for i in range(periods):
            # Tạo đặc trưng cho bước hiện tại
            X_current, _ = self._create_features(current_data)
            
            if len(X_current) == 0:
                # Nếu không thể tạo đặc trưng, sử dụng giá trị cuối cùng
                last_value = current_data.iloc[-1]
                forecasts.append(last_value)
                current_data = pd.concat([current_data, pd.Series([last_value])])
                continue
            
            # Chuẩn hóa dữ liệu
            X_current_scaled = self.scaler.transform(X_current.iloc[[-1]])
            
            # Dự báo bằng mỗi mô hình và tính trung bình có trọng số
            predictions = []
            sum_weights = 0
            
            for name, model in self.trained_models.items():
                try:
                    pred = model.predict(X_current_scaled)[0]
                    # Kiểm tra giá trị hợp lệ và không âm (nếu dữ liệu không có giá trị âm)
                    if not np.isnan(pred) and not np.isinf(pred):
                        # Kiểm tra xem chuỗi dữ liệu có giá trị âm không
                        has_negative = (series < 0).any()
                        # Nếu không có giá trị âm trong dữ liệu lịch sử, đảm bảo dự báo không âm
                        if not has_negative and pred < 0:
                            # Đặt giá trị dương nhỏ thay vì bỏ qua
                            pred = min_threshold if min_threshold > 0 else last_hist_value * 0.01
                        weight = self.weights.get(name, 0)
                        predictions.append((pred, weight))
                        sum_weights += weight
                except Exception as e:
                    continue
            
            # Tính trung bình có trọng số
            if predictions and sum_weights > 0:
                weighted_avg = sum(p * w for p, w in predictions) / sum_weights
            else:
                # Dự phòng: sử dụng giá trị cuối cùng
                weighted_avg = current_data.iloc[-1]
            
            # Áp dụng hiệu chỉnh nếu giá trị dự báo quá khác biệt
            last_value = current_data.iloc[-1]
            
            # Kiểm tra nếu last_value gần bằng 0
            if abs(last_value) < 1e-10:
                # Nếu giá trị gần 0, giới hạn giá trị dự báo
                if abs(weighted_avg) > 1e6:  # Giới hạn giá trị tuyệt đối lớn
                    weighted_avg = np.sign(weighted_avg) * 1e6
            else:
                # Kiểm tra tỷ lệ thay đổi và giới hạn trong khoảng hợp lý
                change_ratio = weighted_avg / last_value - 1
                
                # Giới hạn tỷ lệ thay đổi trong khoảng -35% đến +50% cho mỗi bước dự báo
                if change_ratio > 0.5:  # Tăng quá 50%
                    weighted_avg = last_value * 1.5  # Giới hạn tăng tối đa 50%
                elif change_ratio < -0.35:  # Giảm quá 35%
                    weighted_avg = last_value * 0.65  # Giới hạn giảm tối đa 35%
            
            # Thiết lập ngưỡng sàn cho giá trị dự báo
            if weighted_avg < min_threshold:
                # Không cho phép giá trị nhỏ hơn ngưỡng tối thiểu
                weighted_avg = min_threshold
            
            # Đối với chỉ số lợi nhuận, không cho phép duy trì giá trị gần 0 nhiều kỳ liên tiếp
            is_profit_indicator = any(term in str(series.name).lower() for term in ["lợi nhuận", "profit"])
            if is_profit_indicator and i > 0 and abs(weighted_avg) < last_hist_value * 0.01:
                # Nếu dự báo lợi nhuận gần 0 và đây là chỉ số lợi nhuận, điều chỉnh
                adjustment_factor = 0.1 + (periods - i) / periods * 0.2  # Điều chỉnh tăng dần cho các kỳ sau
                weighted_avg = last_hist_value * adjustment_factor
            
            # Thêm vào dự báo
            forecasts.append(weighted_avg)
            
            # Cập nhật dữ liệu hiện tại với giá trị dự báo
            current_data = pd.concat([current_data, pd.Series([weighted_avg])])
        
        return np.array(forecasts)
    
    def evaluate_forecast(self, historical_series, forecast_values):
        """
        Đánh giá xu hướng dự báo
        
        Args:
            historical_series: Series dữ liệu lịch sử
            forecast_values: Mảng các giá trị dự báo
            
        Returns:
            dict: Kết quả đánh giá
        """
        # Lấy giá trị không phải NaN cuối cùng
        last_values = historical_series.dropna()
        if len(last_values) == 0:
            return {
                "trend": "Không xác định",
                "change_rate": 0,
                "confidence": 0,
                "last_value": 0,
                "forecast_value": forecast_values[-1] if len(forecast_values) > 0 else 0
            }
        
        last_value = last_values.iloc[-1]
        
        # Tính tỷ lệ thay đổi
        if len(forecast_values) > 0:
            forecast_end = forecast_values[-1]
            
            # Tính toán tỷ lệ thay đổi một cách an toàn
            if abs(last_value) < 1e-10:
                # Xử lý trường hợp chia cho 0 hoặc gần 0
                if abs(forecast_end) < 1e-10:
                    change_rate = 0  # Cả hai đều gần 0, không có thay đổi
                else:
                    # Tính tỷ lệ thay đổi tương đối
                    sign = 1 if forecast_end > 0 else -1
                    change_rate = sign * min(abs(forecast_end / max(abs(last_value), 1e-5)), 2)
            else:
                change_rate = (forecast_end - last_value) / abs(last_value)
            
            # Giới hạn tỷ lệ thay đổi trong khoảng hợp lý (-95% đến +200%)
            change_rate = max(min(change_rate, 2.0), -0.95)
            
            # Phân tích xu hướng
            if change_rate > 0.05:
                trend = "Tăng"
            elif change_rate < -0.05:
                trend = "Giảm"
            else:
                trend = "Ổn định"
            
            # Cải thiện tính toán độ tin cậy
            confidence = 0.7  # Giá trị mặc định
            
            if len(forecast_values) >= 3:
                # Đánh giá độ tin cậy dựa trên nhiều yếu tố
                
                # 1. Độ ổn định của dự báo (không dao động quá mạnh)
                if np.all(np.abs(forecast_values) < 1e-5):
                    # Nếu tất cả giá trị dự báo quá nhỏ (gần 0)
                    stability_score = 0  
                else:
                    # Tính độ dao động tương đối của dự báo
                    rel_std = np.std(forecast_values) / (np.mean(np.abs(forecast_values)) + 1e-10)
                    # Giới hạn trong khoảng [0, 1]
                    stability_score = max(0, min(1, 1 - rel_std))
                
                # 2. Đánh giá tính nhất quán xu hướng
                # Kiểm tra xu hướng có nhất quán không (không đổi chiều liên tục)
                diffs = np.diff(forecast_values)
                sign_changes = np.sum(np.diff(np.signbit(diffs)) != 0)
                # Nếu có nhiều thay đổi dấu, độ nhất quán thấp
                trend_consistency = max(0, 1 - sign_changes / (len(forecast_values) - 2)) if len(forecast_values) > 2 else 0.5
                
                # 3. Đánh giá mức độ thay đổi thái quá
                # Tỷ lệ thay đổi quá lớn sẽ giảm độ tin cậy
                change_plausibility = 1.0
                if abs(change_rate) > 0.7:  # Thay đổi >70% cần xem xét
                    change_plausibility = max(0.2, 1 - (abs(change_rate) - 0.7) / 1.3)
                
                # 4. Nếu là lợi nhuận và dự báo gần 0 hoặc âm
                is_profit = any(term in str(historical_series.name).lower() for term in ["lợi nhuận", "profit"])
                if is_profit and forecast_end < last_value * 0.05 and last_value > 0:
                    # Giảm độ tin cậy nếu dự báo lợi nhuận giảm mạnh gần 0
                    profit_factor = max(0.3, min(1.0, forecast_end / (last_value * 0.05)))
                else:
                    profit_factor = 1.0
                
                # Tính điểm tin cậy tổng hợp
                confidence = 0.4 * stability_score + 0.3 * trend_consistency + 0.2 * change_plausibility + 0.1 * profit_factor
                
                # Giới hạn trong khoảng [0, 0.99]
                confidence = max(0.01, min(0.99, confidence))
            
            return {
                "trend": trend,
                "change_rate": change_rate,
                "confidence": confidence,
                "last_value": last_value,
                "forecast_value": forecast_end
            }
        else:
            return {
                "trend": "Không xác định",
                "change_rate": 0,
                "confidence": 0,
                "last_value": last_value,
                "forecast_value": last_value
            }
    
    def plot_forecast(self, historical_series, forecast_values, title, company_code, output_path=None):
        """
        Vẽ biểu đồ dự báo
        
        Args:
            historical_series: Series dữ liệu lịch sử
            forecast_values: Mảng các giá trị dự báo
            title: Tiêu đề biểu đồ
            company_code: Mã công ty
            output_path: Đường dẫn đầu ra (tùy chọn)
            
        Returns:
            str: Đường dẫn đến biểu đồ đã lưu
        """
        plt.figure(figsize=(12, 6))
        
        # Xác định các mốc thời gian
        # Giả sử dữ liệu bắt đầu từ Q1-2020
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        all_quarters = []
        
        # Tạo danh sách tất cả các quý từ Q1-2020 đến Q4-2028
        for year in range(2020, 2029):
            for q in quarters:
                all_quarters.append(f"{q}-{year}")
        
        # Lấy đủ số quý cần thiết (dữ liệu lịch sử + dự báo)
        total_periods = len(historical_series) + len(forecast_values)
        time_labels = all_quarters[:total_periods]
        
        # Vẽ dữ liệu lịch sử
        plt.plot(range(len(historical_series)), historical_series.values, marker='o', 
                 label='Dữ liệu lịch sử', color='blue', linewidth=2)
        
        # Vẽ dữ liệu dự báo
        history_len = len(historical_series)
        forecast_range = range(history_len, history_len + len(forecast_values))
        plt.plot(forecast_range, forecast_values, marker='s', 
                 label='Dự báo', color='red', linestyle='--', linewidth=2)
        
        # Thêm vùng dự báo
        plt.axvspan(history_len - 0.5, history_len + len(forecast_values) - 0.5, 
                    alpha=0.2, color='gray', label='Khoảng dự báo')
        
        # Thông tin dự báo
        evaluation = self.evaluate_forecast(historical_series, forecast_values)
        change_text = f"Xu hướng: {evaluation['trend']} ({evaluation['change_rate']*100:.1f}%)"
        confidence_text = f"Độ tin cậy: {evaluation['confidence']*100:.1f}%"
        
        # Thêm thông tin vào biểu đồ
        plt.annotate(change_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                    fontsize=10, backgroundcolor='white', bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))
        plt.annotate(confidence_text, xy=(0.02, 0.90), xycoords='axes fraction', 
                    fontsize=10, backgroundcolor='white', bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))
        
        # Định dạng đồ thị
        plt.title(f"{title} - {company_code}")
        plt.ylabel('Giá trị (VND)')
        plt.xlabel('Thời gian')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Thêm nhãn x với các quý và năm thực tế
        # Hiển thị đủ điểm mốc nhưng không quá dày
        xtick_positions = list(range(0, total_periods, 4))  # Mỗi năm (4 quý)
        xtick_labels = [time_labels[i] for i in xtick_positions if i < len(time_labels)]
        
        # Thêm điểm cuối cùng nếu chưa có
        if total_periods-1 not in xtick_positions and total_periods > 0:
            xtick_positions.append(total_periods-1)
            xtick_labels.append(time_labels[-1])
            
        plt.xticks(xtick_positions, xtick_labels, rotation=45)
        
        # Lưu biểu đồ
        if output_path is None:
            output_path = OUTPUT_DIR / f"{company_code}_{title.replace(' ', '_')}_forecast.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)


def forecast_financial_indicator(historical_data, indicator_name, company_code, periods=16):
    """
    Dự báo chỉ số tài chính sử dụng mô hình tiên tiến
    
    Args:
        historical_data: DataFrame dữ liệu lịch sử
        indicator_name: Tên chỉ số cần dự báo
        company_code: Mã công ty
        periods: Số kỳ cần dự báo
        
    Returns:
        dict: Kết quả dự báo
    """
    if indicator_name not in historical_data.index:
        return None
    
    # Lấy chuỗi dữ liệu
    series = pd.Series(historical_data.loc[indicator_name])
    
    # Tạo và huấn luyện mô hình
    model = AdvancedFinancialForecaster(seasonality=4)  # 4 quý một năm
    
    # Dự báo
    forecast_values = model.predict(series, periods=periods)
    
    # Đánh giá xu hướng
    evaluation = model.evaluate_forecast(series, forecast_values)
    
    # Thêm dữ liệu dự báo vào kết quả
    evaluation["forecast"] = forecast_values
    
    # Vẽ biểu đồ sau khi đã hoàn tất mọi điều chỉnh
    short_name = indicator_name.split("(")[0].strip()
    chart_path = model.plot_forecast(
        series, 
        evaluation["forecast"],  # Sử dụng dữ liệu từ evaluation để đảm bảo mọi điều chỉnh được áp dụng
        short_name, 
        company_code
    )
    
    evaluation["chart_path"] = chart_path
    
    return evaluation 