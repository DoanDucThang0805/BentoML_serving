import bentoml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import asyncio


my_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("tensorflow[and-cuda]", "numpy", "pandas", "scikit-learn", "keras")


@bentoml.service(
    image = my_image,
    resources={"gpu": 1},
    traffic={"timeout": 10},
)


class Forecast:
    bento_model = bentoml.models.BentoModel("timeseries_model:latest")


    def __init__(self):
        # Load model với đúng tag
        self.model = bentoml.keras.load_model(self.bento_model, device_name = "/device:GPU:0")



    def preprocessing_data(self, data_input: dict, batch_size: int) -> np.ndarray:
        '''
        Preprocessing input data for forecasting model
        Args:
        data_input: dictionary 
        batch_size: int
        Return:
        input_data: ndarray
        '''
        # Convert json to DataFrame
        input_df = pd.DataFrame(data_input)
        
        # Processing Time-based
        input_df['date'] = pd.to_datetime(input_df['date']) if 'date' in input_df.columns else pd.to_datetime(input_df.index)
        input_df['hour'] = input_df['date'].dt.hour
        input_df['day'] = input_df['date'].dt.day 
        input_df['day_of_week'] = input_df['date'].dt.dayofweek  # Monday=0, Sunday=6
        input_df['day_of_year'] = input_df['date'].dt.dayofyear
        input_df['month'] = input_df['date'].dt.month
        input_df['quarter'] = input_df['date'].dt.quarter
        input_df['year'] = input_df['date'].dt.year
        input_df['is_weekend'] = input_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  
        
        # Add time-of-day features
        input_df['is_morning'] = ((input_df['hour'] >= 6) & (input_df['hour'] < 12)).astype(int)
        input_df['is_afternoon'] = ((input_df['hour'] >= 12) & (input_df['hour'] < 18)).astype(int)
        input_df['is_evening'] = ((input_df['hour'] >= 18) & (input_df['hour'] < 22)).astype(int)
        input_df['is_night'] = (((input_df['hour'] >= 22) & (input_df['hour'] <= 23)) | ((input_df['hour'] >= 0) & (input_df['hour'] < 6))).astype(int)    
        
        # Cyclical encoding to handle circular nature of time features
        # For hour (0-23)
        input_df['hour_sin'] = np.sin(2 * np.pi * input_df['hour']/24)
        input_df['hour_cos'] = np.cos(2 * np.pi * input_df['hour']/24)  
        
        # For day of week (0-6)
        input_df['day_of_week_sin'] = np.sin(2 * np.pi * input_df['day_of_week']/7)
        input_df['day_of_week_cos'] = np.cos(2 * np.pi * input_df['day_of_week']/7)

        # For month (1-12)
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['month']/12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['month']/12)

        # For day of year (1-365)
        input_df['day_of_year_sin'] = np.sin(2 * np.pi * input_df['day_of_year']/365)
        input_df['day_of_year_cos'] = np.cos(2 * np.pi * input_df['day_of_year']/365) 

        # Thêm biến lưu sự thay đổi của tot_Exp
        input_df['tot_Exp_diff'] = input_df['tot_Exp'].diff()
        
        # Lag features (previous values) - Thay vol_RN bằng tot_Exp
        for lag in [1, 2, 3, 7, 14]:  # Various lag periods
            input_df[f'tot_Exp_lag_{lag}'] = input_df['tot_Exp'].shift(lag)
            input_df[f'tot_Exp_diff_lag_{lag}'] = input_df['tot_Exp_diff'].shift(lag)

        # Rolling window statistics - Thay vol_RN bằng tot_Exp và tot_Exp_diff
        for window in [6, 12, 24, 48]:  # Different window sizes
            # Thống kê cho tot_Exp
            input_df[f'tot_Exp_rolling_mean_{window}'] = input_df['tot_Exp'].rolling(window=window).mean()
            input_df[f'tot_Exp_rolling_std_{window}'] = input_df['tot_Exp'].rolling(window=window).std()
            input_df[f'tot_Exp_rolling_min_{window}'] = input_df['tot_Exp'].rolling(window=window).min()
            input_df[f'tot_Exp_rolling_max_{window}'] = input_df['tot_Exp'].rolling(window=window).max()

            # Thống kê cho tot_Exp_diff (sự thay đổi)
            input_df[f'tot_Exp_diff_rolling_mean_{window}'] = input_df['tot_Exp_diff'].rolling(window=window).mean()
            input_df[f'tot_Exp_diff_rolling_std_{window}'] = input_df['tot_Exp_diff'].rolling(window=window).std()
            
            # Drop NaN values created by lag and rolling features
            input_df.dropna(inplace=True)
            
        # Cập nhật danh sách features - Bổ sung các đặc trưng liên quan đến tot_Exp
        features = ['vol_RN', 'vol_SN', 'vol_TN', 'vol_RS', 'vol_ST', 'vol_TR', 'tot_Exp', 'tot_Exp_diff',
                    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
                    'is_weekend', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
                    'tot_Exp_lag_1', 'tot_Exp_lag_2', 'tot_Exp_lag_3',
                    'tot_Exp_diff_lag_1', 'tot_Exp_diff_lag_2', 'tot_Exp_diff_lag_3',
                    'tot_Exp_rolling_mean_24', 'tot_Exp_rolling_std_24',
                    'tot_Exp_diff_rolling_mean_24', 'tot_Exp_diff_rolling_std_24']
        # Thay đổi biến mục tiêu thành tot_Exp
        target = 'tot_Exp'

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        input_df[features] = scaler.fit_transform(input_df[features])

        # Thêm một scaler riêng cho biến mục tiêu để dễ dàng inverse transform sau này
        target_scaler = StandardScaler()
        input_df[[target]] = target_scaler.fit_transform(input_df[[target]])
        
        input_data = np.array(input_df[features + [target]])
        input_data = np.expand_dims(input_data, axis = 0)
        return input_data
    

    def post_processing(self, output_data: np.ndarray) -> dict:
        return {"predicted_data": output_data}
    
    
    @bentoml.api
    async def predict(self, data: dict) -> dict:
        input_data = self.preprocessing_data(data_input = data, batch_size = 1)
        result = await asyncio.to_thread(self.model.predict, input_data)
        json_result = self.post_processing(output_data = result.tolist())
        return json_result
    