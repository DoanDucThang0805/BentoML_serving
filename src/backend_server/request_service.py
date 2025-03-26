import requests
import numpy as np
import pandas as pd

# Define the API endpoint
base_url = "http://localhost:5001"
predict_url = f"{base_url}/forecast"
headers = {"Content-Type": "application/json"}
# Number of samples
num_samples = 246

# Create sample data
data = {
    "date": pd.date_range(start="2024-01-01", periods=num_samples, freq="h").strftime('%Y-%m-%d %H:%M:%S').tolist(),
    "cur": np.random.uniform(0, 100, num_samples).tolist(),
    "cur_R": np.random.uniform(0, 100, num_samples).tolist(),
    "cur_S": np.random.uniform(0, 100, num_samples).tolist(),
    "cur_T": np.random.uniform(0, 100, num_samples).tolist(),
    "vol_RN": np.random.uniform(200, 250, num_samples).tolist(),
    "vol_SN": np.random.uniform(200, 250, num_samples).tolist(),
    "vol_TN": np.random.uniform(200, 250, num_samples).tolist(),
    "vol_RS": np.random.uniform(350, 400, num_samples).tolist(),
    "vol_ST": np.random.uniform(350, 400, num_samples).tolist(),
    "vol_TR": np.random.uniform(350, 400, num_samples).tolist(),
    "pow_P": np.random.uniform(500, 1000, num_samples).tolist(),
    "pow_Q": np.random.uniform(50, 200, num_samples).tolist(),
    "pow_S": np.random.uniform(500, 1200, num_samples).tolist(),
    "pow_DC": np.random.uniform(100, 500, num_samples).tolist(),
    "tot_Exp": np.random.randint(1000, 5000, num_samples).tolist(),
    "cosPhi": np.random.uniform(0.8, 1, num_samples).tolist(),
    "temp": np.random.uniform(15, 40, num_samples).tolist(),
}


# Send POST request
response = requests.post(
    predict_url,
    json={"data": data},
    headers={"content-type": "application/json"}
)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
