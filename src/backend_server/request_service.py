import json
import requests
import numpy as np


SERVICE_URL = 'http://localhost:5001/forecast'  # Đảm bảo trùng với endpoint của BentoML


def make_request_to_bento_service(service_url: str, input_array: np.ndarray) -> np.ndarray:
    serialized_input_data = json.dumps(input_array.tolist())  # Chuyển NumPy array thành JSON
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    
    # Chuyển kết quả từ JSON về NumPy array
    result_array = np.array(json.loads(response.text))
    return result_array


def prepare_data():
    data = np.random.rand(1, 150, 32)  # Dữ liệu đầu vào có shape (1, 150, 32)
    return data


if __name__ == '__main__':
    data = prepare_data()
    result = make_request_to_bento_service(SERVICE_URL, data)
    print(result)
    print(type(result))  
    print(result.shape)
