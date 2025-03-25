import json
import requests
import numpy as np

SERVICE_URL = 'http://localhost:5001/forecast'  # Đảm bảo trùng với endpoint của BentoML

def make_request_to_bento_service(service_url: str, input_array: np.ndarray) -> np.ndarray:
    serialized_input_data = json.dumps(input_array.tolist())  # Chuyển NumPy array thành JSON
    try:
        response = requests.post(
            service_url,
            data=serialized_input_data,
            headers={"content-type": "application/json"},
            timeout=10  # Thêm timeout để tránh request treo mãi
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Lỗi kết nối đến API: {e}")

    if response.status_code != 200:
        raise Exception(f"Lỗi {response.status_code}: {response.text}")

    try:
        result_json = json.loads(response.text)  # Kiểm tra response có đúng JSON không
        result_array = np.array(result_json)  # Convert sang NumPy
    except json.JSONDecodeError:
        raise Exception("Lỗi: API không trả về JSON hợp lệ")
    except Exception as e:
        raise Exception(f"Lỗi khi parse JSON: {e}")

    return result_array


def prepare_data():
    data = np.random.rand(1, 150, 32)  # Dữ liệu đầu vào có shape (1, 150, 32)
    return data


if __name__ == '__main__':
    data = prepare_data()
    try:
        result = make_request_to_bento_service(SERVICE_URL, data)
        print(result)
        print(type(result))
        print(result.shape)
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
