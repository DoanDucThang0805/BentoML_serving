import json
import numpy as np

data = np.random.rand(1, 150, 32).tolist()  # Chuyển NumPy array thành danh sách
with open("fake_data.json", "w") as f:
    json.dump(data, f, indent=4)
