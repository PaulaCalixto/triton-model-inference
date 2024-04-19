import requests
import json
import numpy as np

url = "http://localhost:8000/v2/models/modelo_regressao/versions/1/infer"

input_data = np.random.rand(1, 367).astype(np.float32)

payload = json.dumps({
  "inputs": [
    {
      "name": "input",
      "shape": input_data.shape,
      "datatype": "FP32",
      "data": input_data.tolist()
    }
  ]
})

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)