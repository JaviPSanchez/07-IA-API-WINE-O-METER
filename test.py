import requests

# We have put the localhost URL as default, feel free to change it

ENDPOINT = "http://127.0.0.1:4000/predict"

# This a simple example of input
input_simple = {"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]}


test_1 = requests.post(ENDPOINT, json=input_simple)
#assert res.status_code == 200
print(test_1, test_1.json())


# This a example of input with several inputs
# input_multiple = {
#     "input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8],
#               [5.0, 0.98, 0.32, 18.9, 0.050, 75.0, 122.0, 0.401, 3.1, 0.21, 1.2]]
# }

# res = requests.post(ENDPOINT, json=input_multiple)
# assert res.status_code == 200
# print(res.json())