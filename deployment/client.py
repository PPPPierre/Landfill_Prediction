import requests
import yaml
import time
import os

class InferClient:

    END_OF_LOGS_SIGNAL = "$END_OF_LOGS$"

    def __init__(self, server_url):
        self.server_url = server_url

    def send_inference_request(self, config_data):
        headers = {"Content-Type": "application/x-yaml"}
        response = requests.post(f"{self.server_url}/infer", data=config_data, headers=headers)
        response_json = response.json()
        if "task_id" in response_json:
            return response_json["task_id"]
        else:
            raise ValueError(f"Error: {response_json.get('error')}")

    def send_train_request(self, config_data):
        headers = {"Content-Type": "application/x-yaml"}
        response = requests.post(f"{self.server_url}/train", data=config_data, headers=headers)
        response_json = response.json()
        if "task_id" in response_json:
            return response_json["task_id"]
        else:
            raise ValueError(f"Error: {response_json.get('error')}")

    def get_result(self, task_id):
        response = requests.get(f"{self.server_url}/result/{task_id}")
        return response.json()

    def wait_for_result(self, task_id, poll_interval=5):
        while True:
            result = self.get_result(task_id)
            if result["status"] == "completed":
                return result["data"]
            time.sleep(poll_interval)

    def get_log(self, task_id):
        try:
            print("Connecting to the server...")
            response = requests.get(f"{self.server_url}/log/{task_id}", stream=True)

            if response.status_code == 200:
                print("Connected! Receiving logs:")
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        print(f"Log: {line}")
                        # Check whether received end signal
                        if self.END_OF_LOGS_SIGNAL in line.strip():
                            print("Received end of logs signal, closing connection.")
                            break
            else:
                print(f"Failed to connect, server responded with status {response.status_code}")
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    client = InferClient(server_url="http://127.0.0.1:5000")

    # read the config file
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Train
    config_path = os.path.join(root_path, "configs/resnet18_train.yaml")
    with open(config_path, 'r') as file:
        data = file.read()
    task_id_train = client.send_train_request(data)
    print(f"Task ID: {task_id_train}")
    # Get train log
    client.get_log(task_id_train)

    # Inference
    config_path = os.path.join(root_path, "configs/resnet18_pred_client.yaml")
    with open(config_path, 'r') as file:
        data = file.read()
    # data['model']['task_id'] = task_id_train
    task_id_infer = client.send_inference_request(data)
    print(f"Task ID: {task_id_infer}")
    # wait for the inference result
    result = client.wait_for_result(task_id_infer)
    print("Inference Result:")
    print(result)