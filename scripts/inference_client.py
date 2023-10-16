import requests
import yaml
import time
import os

class InferClient:
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

    def get_result(self, task_id):
        response = requests.get(f"{self.server_url}/result/{task_id}")
        return response.json()

    def wait_for_result(self, task_id, poll_interval=5):
        while True:
            result = self.get_result(task_id)
            if result["status"] == "completed":
                return result["data"]
            time.sleep(poll_interval)


if __name__ == "__main__":
    client = InferClient(server_url="http://127.0.0.1:5000")

    # read the config file
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_path, "configs/resnet18_pred.yaml")
    with open(config_path, 'r') as file:
        data = file.read()

    task_id = client.send_inference_request(data)
    print(f"Task ID: {task_id}")

    # wait for the inference result
    result = client.wait_for_result(task_id)
    print("Inference Result:")
    print(result)