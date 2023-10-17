import queue
import yaml
import os
import sys
import uuid
import psutil
import datetime
import time
import traceback
from multiprocessing import Manager

from flask import Flask, request, jsonify, Response, stream_with_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from worker_thread import WorkerThread

from main import main
from src.utils.logger import init_logger

class AppServer:
    def __init__(self):
        self.app = Flask(__name__)

        # Define result dir
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        time_stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")
        log_dir = os.path.join(self.root_path, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logger = init_logger("__main__", log_dir, time_stamp)

        # Set up routes
        self.setup_routes()

        # Set up worker thread and data
        self.manager = Manager()
        self.request_queue = queue.Queue()
        self.results = self.manager.dict()
        self.worker = WorkerThread(self.request_queue, self.results)
        self.worker.start()

    def setup_routes(self):
        @self.app.route('/infer', methods=['POST'])
        def infer():
            data = request.data.decode('utf-8')
            try:
                config = yaml.safe_load(data)
            except yaml.YAMLError:
                return jsonify({"error": "Invalid YAML format"}), 400
            
            # Generate a unique id for each task
            task_id = str(uuid.uuid4())

            if self.inference_config_check(config):
                config['task_type'] = 'prediction'
                root_path = os.path.dirname(SCRIPT_DIR)
                result_dir = os.path.join(root_path, 'results', task_id)
                config['result_dir'] = result_dir
                if 'client_id' in config['model']:
                    config['model']['weight'] = os.path.join(result_dir, config['model']['task_id'], 'checkpoints', 'last.pth')
            else:
                response = jsonify({
                    'status': 'error',
                    'message': 'Invalid config file for a inference task'
                })
                return response, 400

            self.logger.info(f"Received request for task {task_id} with data: {data}")
            self.logger.info(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")
            
            # add request into the queue
            self.request_queue.put((task_id, config))
            self.results[task_id] = self.manager.dict({"status": 1, "result_dir": result_dir})
            
            return jsonify({"task_id": task_id})

        @self.app.route('/train', methods=['POST'])
        def train():
            data = request.data.decode('utf-8')
            try:
                config = yaml.safe_load(data)
            except yaml.YAMLError:
                return jsonify({"error": "Invalid YAML format"}), 400
            
            # Generate a unique id for each task
            task_id = str(uuid.uuid4())
            
            if self.train_config_check(config):
                config['task_type'] = 'train'
                root_path = os.path.dirname(SCRIPT_DIR)
                result_dir = os.path.join(root_path, 'results', task_id)
                config['result_dir'] = result_dir
            else:
                response = jsonify({
                    'status': 'error',
                    'message': 'Invalid config file for a training task'
                })
                return response, 400

            self.logger.info(f"Received request for task {task_id} with data: {data}")
            self.logger.info(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")
            
            # add request into the queue
            self.request_queue.put((task_id, config))
            self.results[task_id] = {"status": 1, "result_dir": result_dir}
            
            return jsonify({"task_id": task_id})
        
        @self.app.route('/result/<task_id>', methods=['GET'])
        def get_result(task_id):
            self.logger.info(f"Received request for getting results for task {task_id}")
            if task_id not in self.results:
                return jsonify({"status": "Task not found"}), 404
            elif self.results[task_id]["status"] == 1:
                return jsonify({"status": "Task not completed yet"}), 202
            elif self.results[task_id]["status"] == -1:
                return jsonify({"status": "Inference task failed"}), 500
            else:
                result_dir = self.results[task_id]["result_dir"]
                result_file_path = os.path.join(result_dir, "result.geojson")
                if not os.path.exists(result_file_path):
                    return jsonify({'error': 'Result file not found.'}), 404
                else:
                    with open(result_file_path, 'r') as f:
                        content = f.read()
                        self.logger.info(f"Return data for task {task_id}: {content}")
                        return jsonify({"status": "completed", "data": content})

        @self.app.route('/log/<task_id>', methods=['GET'])
        def get_log_stream(task_id):
            self.logger.info(f"Received request for getting log for task {task_id}")
            if task_id not in self.results:
                return jsonify({"status": "Task not found"}), 404
            else:
                result_dir = self.results[task_id]["result_dir"]
                log_file_path = os.path.join(result_dir, "run.log")
                if not os.path.exists(log_file_path):
                    return jsonify({'error': 'Log file not found.'}), 404
                else:
                    return Response(stream_with_context(self.generate_log_stream(log_file_path)), mimetype="text/plain")

        @self.app.errorhandler(Exception)
        def handle_exception(e):
            error_message = traceback.format_exc()
            # Log the exception
            self.logger.error(f"Exception occurred: {error_message}")
            # Return error response
            return jsonify({"error": "An error occurred while processing your request."}), 500

    def inference_config_check(self, config: dict):
        """
        check the format of the inference config
        """
        return True

    def train_config_check(self, config: dict):
        """
        check the format of the training config
        """
        return True
    
    def generate_log_stream(self, log_path: str):
        with open(log_path, "r") as f:
            # f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                yield line

    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    server = AppServer()
    server.run()