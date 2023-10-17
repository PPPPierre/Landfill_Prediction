import queue
import threading
import yaml
import os
import sys
import uuid
import psutil
import datetime
import time
import traceback

from flask import Flask, request, jsonify, Response, stream_with_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from main import main
from src.utils.logger import init_logger

app = Flask(__name__)

# Set save dir
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
time_stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")
log_dir = os.path.join(root_path, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logger = init_logger("__main__", log_dir, time_stamp)

# The request quueue and result dict
request_queue = queue.Queue()
results = {}

# path of results
RESULT_DIR = 'results'

@app.route('/infer', methods=['POST'])
def infer():
    data = request.data.decode('utf-8')
    try:
        config = yaml.safe_load(data)
    except yaml.YAMLError:
        return jsonify({"error": "Invalid YAML format"}), 400
    
    # Generate a unique id for each task
    task_id = str(uuid.uuid4())

    if inference_config_check(config):
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

    logger.info(f"Received request for task {task_id} with data: {data}")
    logger.info(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")
    
    # add request into the queue
    request_queue.put((task_id, config))
    results[task_id] = {"status": 1, "result_dir": result_dir}
    
    return jsonify({"task_id": task_id})

def inference_config_check(config: dict):
    return True

@app.route('/train', methods=['POST'])
def train():
    data = request.data.decode('utf-8')
    try:
        config = yaml.safe_load(data)
    except yaml.YAMLError:
        return jsonify({"error": "Invalid YAML format"}), 400
    
    # Generate a unique id for each task
    task_id = str(uuid.uuid4())
    
    if train_config_check(config):
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

    logger.info(f"Received request for task {task_id} with data: {data}")
    logger.info(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")
    
    # add request into the queue
    request_queue.put((task_id, config))
    results[task_id] = {"status": 1, "result_dir": result_dir}
    
    return jsonify({"task_id": task_id})

def train_config_check(config: dict):
    return True

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    logger.info(f"Received request for getting results for task {task_id}")
    if task_id not in results:
        return jsonify({"status": "Task not found"}), 404
    elif results[task_id]["status"] == 1:
        return jsonify({"status": "Task not completed yet"}), 202
    elif results[task_id]["status"] == -1:
        return jsonify({"status": "Inference task failed"}), 500
    else:
        result_dir = results[task_id]["result_dir"]
        result_file_path = os.path.join(result_dir, "result.geojson")
        if not os.path.exists(result_file_path):
            return jsonify({'error': 'Result file not found.'}), 404
        else:
            with open(result_file_path, 'r') as f:
                content = f.read()
                logger.info(f"Return data for task {task_id}: {content}")
                return jsonify({"status": "completed", "data": content})

@app.route('/log/<task_id>', methods=['GET'])
def get_log_stream(task_id):
    logger.info(f"Received request for getting log for task {task_id}")
    if task_id not in results:
        return jsonify({"status": "Task not found"}), 404
    else:
        result_dir = results[task_id]["result_dir"]
        log_file_path = os.path.join(result_dir, "run.log")
        if not os.path.exists(log_file_path):
            return jsonify({'error': 'Log file not found.'}), 404
        else:
            return Response(stream_with_context(generate_log_stream(log_file_path)), mimetype="text/plain")

def generate_log_stream(log_path: str):
    with open(log_path, "r") as f:
        # f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            yield line

@app.errorhandler(Exception)
def handle_exception(e):
    error_message = traceback.format_exc()
    # Log the exception
    logger.error(f"Exception occurred: {error_message}")
    # Return error response
    return jsonify({"error": "An error occurred while processing your request."}), 500

def worker():
    while True:
        task_id, config = request_queue.get()
        logger.info(f"start task {task_id}, saved to {config['result_dir']}")
        # Training or inference task
        success = main(config)
        # save result file path into the dict
        logger.info(f"task {task_id} finished, saved to f{config['result_dir']}")
        if success:
            results[task_id].update({"status": 2})
        else:
            results[task_id].update({"status": -1})

if __name__ == '__main__':
    # make sure the result dir exists
    os.makedirs(RESULT_DIR, exist_ok=True)

    # start the working thread
    threading.Thread(target=worker, daemon=True).start()
    app.run(debug=True)