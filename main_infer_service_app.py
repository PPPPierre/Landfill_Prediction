from flask import Flask, request, jsonify
import queue
import threading
import yaml
import os
import uuid
import psutil
import datetime
import traceback
from collections import defaultdict

from main_pred import main as predic_job
from src.utils.logger import init_logger

app = Flask(__name__)

# Set save dir
root_path = os.path.dirname(os.path.abspath(__file__))
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

    # Generate a unique id for each job
    task_id = str(uuid.uuid4())

    logger.info(f"Received request for task {task_id} with data: {data}")
    logger.info(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")
    
    # add request into the queue
    request_queue.put((task_id, config))
    results[task_id] = {"status": 1, "result": ""}
    
    return jsonify({"task_id": task_id})

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    logger.info(f"Received request for getting results for task {task_id}")
    if task_id not in results:
        return jsonify({"status": "Job not found"}), 404
    elif results[task_id]["status"] == 1:
        return jsonify({"status": "Job not completed yet"}), 202
    elif results[task_id]["status"] == -1:
        return jsonify({"status": "Inference job failed"}), 500
    else:
        result_file_path = results[task_id]["result_path"]
        with open(result_file_path, 'r') as f:
            content = f.read()
            logger.info(f"Return data for task {task_id}: {content}")
            return jsonify({"status": "completed", "data": content})

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the exception
    logger.error(f"Exception occurred: {e}")
    # Return error response
    return jsonify({"error": "An error occurred while processing your request."}), 500

def worker():
    while True:
        task_id, config = request_queue.get()
        # The model inference
        result_path = predic_job(config)
        # save result file path into the dict
        if os.path.exists(result_path):
            results[task_id] = {"status": 2, "result_path": result_path}
        else:
            results[task_id] = {"status": -1}

if __name__ == '__main__':
    # make sure the result dir exists
    os.makedirs(RESULT_DIR, exist_ok=True)

    # start the working thread
    threading.Thread(target=worker, daemon=True).start()
    app.run(debug=True)