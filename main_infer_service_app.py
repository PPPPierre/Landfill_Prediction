from flask import Flask, request, jsonify
import queue
import threading
import yaml
import os
import uuid

from main_pred import main as predic_job

app = Flask(__name__)

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
    
    # add request into the queue
    request_queue.put((task_id, config))
    
    return jsonify({"task_id": task_id})


@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    if task_id not in results:
        return jsonify({"status": "processing"}), 202
    else:
        result_file_path = results[task_id]
        with open(result_file_path, 'r') as f:
            content = f.read()
            return jsonify({"status": "completed", "data": content})


def worker():
    while True:
        task_id, config = request_queue.get()
        
        # The model inference
        result_file = predic_job(config)
        
        # Fake geojson file for test
        # result_file = os.path.join(RESULT_DIR, f"{task_id}.geojson")
        # with open(result_file, 'w') as f:
        #     f.write('{"type": "FeatureCollection", "features": []}')
        
        # save result file path into the dict
        results[task_id] = result_file


if __name__ == '__main__':
    # make sure the result dir exists
    os.makedirs(RESULT_DIR, exist_ok=True)

    # start the working thread
    threading.Thread(target=worker, daemon=True).start()
    app.run(debug=True)