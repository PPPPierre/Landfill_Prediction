import threading
import logging
from main import main  # or wherever your main function resides.

class WorkerThread(threading.Thread):
    def __init__(self, request_queue, results):
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.results = results
        self.logger = logging.getLogger('__main__')

    def run(self):
        while True:
            task_id, config = self.request_queue.get()
            self.logger.info(f"start task {task_id}, saved to {config['result_dir']}")
            # Training or inference task
            success = main(config)
            # save result file path into the dict
            self.logger.info(f"task {task_id} finished, saved to f{config['result_dir']}")
            if success:
                self.results[task_id].update({"status": 2})
            else:
                self.results[task_id].update({"status": -1})