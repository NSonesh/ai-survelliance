import csv, os

class AlertLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.w = csv.writer(self.file)
        self.w.writerow(['timestamp','type','track_id','details'])

    def log(self, timestamp, typ, track_id, details):
        self.w.writerow([timestamp, typ, track_id, details])
        self.file.flush()

    def close(self):
        self.file.close()
