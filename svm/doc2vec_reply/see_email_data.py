import json
import sys

class DataViewer(object):
    def __init__(self):
        self.emails = []

    def load_data(self, data_filename, stopword_filename="../stopwords.txt"):
        count = 0
        with open(data_filename, "r") as in_file:
            for line in in_file:
                msg = dict({})
                msg = json.loads(line)
                self.emails.append(msg)
                count += 1

view = DataViewer()
data_filename = sys.argv[1]
view.load_data(data_filename)
for i in range(5):
    print view.emails[i]