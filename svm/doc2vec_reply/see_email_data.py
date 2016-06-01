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
num_yes_emails = 0
num_no_emails = 0
for email in view.emails:
    if email["got_reply"] == "yes" and num_yes_emails < 10:
        print "YES:"
        print email['body'].encode("utf-8")
        print ""
        num_yes_emails += 1
    elif num_no_emails < 10:
        print "NO:"
        print email['body'].encode("utf-8")
        print ""
        num_no_emails += 1

num_contain_thanks = 0
num_no_thanks = 0
for email in view.emails:
    body_text = email['body'].encode('utf-8').lower()
    if 'thank' in body_text or 'thanks' in body_text:
        num_contain_thanks += 1
    else:
        num_no_thanks += 1

print "num_contain_thanks:", num_contain_thanks
print "num_no_thanks:", num_no_thanks









