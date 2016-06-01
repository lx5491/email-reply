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

def words_or_in_text(words, text):
    for word in words:
        if word in text:
            return True

    return False

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

num_contain_words_got_reply = 0
num_contain_words_no_reply = 0
num_no_words_got_reply = 0
num_no_words_no_reply = 0
words = ['thank', 'thanks']
for email in view.emails:
    body_text = email['body'].encode('utf-8').lower()
    if words_or_in_text(words, body_text) and email['got_reply'] == 'yes':
        num_contain_words_got_reply += 1
    elif words_or_in_text(words, body_text) and email['got_reply'] == 'no':
        num_contain_words_no_reply += 1
    elif not words_or_in_text(words, body_text) and email['got_reply'] == 'yes':
        num_no_words_got_reply += 1
    elif not words_or_in_text(words, body_text) and email['got_reply'] == 'no':
        num_no_words_no_reply += 1


print "words:", words
print "num_contain_words_got_reply:", num_contain_words_got_reply
print "num_contain_words_no_reply:", num_contain_words_no_reply
print "num_no_words_got_reply:", num_no_words_got_reply
print "num_no_words_no_reply:", num_no_words_no_reply
print "with the word '%s', %0.5f emails got reply" % ('thank', float(num_contain_words_got_reply) / (num_contain_words_got_reply + num_contain_words_no_reply))
print "without the word '%s', %0.5f emails got reply" % ('thank', float(num_no_words_got_reply) / (num_no_words_got_reply + num_no_words_no_reply))
print "total emails num:", (num_contain_words_got_reply + num_contain_words_no_reply + num_no_words_got_reply + num_no_words_no_reply)









