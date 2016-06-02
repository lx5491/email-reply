
import numpy as np
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import json
import sys
import os
import logging

class PersonReply(object):
    def __init__(self):
        self.emails = []

    def load_data(self, data_filename):
        count = 0
        with open(data_filename, "r") as in_file:
            for line in in_file:
                msg = dict({})
                msg = json.loads(line)
                self.emails.append(msg)
                count += 1

        self.basic_process()

    def basic_process(self):
        for email in self.emails:
            email["subject"] = email["subject"].lower() if email["subject"] != None else ""
            email["body"] = email["body"].lower()


class PersonReplyPlayground(object):
    def __init__(self, person_reply):
        self.person_reply = person_reply

    def subject_has_words(self, subject, words, relation="or"):
        word_in_count = 0
        if relation == "or":
            for word in words:
                if word in subject:
                    word_in_count += 1
                    break
        elif relation == "and":
            for word in words:
                if word in subject:
                    word_in_count += 1

        has_words = True if (relation == "or" and word_in_count > 0) or \
            (relation == "and" and word_in_count == len(words)) else False

        return has_words

    def subject_has_words_experiment(self, words, relation="or"):
        for email in self.person_reply.emails:
            userid = email["user"]
            reply_users = email["replied_from"]
            this_user_replied = True if userid in reply_users else False
            subject = email["subject"]
            
            test_words = ["please", "reply"]
            # TODO
            
    def experiment_1(self):
        X = np.zeros((len(self.person_reply.emails), 4), dtype=float)
        Y = np.zeros(len(self.person_reply.emails))
        print self.person_reply.emails
        for idx, email in enumerate(self.person_reply.emails):
            user_id = email["user"]
            sender_id = email["from"]["id"]
            if user_id == sender_id:
                print "Skipping user_id", user_id
                break
            reply_users = email["replied_from"]
            this_user_replied = True if user_id in reply_users else False
            num_questions = len(email["questions"])
            relative_connection_score = email["from"]["relative_score"]
            raw_connection_score = email["from"]["raw_score"]
            reply_rate = email["from"]["reply_rate"]
            feature_vec = np.array([num_questions, relative_connection_score, raw_connection_score, reply_rate])
            X[idx] = feature_vec
            Y[idx] = 1 if this_user_replied else 0

        print "X:"
        print X
        print "Y:"
        print Y





# {"user":"2927736207583065031","body_wordcount":49,"time":"2014-05-29 02:21:45","reply_desired":"no","is_mailing_list":"no","subject":"RE: Chicago Venture Summit  **Call for company nominations**","body":"Guy \u2013 Looks like a great event. Looking forward to it. Let us know how Jump can best be involved.\n\nWe\u2019ll think about companies to nominate and get back to you on that.\n\n\u200b\u200b\u200b\u200b\u200bThanks,\nPeter J. Johnson\nJump Capital LLC\n600 W. Chicago  Suite 825  |  Chicago, IL 60654\nOffice: 312.205.8390  |  Mobile: 763.656.7590\n\n[Description: JumpCapitalLogoSmall]\n\n","is_forward":"no","questions":["Let us know how Jump can best be involved."],"from":{"id":"14167754590843611851","relative_score":0,"raw_score":35,"reply_rate":1},"to":["14538670657664558777"],"is_to":"no","cc":["9065775327488645"],"is_cc":"no","is_bcc":"yes","is_reply":"no","replied_from":[]}



if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    person_reply = PersonReply()
    data_filename = sys.argv[1]
    person_reply.load_data(data_filename)
    
    pr_playground = PersonReplyPlayground(person_reply)
    # print pr_playground.subject_has_words("Hello World", ["Hello", "World", "Go"], relation="or")
    pr_playground.experiment_1()




















