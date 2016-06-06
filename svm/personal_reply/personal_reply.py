
import numpy as np

from random import shuffle

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import sys
import os
import logging

class PersonReply(object):
    def __init__(self, logger):
        self.emails = []
        self.logger = logger

    def load_data(self, data_filename):
        count = 0
        with open(data_filename, "r") as in_file:
            for line in in_file:
                msg = dict({})
                msg = json.loads(line)
                self.emails.append(msg)
                count += 1
        self.logger.info("loaded %d emails" % len(self.emails))
        # self.basic_process()

    def basic_process(self):
        for email in self.emails:
            email["subject"] = email["subject"].lower() if email["subject"] != None else ""
            email["body"] = email["body"].lower()


class PersonReplyPlayground(object):
    def __init__(self, person_reply, logger):
        self.person_reply = person_reply
        self.logger = logger

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
        # X = np.zeros((len(self.person_reply.emails), 4), dtype=float)
        # Y = np.zeros(len(self.person_reply.emails))
        X_pos = np.empty([0, 4])
        X_neg = np.empty([0, 4])
        Y_pos = np.empty([0])
        Y_neg = np.empty([0])
        # print self.person_reply.emails
        count = 0
        for email in self.person_reply.emails:
            user_id = email["user"]
            sender_id = email["from"]["id"]
            if user_id == sender_id:
                self.logger.info("Skipping user_id %s" % user_id)
                continue
            reply_user_ids = [person["id"] for person in email["replied_from"]]
            this_user_replied = True if user_id in reply_user_ids else False
            num_questions = len(email["questions"]) if email["questions"] else 0
            relative_connection_score = email["from"]["relative_score"]
            raw_connection_score = email["from"]["raw_score"]
            reply_rate = email["from"]["reply_rate"]
            feature_vec = np.array([[num_questions, relative_connection_score, raw_connection_score, reply_rate]])
            label_vec= np.array([1 if this_user_replied else 0])
            if label_vec == 0:
                X_neg = np.concatenate((X_neg, feature_vec), axis=0)
                Y_neg = np.concatenate((Y_neg, label_vec), axis=0)
            else:
                X_pos = np.concatenate((X_pos, feature_vec), axis=0)
                Y_pos = np.concatenate((Y_pos, label_vec), axis=0)

            count += 1

        train_pos_arrays, test_pos_arrays, train_pos_labels, test_pos_labels = train_test_split(X_pos, Y_pos, test_size=0.2, random_state=42)
        train_neg_arrays, test_neg_arrays, train_neg_labels, test_neg_labels = train_test_split(X_neg, Y_neg, test_size=0.2, random_state=42)
        train_arrays = np.concatenate((train_pos_arrays, train_neg_arrays), axis=0)
        train_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)
        test_arrays = np.concatenate((test_pos_arrays, test_neg_arrays), axis=0)
        test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)


        self.logger.info("train_arrays shape: %s" % str(train_arrays.shape))
        self.logger.info("test_arrays shape: %s" % str(test_arrays.shape))
        self.logger.info("train_labels shape: %s" % str(train_labels.shape))
        self.logger.info("test_labels shape: %s" % str(test_labels.shape))


        param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 100]}
        clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
        # classifier = SVC()
        clf = GridSearchCV(clf, param_grid, n_jobs=1, scoring="precision")


        logger.info('Fitting')
        clf.fit(train_arrays, train_labels)
        logger.info("finished fitting")
        test_predict = clf.predict(test_arrays)
        logger.info(np.count_nonzero(test_predict))
        logger.info("finished predicting %s" % test_predict.shape)


        precision = metrics.precision_score(test_labels, test_predict)
        recall = metrics.recall_score(test_labels, test_predict)
        self.logger.info("LogisticRegression -- Precision on test data: %0.4f" % precision)
        self.logger.info("LogisticRegression -- Recall on test data: %0.4f" % recall)

    def experiment_2(self):
        # X = np.zeros((len(self.person_reply.emails), 4), dtype=float)
        # Y = np.zeros(len(self.person_reply.emails))
        X_pos = np.empty([0, 4])
        X_neg = np.empty([0, 4])
        Y_pos = np.empty([0])
        Y_neg = np.empty([0])
        # print self.person_reply.emails
        count = 0
        for email in self.person_reply.emails:
            user_id = email["user"]
            sender_id = email["from"]["id"]
            if user_id == sender_id:
                self.logger.info("Skipping user_id %s" % user_id)
                continue
            reply_user_ids = [person["id"] for person in email["replied_from"]]
            this_user_replied = True if user_id in reply_user_ids else False
            num_questions = len(email["questions"]) if email["questions"] else 0
            relative_connection_score = email["from"]["relative_score"]
            raw_connection_score = email["from"]["raw_score"]
            reply_rate = email["from"]["reply_rate"]
            feature_vec = np.array([[num_questions, relative_connection_score, raw_connection_score, reply_rate]])
            label_vec= np.array([1 if this_user_replied else 0])
            if label_vec == 0:
                X_neg = np.concatenate((X_neg, feature_vec), axis=0)
                Y_neg = np.concatenate((Y_neg, label_vec), axis=0)
            else:
                X_pos = np.concatenate((X_pos, feature_vec), axis=0)
                Y_pos = np.concatenate((Y_pos, label_vec), axis=0)

            count += 1

        train_pos_arrays, test_pos_arrays, train_pos_labels, test_pos_labels = train_test_split(X_pos, Y_pos, test_size=0.2, random_state=42)
        train_neg_arrays, test_neg_arrays, train_neg_labels, test_neg_labels = train_test_split(X_neg, Y_neg, test_size=0.2, random_state=42)
        train_arrays = np.concatenate((train_pos_arrays, train_neg_arrays), axis=0)
        train_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)
        test_arrays = np.concatenate((test_pos_arrays, test_neg_arrays), axis=0)
        test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)


        self.logger.info("train_arrays shape: %s" % str(train_arrays.shape))
        self.logger.info("test_arrays shape: %s" % str(test_arrays.shape))
        self.logger.info("train_labels shape: %s" % str(train_labels.shape))
        self.logger.info("test_labels shape: %s" % str(test_labels.shape))


        param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 50], 'gamma': [0.01, 0.05, 0.1, 0.5, 1]}
        clf = SVC(kernel='rbf')
        # classifier = SVC()
        clf = GridSearchCV(clf, param_grid, n_jobs=5, scoring="precision")


        logger.info('Fitting')
        clf.fit(train_arrays, train_labels)
        logger.info("finished fitting")
        test_predict = clf.predict(test_arrays)
        logger.info(np.count_nonzero(test_predict))
        logger.info("finished predicting %s" % test_predict.shape)


        precision = metrics.precision_score(test_labels, test_predict)
        recall = metrics.recall_score(test_labels, test_predict)
        self.logger.info("SVM Classification -- Precision on test data: %0.4f" % precision)
        self.logger.info("SVM Classification -- Recall on test data: %0.4f" % recall)

    def experiment_3(self):
        # X = np.zeros((len(self.person_reply.emails), 4), dtype=float)
        # Y = np.zeros(len(self.person_reply.emails))
        X_pos = np.empty([0, 4])
        X_neg = np.empty([0, 4])
        Y_pos = np.empty([0])
        Y_neg = np.empty([0])
        # print self.person_reply.emails
        count = 0
        for email in self.person_reply.emails:
            user_id = email["user"]
            sender_id = email["from"]["id"]
            if user_id == sender_id:
                self.logger.info("Skipping user_id %s" % user_id)
                continue
            reply_user_ids = [person["id"] for person in email["replied_from"]]
            this_user_replied = True if user_id in reply_user_ids else False
            num_questions = len(email["questions"]) if email["questions"] else 0
            relative_connection_score = email["from"]["relative_score"]
            raw_connection_score = email["from"]["raw_score"]
            reply_rate = email["from"]["reply_rate"]
            feature_vec = np.array([[num_questions, relative_connection_score, raw_connection_score, reply_rate]])
            label_vec= np.array([1 if this_user_replied else 0])
            if label_vec == 0:
                X_neg = np.concatenate((X_neg, feature_vec), axis=0)
                Y_neg = np.concatenate((Y_neg, label_vec), axis=0)
            else:
                X_pos = np.concatenate((X_pos, feature_vec), axis=0)
                Y_pos = np.concatenate((Y_pos, label_vec), axis=0)

            count += 1

        train_pos_arrays, test_pos_arrays, train_pos_labels, test_pos_labels = train_test_split(X_pos, Y_pos, test_size=0.2, random_state=42)
        train_neg_arrays, test_neg_arrays, train_neg_labels, test_neg_labels = train_test_split(X_neg, Y_neg, test_size=0.2, random_state=42)
        train_arrays = np.concatenate((train_pos_arrays, train_neg_arrays), axis=0)
        train_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)
        test_arrays = np.concatenate((test_pos_arrays, test_neg_arrays), axis=0)
        test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)


        self.logger.info("train_arrays shape: %s" % str(train_arrays.shape))
        self.logger.info("test_arrays shape: %s" % str(test_arrays.shape))
        self.logger.info("train_labels shape: %s" % str(train_labels.shape))
        self.logger.info("test_labels shape: %s" % str(test_labels.shape))


        clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)

        logger.info('Fitting')
        clf.fit(train_arrays, train_labels)
        logger.info("finished fitting")
        test_predict = clf.predict(test_arrays)
        logger.info(np.count_nonzero(test_predict))
        logger.info("finished predicting %s" % test_predict.shape)


        precision = metrics.precision_score(test_labels, test_predict)
        recall = metrics.recall_score(test_labels, test_predict)
        self.logger.info("Decision Tree Classification -- Precision on test data: %0.4f" % precision)
        self.logger.info("Decision Tree Classification -- Recall on test data: %0.4f" % recall)

    #draw the relationship between connection_score and reply_rate
    def experiment_4(self):

        count = 0
        relative_scores = []
        reply_rates = []
        for email in self.person_reply.emails:
            user_id = email["user"]
            sender_id = email["from"]["id"]
            if user_id == sender_id:
                self.logger.info("Skipping user_id %s" % user_id)
                continue
            reply_user_ids = [person["id"] for person in email["replied_from"]]
            this_user_replied = True if user_id in reply_user_ids else False
            num_questions = len(email["questions"]) if email["questions"] else 0
            
            relative_connection_score = email["from"]["relative_score"]

            raw_connection_score = email["from"]["raw_score"]
            
            reply_rate = email["from"]["reply_rate"]

            if relative_connection_score and reply_rate and count < 2000:
                relative_scores.append(relative_connection_score)
                reply_rates.append(reply_rate)

            count += 1

        relative_scores = np.array(relative_scores)
        self.logger.info("min score: %0.4f" % np.amin(relative_scores))
        self.logger.info("max score: %0.4f" % np.amax(relative_scores))
        self.logger.info("mean score: %0.4f" % np.mean(relative_scores))

        reply_rates = np.array(reply_rates)
        self.logger.info("min score: %0.4f" % np.amin(reply_rates))
        self.logger.info("max score: %0.4f" % np.amax(reply_rates))
        self.logger.info("mean score: %0.4f" % np.mean(reply_rates))

        self.logger.info("relative score shape: %s" % str(relative_scores.shape))
        self.logger.info("reply rates shape: %s" % str(reply_rates.shape))
        plt.scatter(relative_scores, reply_rates)
        plt.savefig("score_reply.png")

    #show some examples of replied emails and unreplied emails
    def experiment_5(self):
        num_replied = 30
        num_unreplied = 30
        emails_replied = []
        emails_unreplied = []
        for email in self.person_reply.emails:
            user_id = email["user"]
            sender_id = email["from"]["id"]
            if user_id == sender_id:
                self.logger.info("Skipping user_id %s" % user_id)
                continue
            reply_user_ids = [person["id"] for person in email["replied_from"]]
            this_user_replied = True if user_id in reply_user_ids else False
            if this_user_replied and len(emails_replied) < num_replied:
                emails_replied.append(email)
            elif not this_user_replied and len(emails_unreplied) < num_unreplied:
                emails_unreplied.append(email)

        self.logger.info("There are %d replied emails and %d unreplied emails" % (len(emails_replied), len(emails_unreplied)))
        questions_replied = np.count_nonzero([len(email["questions"]) if email["questions"] else 0 for email in emails_replied])
        self.logger.info("the number of replied emails that have questions: %d" % questions_replied)
        questions_unreplied = np.count_nonzero([len(email["questions"]) if email["questions"] else 0 for email in emails_unreplied])
        self.logger.info("the number of unreplied emails that have questions: %d" %questions_unreplied)

        # examples of replied emails
        # for email in emails_replied:
        #     print ""
        #     print "============================================================================="
        #     print email
        #     print "-----------------------------------------------------------------------------"
        #     print email["body"].encode('utf-8')
        #     print "============================================================================="
        #     print ""

        # examples of replied emails without a question:
        # for email in emails_replied:
        #     if email["questions"] and len(email["questions"]) > 0:
        #         continue
        #     print ""
        #     print "============================================================================="
        #     print email
        #     print "-----------------------------------------------------------------------------"
        #     print email["body"].encode('utf-8')
        #     print "============================================================================="
        #     print ""

        # examples of unreplied emails with a question:
        for email in emails_unreplied:
            if not email["questions"] or len(email["questions"]) == 0:
                continue
            print ""
            print "============================================================================="
            print email
            print "-----------------------------------------------------------------------------"
            print email["body"].encode('utf-8')
            print "============================================================================="
            print ""

        # examples of unreplied emails with a question:
        # for email in emails_unreplied:
        #     if email["questions"] and len(email["questions"]) > 0:
        #         continue
        #     print ""
        #     print "============================================================================="
        #     print email
        #     print "-----------------------------------------------------------------------------"
        #     print email["body"].encode('utf-8')
        #     print "============================================================================="
        #     print ""




if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    person_reply = PersonReply(logger)
    data_filename = sys.argv[1]
    person_reply.load_data(data_filename)
    
    logger.info("person_reply has %d emails" % len(person_reply.emails))

    pr_playground = PersonReplyPlayground(person_reply, logger)
    # pr_playground.experiment_1()
    # pr_playground.experiment_2()
    # pr_playground.experiment_3()
    # pr_playground.experiment_4()
    pr_playground.experiment_5()





















