from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import json

def grab_data(filename):
    emails = []
    with open(filename, "r") as in_file:
        for line in in_file:
            msg = dict({})
            msg = json.loads(line)
            emails.append(msg)

    return emails

def transform_email_dict_to_vectors(email_dict):
    x = []
    x.append(email_dict["body_length"])
    x.append(email_dict["cc_num"])
    x.append(1 if email_dict["reply_desired"] == "yes" else -1)
    x.append(1 if email_dict["is_mailing_list"] == "yes" else -1)
    x.append(1 if email_dict["is_forward"] == "yes" else -1)
    x.append(1 if email_dict["is_mailing_list"] == "yes" else -1)
    x.append(1 if email_dict["is_to"] == "yes" else -1)
    x.append(1 if email_dict["is_cc"] == "yes" else -1)
    x.append(1 if email_dict["is_bcc"] == "yes" else -1)
    x.append(1 if email_dict["is_reply"] == "yes" else -1)

    y = 1 if email_dict["got_reply"] == "yes" else -1
    return x, y

def preprocess(emails):
    X = []
    Y = []
    min_body_length = 1e10
    max_body_length = -1
    min_cc_num = 1e10
    max_cc_num = -1
    for email in emails:
        x, y = transform_email_dict_to_vectors(email)
        if email["body_length"] > max_body_length:
            max_body_length = email["body_length"]
        if email["body_length"] < min_body_length:
            min_body_length = email["body_length"]
        if email["cc_num"] > max_cc_num:
            max_cc_num = email["cc_num"]
        if email["cc_num"] < min_cc_num:
            min_cc_num = email["cc_num"]
        X.append(x)
        Y.append(y)

    body_s = max_body_length + min_body_length
    body_d = max_body_length - min_body_length
    cc_s = max_cc_num + min_cc_num
    cc_d = max_cc_num - min_cc_num
    for i in xrange(len(X)):
        X[i][0] = float(2 * X[i][0] - body_s) / body_d
        X[i][1] = float(2 * X[i][1] - cc_s) / cc_d

    print "max_body_length:", max_body_length
    print "min_body_length", min_body_length
    print "max_cc_num:", max_cc_num
    print "min_cc_num:", min_cc_num

    return X, Y

def metric_predict(y_true, y_predict):
    metric = dict({})
    correct = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in xrange(len(y_true)):
        if y_true[i] == y_predict[i] and y_predict[i] == 1:
            true_positive += 1
        elif y_true[i] == y_predict[i] and y_predict[i] == -1:
            true_negative += 1
        elif y_true[i] != y_predict[i] and y_predict[i] == 1:
            false_positive += 1
        elif y_true[i] != y_predict[i] and y_predict[i] == -1:
            false_negative += 1
    print true_positive, true_negative, false_positive, false_negative
    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive) / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)
    metric = {"precision":precision, "recall":recall, "f1_score":f1_score, \
        "tp":true_positive, "fp":false_positive, "tn":true_negative, "fn":false_negative}
    return metric

if __name__ == "__main__":
    emails = grab_data("email_data2.txt")
    print "There are", len(emails), "emails"
    train_num = 1500
    test_num = len(emails) - train_num

    X, Y = preprocess(emails)
    print X[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=24)
    print "Train size:", len(y_train)
    print "Test size:", len(y_test)

    t0 = time()
    param_grid = {'C': [1000, 1500, 2000],
              'gamma': [0.005, 0.1, 0.5, 1, 2, 5, 10]}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    # clf = SVC(C=0.1, gamma=0.005, class_weight='balanced')
    clf = clf.fit(X_train, y_train)
    y_predict_train = clf.predict(X_train)
    y_predict = clf.predict(X_test)
    print "Best estimator:", clf.best_estimator_
    print "Done in %0.3f secs" % (time() - t0)

    # print "y_train:", y_train
    # print "y_predict_train", y_predict_train

    metric_train = metric_predict(y_train, y_predict_train)
    metric_test = metric_predict(y_test, y_predict)

    train_precision = metric_train["precision"]
    train_recall = metric_train["recall"]
    train_f1_score = metric_train["f1_score"]
    print "Train precision:", metric_train["tp"], "/", (metric_train["tp"] + metric_train["fp"]), ",", train_precision
    print "Train recall: ", metric_train["tp"], "/", (metric_train["tp"] + metric_train["fn"]), ",", train_recall
    print "Train F1 score:", train_f1_score

    test_precision = metric_test["precision"]
    test_recall = metric_test["recall"]
    test_f1_score = metric_test["f1_score"]
    print "Test precision:", metric_test["tp"], "/", (metric_test["tp"] + metric_test["fp"]), ",", test_precision
    print "Test recall: ", metric_test["tp"], "/", (metric_test["tp"] + metric_test["fn"]), ",", test_recall
    print "Test F1 score:", test_f1_score
















