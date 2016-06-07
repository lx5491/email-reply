from sklearn.metrics import mutual_info_score

import json
import numpy as np
import re
import string
import sys
import os
import logging
import Queue as Q

class PersonReply(object):
    def __init__(self, logger):
        self.emails = []
        self.stopwords = dict({})
        self.logger = logger
        self.invert_idx = dict({})
        self.got_reply_array = []
        self.num_unique_words = 0

    def load_data(self, data_filename, stopwords_filename):
        count = 0
        with open(data_filename, "r") as in_file:
            for line in in_file:
                msg = dict({})
                msg = json.loads(line)
                self.emails.append(msg)
                count += 1
        self.logger.info("loaded %d emails" % len(self.emails))
        self.read_stopwords(stopwords_filename)
        self.basic_process()

    def read_stopwords(self, stopwords_filename):
        with open(stopwords_filename, "r") as sf:
            for line in sf:
                word = line.strip()
                self.stopwords[word] = 1

    def basic_process(self):
        for email in self.emails:
            email["subject"] = email["subject"].lower() if email["subject"] != None else ""
            email["body"] = email["body"].lower()
            remove_punctuation_map = dict((ord(char), ord(' ')) for char in string.punctuation)
            email["body"] = email["body"].translate(remove_punctuation_map)

    def build_inverted_index(self):
        for email_idx, email in enumerate(self.emails):
            email = self.emails[email_idx]
            body_text = email["body"]
            body_words = body_text.split()
            for word in body_words:
                if word in self.stopwords:
                    continue
                if word not in self.invert_idx:
                    self.num_unique_words += 1
                    self.invert_idx[word] = dict({email_idx:0})
                if email_idx not in self.invert_idx[word]:
                    self.invert_idx[word][email_idx] = 0
                self.invert_idx[word][email_idx] += 1

    def email_got_reply(self, email):
        user_id = email["user"]
        reply_user_ids = [person["id"] for person in email["replied_from"]]
        this_user_replied = True if user_id in reply_user_ids else False
        return this_user_replied

    def print_an_email(self, email):
        print ""
        print "============================================================================="
        print self.email_got_reply
        print email
        print "-----------------------------------------------------------------------------"
        print email["body"].encode('utf-8')
        print "============================================================================="
        print ""

    def build_got_reply_array(self):
        for email_idx, email in enumerate(self.emails):
            got_reply = self.email_got_reply(email)
            self.got_reply_array.append(1 if got_reply else 0)

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    person_reply = PersonReply(logger)
    data_filename = sys.argv[1]
    stopwords_filename = sys.argv[2]
    person_reply.load_data(data_filename, stopwords_filename)

    logger.info("person_reply has %d emails" % len(person_reply.emails))

    logger.info("building inverted index...")
    person_reply.build_inverted_index()
    logger.info("there are %d unique words" % person_reply.num_unique_words)
    logger.info("building reply array...")
    person_reply.build_got_reply_array()

    # word = "meeting"
    # print ">>>>>>>>>>>>>>>>>>>>>>>>>The word '%s' inverted index email text:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" % word
    # show_num = 10
    # show_count = 0
    # for email_idx in person_reply.invert_idx[word]:
    #     if show_count >= show_num:
    #         break
    #     person_reply.print_an_email(person_reply.emails[email_idx])
    #     show_count += 1

    # word_vec = np.zeros(len(person_reply.emails))
    # for email_idx in person_reply.invert_idx[word]:
    #     word_vec[email_idx] = 1
    # mi_score = mutual_info_score(word_vec, person_reply.got_reply_array)
    # logger.info("The word '%s''s mi score: %f" % (word, mi_score))

    logger.info("calculating mutual information score...")
    mi_queue = Q.PriorityQueue()
    word_count_stop = None
    word_count = 0
    for word in person_reply.invert_idx:
        if word_count != 0 and word_count % 1000 == 0:
            logger.info("processed %d words" % word_count)
        if word_count_stop and word_count >= word_count_stop:
            break
        word_vec = np.zeros(len(person_reply.emails))
        for email_idx in person_reply.invert_idx[word]:
            word_vec[email_idx] = 1
        mi_score = mutual_info_score(word_vec, person_reply.got_reply_array)
        mi_queue.put((-mi_score, word))
        word_count += 1

    show_count_words = 500
    logger.info("This is top %d words according to MI:" % show_count_words)
    

    with open("mi_queue.txt", "w+") as mq_file:
        for i in range(show_count_words):
            line = str(mi_queue.get())
            mq_file.write(line + "\n")
            logger.info(line)














