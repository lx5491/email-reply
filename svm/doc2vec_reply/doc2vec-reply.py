# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy as np

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys
import os
import json
import re
import string

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class TaggedLineSentence(object):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def load_data(self, data_filename, stopword_filename="../stopwords.txt"):
        self.emails = []
        count = 0
        with open(data_filename, "r") as in_file:
            for line in in_file:
                # if count == 200:
                #     break
                msg = dict({})
                msg = json.loads(line)
                self.emails.append(msg)
                count += 1

        self.stop_words = dict({})
        with open(stopword_filename, "r") as sf:
            for line in sf:
                word = line.strip()
                self.stop_words[word] = 1

    # bad words are stop words, a too-lengthy word, urls, and email addresses
    def remove_bad_words_and_make_lower(bag_of_words):
        new_bag_of_words = []
        for word in bag_of_words:
            if word not in stop_words and len(word) <= 20 and url_re_matcher.match(word) == None \
                and email_re_matcher.match(word) == None:
                word = word.encode('utf-8') # change unicode object to normal string
                word = word.translate(string.maketrans("", ""), string.punctuation)
                word = word.lower()
                new_bag_of_words.append(word)

        return new_bag_of_words

    def make_clean(self):
        for idx, _ in enumerate(self.emails):
            body_text = self.emails[idx]["body"]
            body_words_lst = body_text.split()
            body_words_lst = self.remove_bad_words_and_make_lower(body_words_lst)


    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

logger.info("program starts")
source = "../email_data_text.txt"
sentences = TaggedLineSentence(source)
logger.info("loading source: %s" % source)
sentences.load_data("../email_data_text_big.txt")
logger.info("finished loading source")


























