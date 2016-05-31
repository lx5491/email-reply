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
import os.path

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class TaggedLineSentence(object):
    def __init__(self, source):
        self.source = source
        self.url_re_matcher = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_re_matcher = re.compile('[^@]+@[^@]+\.[^@]+')

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
    def remove_bad_words_and_make_lower(self, bag_of_words):
        new_bag_of_words = []
        for word in bag_of_words:
            if word not in self.stop_words and len(word) <= 30 and self.url_re_matcher.match(word) == None \
                and self.email_re_matcher.match(word) == None:
                word = word.encode('utf-8') # change unicode object to normal string
                word = word.translate(string.maketrans("", ""), string.punctuation)
                word = word.lower()
                word = word.decode('utf-8')
                new_bag_of_words.append(word)

        return new_bag_of_words

    def make_clean(self):
        self.num_emails_yes = 0
        self.num_emails_no = 0
        for idx, _ in enumerate(self.emails):
            body_text = self.emails[idx]["body"]
            del self.emails[idx]["body"]
            body_words_lst = body_text.split()
            body_words_lst = self.remove_bad_words_and_make_lower(body_words_lst)
            self.emails[idx]["clean_words"] = body_words_lst
            if self.emails[idx]["got_reply"] == "yes":
                self.num_emails_yes += 1
            else:
                self.num_emails_no += 1

    def to_array(self):
        if not hasattr(self, 'sentences') or not self.sentences:
            self.sentences = []
            yes_count = 0
            no_count = 0
            for email_no, email in enumerate(self.emails):
                label = ""
                if email["got_reply"] == "yes":
                    prefix = "YES"
                    label = prefix + '_%s' % str(yes_count)
                    print "label:", label
                    yes_count += 1
                else:
                    prefix = "NO"
                    label = prefix + '_%s' % str(no_count)
                    print "label:", label
                    no_count += 1
                self.sentences.append(TaggedDocument(email["clean_words"], [label]))

        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

logger.info("program starts")
source = sys.argv[1]
sentences = TaggedLineSentence(source)

logger.info("loading source: %s" % source)
sentences.load_data(source)
logger.info("finished loading source")

logger.info("making words in emails clean")
sentences.make_clean()

logger.info("D2V")
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

d2v_filename = sys.argv[2]
if not os.path.isfile(d2v_filename):
    logger.info('Epoch')
    for epoch in range(10):
        logger.info('EPOCH: {}'.format(epoch))
        model.train(sentences.sentences_perm())
    logger.info('Model Save')
    model.save('./reply.d2v')
else:
    logger.info("D2V file found, no need to train")

model = Doc2Vec.load('./reply.d2v')

num_total_emails = sentences.num_emails_yes + sentences.num_emails_no
email_arrays = np.zeros((num_total_emails, 100))
email_labels = np.zeros(num_total_emails)

for i in range(sentences.num_emails_yes):
    sentence_label = "YES_%s" % i
    email_arrays[i] = model.docvecs[sentence_label]
    email_labels[i] = 1

for i in range(sentences.num_emails_no):
    sentence_label = "NO_%s" % i
    email_arrays[i + sentences.num_emails_yes] = model.docvecs[sentence_label]
    email_labels[i + sentences.num_emails_yes] = 0

logger.info('Fitting')
classifier = LogisticRegression()
classifier.fit(email_arrays, email_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.score(email_arrays, email_labels) # using train to test train accuracy


























