DATA_FILE = data/email_data_100Klines.txt
D2V_TARGET = data/reply_300_20_1464729586.d2v
# DATA_FILE = ../email_data_1464276919_2.txt
# D2V_TARGET = reply_100_10_1464276919_2.d2v
D2V_EPOCH_RANGE = 20

# PERSONAL_REPLY_DATA_FILE = data/email_person_reply_12345.txt

OS := $(shell uname)
ifeq ($(OS), Darwin)
# Run MacOS commands 
	PERSONAL_REPLY_DATA_FILE = data/email_person_reply_12345.txt
else
# check for Linux and run other commands
	PERSONAL_REPLY_DATA_FILE = data/email_data_1464894150.txt
endif

d2v: doc2vec_reply/doc2vec-reply.py $(DATA_FILE)
	python doc2vec_reply/doc2vec-reply.py $(DATA_FILE) $(D2V_TARGET) $(D2V_EPOCH_RANGE)

view_d2v: doc2vec_reply/see_email_data.py $(DATA_FILE)
	python doc2vec_reply/see_email_data.py $(DATA_FILE)

metric: metric_svm.py $(DATA_FILE)
	python metric_svm.py $(DATA_FILE)

person_reply: personal_reply/personal_reply.py $(PERSONAL_REPLY_DATA_FILE)
	python personal_reply/personal_reply.py $(PERSONAL_REPLY_DATA_FILE)

clean:
	rm $(D2V_TARGET)

clean_all:
	rm *.d2v