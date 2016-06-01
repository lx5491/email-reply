DATA_FILE = data/email_data_100Klines.txt
D2V_TARGET = data/reply_300_20_1464729586.d2v
# DATA_FILE = ../email_data_1464276919_2.txt
# D2V_TARGET = reply_100_10_1464276919_2.d2v
D2V_EPOCH_RANGE = 20

d2v: doc2vec-reply.py $(DATA_FILE)
	python doc2vec-reply.py $(DATA_FILE) $(D2V_TARGET) $(D2V_EPOCH_RANGE)

view_d2v: doc2vec_reply/see_email_data.py $(DATA_FILE)
	python see_email_data.py $(DATA_FILE)

metric: doc2vec_reply/metric_svm.py $(DATA_FILE)
	python metric_svm.py $(DATA_FILE)

clean:
	rm $(D2V_TARGET)

clean_all:
	rm *.d2v