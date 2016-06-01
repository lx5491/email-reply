DATA_FILE = ./email_data_1464729586.txt
D2V_TARGET = reply_300_20_1464729586.d2v
# DATA_FILE = ../email_data_1464276919_2.txt
# D2V_TARGET = reply_100_10_1464276919_2.d2v
EPOCH_RANGE = 20

all: doc2vec-reply.py $(DATA_FILE)
	python doc2vec-reply.py $(DATA_FILE) $(D2V_TARGET) $(EPOCH_RANGE)

view: see_email_data.py $(DATA_FILE)
	python see_email_data.py $(DATA_FILE)

clean:
	rm $(D2V_TARGET)

clean_all:
	rm *.d2v