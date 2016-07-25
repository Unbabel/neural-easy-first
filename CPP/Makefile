OBJS =
CC = g++
INCLUDES = -I. -I./Eigen
LIBS = -L/usr/local/lib
DEBUG = -g
CFLAGS = -std=c++11 -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 -fPIC $(INCLUDES)
LFLAGS = $(LIBS)

all: attentive_quality_estimator attentive_quality_estimator_test

attentive_quality_estimator_test: $(OBJS) attentive_quality_estimator_test.o
	$(CC) $(OBJS) attentive_quality_estimator_test.o $(LFLAGS) -o attentive_quality_estimator_test

attentive_quality_estimator_test.o: attentive_quality_estimator_test.cpp lookup_layer.h linear_layer.h rnn_layer.h attention_layer.h alignment_layer.h concatenator_layer.h feedforward_layer.h softmax_layer.h layer.h
	$(CC) $(CFLAGS) attentive_quality_estimator_test.cpp

attentive_quality_estimator: $(OBJS) attentive_quality_estimator.o
	$(CC) $(OBJS) attentive_quality_estimator.o $(LFLAGS) -o attentive_quality_estimator

attentive_quality_estimator.o: attentive_quality_estimator.cpp attentive_quality_estimator.h neural_network.h data.h nn_utils.h utils.h lookup_layer.h linear_layer.h rnn_layer.h attention_layer.h alignment_layer.h concatenator_layer.h feedforward_layer.h softmax_layer.h layer.h
	$(CC) $(CFLAGS) attentive_quality_estimator.cpp

clean:
	rm -f *.o *~ attentive_quality_estimator

