read_proto:
	g++ read_proto.cc -I. -pthread `pkg-config --cflags --libs protobuf` include/caffe.pb.cc -o read_proto

read_model:
	g++ read_model.cc -I./include -D_THREAD_SAFE -I/usr/local/Cellar/protobuf/2.6.1/include -L/usr/local/Cellar/protobuf/2.6.1/lib -lprotobuf -D_THREAD_SAFE -I/usr/local/include -L/usr/local/lib -lglog -pthread include/caffe.pb.cc -o read_model

blob.o:
	g++ -c  src/blob.cpp -I./include  -I/usr/local/include -L/usr/local/lib

clean:
	rm -f read_proto
	rm -f blob.o
