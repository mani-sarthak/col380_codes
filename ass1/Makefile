CC = g++
CFLAGS = -fopenmp


serial : ./src/serial.cpp
	$(CC) $(CFLAGS) -o ./obj/serial.out ./src/serial.cpp
	time ./obj/serial.out < ./test/test_matrix.txt

openmp : ./src/openmp.cpp
	$(CC) $(CFLAGS) -o ./obj/openmp.out ./src/openmp.cpp
	time ./obj/openmp.out "./test/test_matrix.txt" $(t)

pthread: ./src/pthread.cpp
	$(CC) $(CFLAGS) -pthread ./src/pthread.cpp -o ./obj/pthread.out
	time ./obj/pthread.out  ./test/test_matrix.txt $(num)

gen: ./test/generate.c
	gcc $(CFLAGS) -o ./obj/gen.out ./test/generate.c
	./obj/gen.out $(size) ./test/test_matrix.txt

clean: 
	rm ./out/*
