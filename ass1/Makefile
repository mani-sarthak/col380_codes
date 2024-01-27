CC = g++
CFLAGS = -fopenmp -pthread
.PHONY: test


serial : ./src/serial.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) -o ./obj/serial.out ./src/serial.cpp
	./obj/serial.out "./test/test_matrix.txt"

openmp : ./src/openmp.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) -o ./obj/openmp.out ./src/openmp.cpp
	./obj/openmp.out "./test/test_matrix.txt" $(num)

pthread: ./src/pthread.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) ./src/pthread.cpp -o ./obj/pthread.out
	./obj/pthread.out "./test/test_matrix.txt" $(num)

gen: ./test/generate.c
	mkdir -p obj
	gcc -o ./obj/gen.out ./test/generate.c
	./obj/gen.out $(size) ./test/test_matrix.txt

test:
	./scripts/run.sh

clean: 
	rm ./obj/*
