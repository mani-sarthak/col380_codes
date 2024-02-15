CC = gcc
CXX = g++
.PHONY: test


sequential : ./src/sequential.cpp
	mkdir -p obj
	$(CXX) -o ./obj/sequential.out ./src/sequential.cpp
	./obj/sequential.out "./test/test_matrix.txt"

openmp : ./src/openmp.cpp
	mkdir -p obj
	$(CXX) -fopenmp -o ./obj/openmp.out ./src/openmp.cpp
	./obj/openmp.out "./test/test_matrix.txt" $(num)

pthread: ./src/pthread.cpp
	mkdir -p obj
	$(CXX) -pthread ./src/pthread.cpp -o ./obj/pthread.out
	./obj/pthread.out "./test/test_matrix.txt" $(num)

gen: ./test/generate.c
	mkdir -p obj
	$(CC) -o ./obj/gen.out ./test/generate.c
	./obj/gen.out $(size) ./test/test_matrix.txt

test:
	./scripts/run.sh

clean: 
	rm ./obj/*
