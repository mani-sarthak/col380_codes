CC = g++
CFLAGS = -fopenmp


serial : ./src/serial.cpp
	$(CC) $(CFLAGS) -o ./obj/serial.out ./src/serial.cpp
	time ./obj/serial.out < ./test/test_matrix.txt

openmp : ./src/openmp.cpp
	$(CC) $(CFLAGS) -o ./obj/openmp.out ./src/openmp.cpp
	time ./obj/openmp.out "./test/test_matrix.txt" $(t)

check : ./src/check.cpp
	$(CC) $(CFLAGS) -o ./obj/check.out ./src/check.cpp
	./obj/check.out < ./test/test_matrix.txt

gen: ./test/generate.c
	gcc $(CFLAGS) -o ./obj/gen.out ./test/generate.c
	./obj/gen.out $(size) ./test/test_matrix.txt

yash: ./src/yash.cpp
	$(CC) $(CFLAGS) -o ./obj/yash.out ./src/yash.cpp
	time ./obj/yash.out $(size) $(num) ./test/test_matrix.txt 1

purva: ./src/purva.c
	gcc $(CFLAGS) -o ./obj/purva.out ./src/purva.c
	time ./obj/purva.out $(size) ./test/test_matrix.txt $(num)

clean: 
	rm ./out/*
