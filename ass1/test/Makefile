CC = g++
CFLAGS = -fopenmp

gen : generate.cpp
	$(CC) $(CFLAGS) -o gen.out generate.cpp
	./gen.out > ../test_matrix.txt