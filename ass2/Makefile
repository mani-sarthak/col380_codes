part1: functions.cpp
	mkdir -p obj
	g++ functions.cpp -o obj/part1
	./obj/part1 > part1.txt

lenet:
	mkdir -p obj
	g++ lenet.cpp functions.cpp -o obj/lenet
	./obj/lenet > output.txt

img:
	python3 ./preprocess.py ./2.png > img.txt
	make lenet
	./obj/lenet

ch: check.cpp
	mkdir -p obj
	g++ check.cpp -o obj/check
	./obj/check > output.txt


clean:
	-rm obj/part1 part1.txt
