part1: functions.cpp
	mkdir -p obj
	g++ functions.cpp -o obj/part1
	./obj/part1 > part1.txt

lenet:
	mkdir -p obj
	g++ lenet.cpp functions.cpp -o obj/lenet

img:
	python3 ./preprocess.py ./test/000000-num7.png > ./img_matrix/sample_num7.txt

clean:
	-rm obj/part1 part1.txt
