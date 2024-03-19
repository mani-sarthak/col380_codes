part1: functions.cpp
	mkdir -p obj
	g++ functions.cpp -o obj/part1
	./obj/part1 > part1.txt

lenet:
	mkdir -p obj
	g++ -O2 lenet.cpp functions.cpp -o obj/lenet

img: lenet
	# python3 ./preprocess.py ./samples/2.png > ./img_matrix/sample_img_num2.txt
	./obj/lenet ./img_matrix/sample_img_num2.txt

clean:
	-rm obj/part1 part1.txt
