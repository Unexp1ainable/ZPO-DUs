all:
	g++ -o xrepka07 tasks/mt01.cpp -Wall -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

clean:
	rm xrepka07