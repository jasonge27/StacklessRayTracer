nvcc -arch=sm_21  -O3 -I /usr/include/opencv -I /usr/include/opencv2 main.cpp scene.cpp kernel.cu -lopencv_core -lopencv_highgui -o raytracer
