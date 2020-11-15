
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;


#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }


__global__ void mirror(unsigned char* input, unsigned char* output,
                            int num_of_pixels, int num_of_rows,
                            int num_of_cols)
{
	const int i = 3*(blockIdx.x * blockDim.x + threadIdx.x);
	if(i >= num_of_pixels) return;
	const int rowi = i/(3*num_of_cols);
	const int offset = i - rowi*(3*num_of_cols);
	const int mirror_i = rowi*(3*num_of_cols) + 3*num_of_cols - offset;
	if (i <= rowi*(3*num_of_cols) + 3*num_of_cols/2){
		output[i] = input[mirror_i];
		output[i + 1] = input[mirror_i + 1];
		output[i + 2] = input[mirror_i + 2];
		output[mirror_i] = input[i];
		output[mirror_i + 1] = input[i + 1];
		output[mirror_i + 2] = input[i + 2];
	}
}

int main( int argc, char** argv )
{
	/************************************************/

    Mat image1,image2;
    image1 = imread("cat4.jpg", CV_LOAD_IMAGE_COLOR);
	image2 = imread("cat4.jpg", CV_LOAD_IMAGE_COLOR);

    if(! image1.data && !image2.data)
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    namedWindow( "Display window INPUT", WINDOW_AUTOSIZE );
    imshow("Display window INPUT",image1);
	cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;
	cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

	/************************************************/

	startCPU = clock();
	for(int i = 0; i < image1.rows; i++)
    {
        Vec3b* p = image1.ptr<Vec3b>(i);
        for (int j = 0, reverse_j = image1.cols - 1; j < reverse_j; j++, reverse_j-- ){
            for (int ch = 0; ch < 3; ch++)
                swap(p[j][ch],p[reverse_j][ch]);
        }
    }
	elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
    cout << "CPU time = " << elapsedTimeCPU*1000 << " ms\n";

	/************************************************/

    unsigned char *d_input,*d_output;
    int num_of_pixels = 3*image2.rows*image2.cols;
    CHECK(cudaMalloc(&d_input, num_of_pixels));
	CHECK(cudaMalloc(&d_output, num_of_pixels));
    CHECK(cudaMemcpy(d_input, image2.data, num_of_pixels, cudaMemcpyHostToDevice));
	cudaEventRecord(startCUDA,0);
    mirror<<<(num_of_pixels/3 + 255)/256,256>>>(d_input,d_output,num_of_pixels,image2.rows,image2.cols);
	cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());
	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
    cout << "CUDA time = " << elapsedTimeCUDA << " ms\n";

	/************************************************/

    CHECK(cudaMemcpy(image2.data, d_output, num_of_pixels, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_input));
	CHECK(cudaFree(d_output));
	imwrite("outputCPU.jpg",image1);
    imwrite("outputCUDA.jpg",image2);
	//namedWindow( "Display window CPU", WINDOW_AUTOSIZE );
    namedWindow( "Display window CUDA", WINDOW_AUTOSIZE  );
	//imshow("Display window CPU", image1);
	imshow("Display window CUDA", image2);
    waitKey(0);
    return 0;
}
