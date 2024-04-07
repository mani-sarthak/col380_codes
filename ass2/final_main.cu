
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
// #include <cuda_runtime.h>
#include "cuda_runtime.h"
#include <limits>
#include <cfloat>
#include <cassert>
#include "device_launch_parameters.h"
#include<algorithm>
#include<cmath>
#include<time.h>
// #include <filesystem>

using namespace std;


#define data_type float

string conv1_filename = "./trained_weights/conv1.txt";
string conv2_filename = "./trained_weights/conv2.txt";

string fc1_filename = "./trained_weights/fc1.txt";
string fc2_filename = "./trained_weights/fc2.txt";

float* conv1_weights, *conv2_weights, *fc1_weights, *fc2_weights;
float* conv1_bias, *conv2_bias, *fc1_bias, *fc2_bias;

float* conv1_weights_cuda, *conv2_weights_cuda, *fc1_weights_cuda, *fc2_weights_cuda;
float* conv1_bias_cuda, *conv2_bias_cuda, *fc1_bias_cuda, *fc2_bias_cuda;

float *d_layer0, *d_layer1, *d_layer2, *d_layer3, *d_layer4, *d_layer5, *d_layer6;
float *h_layer6;

__global__ void  convolve3d1kernel(float* input,float* kernel, float* output,int inputsize,int kernelsize,int depth,float*bias){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int out = (inputsize - kernelsize+1);
    if(i<out && j<out && k<depth){
        float sum = 0.0;
        for(int x=0;x<kernelsize;x++){
            for(int y=0;y<kernelsize;y++){
                sum += input[(i+x)*inputsize + (j+y)] * kernel[k*kernelsize*kernelsize + x*kernelsize + y];
            }
        }
        output[k*out*out + i*out + j] = (sum + bias[k]);
    }
}

void convolve3d1(int inputsize, int kernelsize, int depth){
    int outputsize = inputsize - kernelsize + 1;

    dim3 grid(1,1,depth);
    dim3 block(outputsize,outputsize,1);
    convolve3d1kernel<<<grid,block>>>(d_layer0,conv1_weights_cuda,d_layer1,inputsize,kernelsize,depth,conv1_bias_cuda);

}

__global__ void initialize_bias_kernel(float* input, float* bias, int mat_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = bias[idx / mat_size];
    }
}

void initialize_bias(float* input, float* bias, int mat_size, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    initialize_bias_kernel<<<numBlocks, blockSize>>>(input, bias, mat_size, size);
}

__global__ void convolve3d2kernel(float* input,float* kernel, float* output,int inputsize,int kernelsize,int depth,int channels){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int out = (inputsize - kernelsize+1);
    int s_depth = k%depth;
    int s_channel = k/depth;

    if(i<out && j<out && s_channel<channels){
        float sum = 0.0;
        for(int x=0;x<kernelsize;x++){
            for(int y=0;y<kernelsize;y++){
                sum += input[(i+x)*inputsize + (j+y) + s_depth*inputsize*inputsize] * kernel[s_channel*kernelsize*kernelsize*depth + s_depth*kernelsize*kernelsize + x*kernelsize + y];
            }
        }
        atomicAdd(&output[s_channel*out*out + i*out + j], sum);
    }
}

void convolve3d2(int inputsize, int kernelsize, int depth,int channels, int layer_num){

    int outputsize = inputsize - kernelsize + 1;

    dim3 grid(1,1,depth*channels);
    dim3 block(outputsize,outputsize,1);
    if(layer_num == 3){
        initialize_bias(d_layer3, conv2_bias_cuda, 8 * 8, 8 * 8 * 50);
        convolve3d2kernel<<<grid,block>>>(d_layer2,conv2_weights_cuda,d_layer3,inputsize,kernelsize,depth,channels);
    }else if(layer_num == 5){
        initialize_bias(d_layer5, fc1_bias_cuda, 1 * 1, 1 * 1 * 500);
        convolve3d2kernel<<<grid,block>>>(d_layer4,fc1_weights_cuda,d_layer5,inputsize,kernelsize,depth,channels);
    }else if(layer_num == 6){
        initialize_bias(d_layer6, fc2_bias_cuda, 1 * 1, 1 * 1 * 10);
        convolve3d2kernel<<<grid,block>>>(d_layer5,fc2_weights_cuda,d_layer6,inputsize,kernelsize,depth,channels);
    }

}


__global__ void pool3dkernel(float* input, float*output, int inputsize, int poolsize, int depth){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int out = inputsize/poolsize;
    if(i<out && j<out && k<depth){
        float maxval = -FLT_MAX;
        for(int x=0;x<poolsize;x++){
            for(int y=0;y<poolsize;y++){
                maxval = fmax(maxval,input[(i*poolsize+x)*inputsize + (j*poolsize+y) + k*inputsize*inputsize]);
            }
        }
        output[k*out*out + i*out + j] = maxval;
    }
}

void pool3d(int inputsize, int poolsize, int depth, int layer_num){
    int outputsize = inputsize/poolsize;

    dim3 grid(1,1,depth);
    dim3 block(outputsize,outputsize,1);

    if (layer_num == 1) {
        pool3dkernel<<<grid,block>>>(d_layer1,d_layer2,inputsize,poolsize,depth);
    } else if (layer_num == 3) {
        pool3dkernel<<<grid,block>>>(d_layer3,d_layer4,inputsize,poolsize,depth);
    }

}

__global__ void relu_kernel(float* input, int size) {
    for(int i=0;i<size;i++){
        input[i] = fmax(0,input[i]);
    }
}

void relu(int size){
    relu_kernel<<<1,1,1>>>(d_layer5, size);
}

__global__ void softmax_kernel(float* input, int size) {
    float sum = 0.0;
    for(int i=0;i<size;i++){
        sum += (float)exp(input[i]);
    }
    for(int i=0;i<size;i++){
        input[i] = (float)exp(input[i])/sum;
    }
}

void softmax(int size){
    softmax_kernel<<<1,1,1>>>(d_layer6, size);
}

void read_weight_file(string filename, float* kernel, float* bias, int dim1,int dim2, int depth) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);
    // kernel.resize(dim1*dim1*dim2*depth);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < depth*dim1*dim1*dim2; i++) {
        fin >> kernel[i];
    }
    // bias.resize(depth);
    for (int i = 0; i < depth; i++) {
        fin >> bias[i];
    }
}

void read_image_file(string filename, float* image, int dim1) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);
    // image.resize(dim1*dim1);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < dim1*dim1; i++) {
        fin >> image[i];
    }
}


void lenet(){
    // struct timespec start, end;
    // long timediff;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    convolve3d1(28, 5, 20);
    pool3d(24, 2, 20, 1);
    convolve3d2(12, 5, 20, 50, 3);
    pool3d(8, 2, 50, 3);
    convolve3d2(4, 4, 50, 500, 5);
    relu(1*1*500);
    convolve3d2(1, 1, 500, 10, 6);
    softmax(10);
    cudaMemcpy(h_layer6, d_layer6, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // timediff = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    // std::cout << "Time taken: " << timediff << " milliseconds" << std::endl;

    vector<pair<data_type, int> > predictions(10);
    for (int i = 0; i < predictions.size(); i++) {
        predictions[i] = make_pair(100.0 * h_layer6[i], i);
    }

    sort(predictions.rbegin(), predictions.rend());

    for (int i = 0; i < 5; i++) {
        cout << predictions[i].first << " class " << predictions[i].second << '\n';
    }
    // free(out1);
    // free(out2);
    // free(out3);
    // free(out4);
    // free(out5);
    // free(out6);
}


int main (int argc, char *argv[]) {

    if (argc < 2) {
        cerr << "Filename required as argument\n";
        exit(1);
    }
    conv1_weights = new float[5*5*1*20];
    conv1_bias = new float[20];
    read_weight_file(conv1_filename, conv1_weights, conv1_bias,5,1,20);

    cudaMalloc(&conv1_weights_cuda, 5*5*1*20*sizeof(float));
    cudaMemcpy(conv1_weights_cuda, conv1_weights, 5*5*1*20*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&conv1_bias_cuda, 20*sizeof(float));
    cudaMemcpy(conv1_bias_cuda, conv1_bias, 20*sizeof(float), cudaMemcpyHostToDevice);

    conv2_weights = new float[5*5*20*50];
    conv2_bias = new float[50];
    read_weight_file(conv2_filename, conv2_weights, conv2_bias, 5, 20, 50);

    cudaMalloc(&conv2_weights_cuda, 5*5*20*50*sizeof(float));
    cudaMemcpy(conv2_weights_cuda, conv2_weights, 5*5*20*50*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&conv2_bias_cuda, 50*sizeof(float));
    cudaMemcpy(conv2_bias_cuda, conv2_bias, 50*sizeof(float), cudaMemcpyHostToDevice);

    fc1_weights = new float[4*4*50*500];
    fc1_bias = new float[500];
    read_weight_file(fc1_filename, fc1_weights, fc1_bias, 4, 50, 500);

    cudaMalloc(&fc1_weights_cuda, 4*4*50*500*sizeof(float));
    cudaMemcpy(fc1_weights_cuda, fc1_weights, 4*4*50*500*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&fc1_bias_cuda, 500*sizeof(float));
    cudaMemcpy(fc1_bias_cuda, fc1_bias, 500*sizeof(float), cudaMemcpyHostToDevice);

    fc2_weights = new float[1*1*500*10];
    fc2_bias = new float[10];
    read_weight_file(fc2_filename, fc2_weights, fc2_bias, 1, 500, 10);

    cudaMalloc(&fc2_weights_cuda, 1*1*500*10*sizeof(float));
    cudaMemcpy(fc2_weights_cuda, fc2_weights, 1*1*500*10*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&fc2_bias_cuda, 10*sizeof(float));
    cudaMemcpy(fc2_bias_cuda, fc2_bias, 10*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_layer0, 28 * 28 * sizeof(float));
    cudaMalloc(&d_layer1, 24 * 24 * 20 * sizeof(float));
    cudaMalloc(&d_layer2, 12 * 12 * 20 * sizeof(float));
    cudaMalloc(&d_layer3, 8 * 8 * 50 * sizeof(float));
    cudaMalloc(&d_layer4, 4 * 4 * 50 * sizeof(float));
    cudaMalloc(&d_layer5, 1 * 1 * 500 * sizeof(float));
    cudaMalloc(&d_layer6, 1 * 1 * 10 * sizeof(float));
    h_layer6 = new float[10];

    float img_matrices[argc - 1][28 * 28];
    for (int i = 1; i < argc; i++) {
        read_image_file(argv[i], img_matrices[i-1], 28);
    }
    
    // strin directory = argv[i];
    // for (const auto& entry : fs::directory_iterator(directory)) {
    //     lenet(entry.path().string());
    // }
    for(int i = 1; i < argc; i++){
        cudaMemcpy(d_layer0, img_matrices[i-1], 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
        lenet();
    }
    free(conv1_weights);
    free(conv1_bias);
    free(conv2_weights);
    free(conv2_bias);
    free(fc1_weights);
    free(fc1_bias);
    free(fc2_weights);
    free(fc2_bias);
    free(h_layer6);
    cudaFree(conv1_weights_cuda);
    cudaFree(conv1_bias_cuda);
    cudaFree(conv2_weights_cuda);
    cudaFree(conv2_bias_cuda);
    cudaFree(fc1_weights_cuda);
    cudaFree(fc1_bias_cuda);
    cudaFree(fc2_weights_cuda);
    cudaFree(fc2_bias_cuda);
    cudaFree(d_layer0);
    cudaFree(d_layer1);
    cudaFree(d_layer2);
    cudaFree(d_layer3);
    cudaFree(d_layer4);
    cudaFree(d_layer5);
    cudaFree(d_layer6);
    return 0;
}
