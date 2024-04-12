
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include <limits>
#include <cfloat>
#include <cassert>
#include "device_launch_parameters.h"
#include<algorithm>
#include<cmath>
#include<time.h>

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
        output[k*out*out + i*out + j] =(sum + bias[k]);
    }
}

float*convolve3d1(float* input, int inputsize, int kernelsize, int depth){
    int outputsize = inputsize - kernelsize + 1;
    float* output = new float[outputsize*outputsize*depth];
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, inputsize*inputsize*sizeof(float));
    cudaMemcpy(d_input, input, inputsize*inputsize*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, outputsize*outputsize*depth*sizeof(float));

    dim3 grid(1,1,depth);
    dim3 block(outputsize,outputsize,1);
    convolve3d1kernel<<<grid,block>>>(d_input,conv1_weights_cuda,d_output,inputsize,kernelsize,depth,conv1_bias_cuda);

    cudaMemcpy(output, d_output, outputsize*outputsize*depth*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&d_input);
    cudaFree(&d_output);

    return output;
}

__global__ void addbiaskernel(float* output, float* bias, int size,int depth){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(i<size && j<size && k<depth){
        output[k*size*size + i*size + j] += bias[k];
    }
}
void addbias(float* input, int size, int depth,string type){
    float* output = new float[size*size*depth];
    float* d_input;

    cudaMalloc(&d_input, size*size*depth*sizeof(float));
    cudaMemcpy(d_input, input, size*size*depth*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(1,1,depth);
    dim3 block(size,size,1);
    if(type == "conv1"){
        addbiaskernel<<<grid,block>>>(d_input,conv1_bias_cuda,size,depth);
    }
    else if(type == "conv2"){
        addbiaskernel<<<grid,block>>>(d_input,conv2_bias_cuda,size,depth);
    }
    else if(type == "fc1"){
        addbiaskernel<<<grid,block>>>(d_input,fc1_bias_cuda,size,depth);
    }
    else if(type == "fc2"){
        addbiaskernel<<<grid,block>>>(d_input,fc2_bias_cuda,size,depth);
    }

    cudaMemcpy(input, d_input, size*size*depth*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&d_input);
    // cudaFree(&d_output);

    // return output;
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
        atomicAdd(&output[s_channel*out*out + i*out + j],sum);
    }
}

float* convolve3d2(float* input, int inputsize, int kernelsize, int depth,int channels, string type){

    int outputsize = inputsize - kernelsize + 1;
    float* output = new float[outputsize*outputsize*channels];

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, inputsize*inputsize*depth*sizeof(float));
    cudaMemcpy(d_input, input, inputsize*inputsize*depth*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, outputsize*outputsize*channels*sizeof(float));

    dim3 grid(1,1,depth*channels);
    dim3 block(outputsize,outputsize,1);
    if(type == "conv2"){
        convolve3d2kernel<<<grid,block>>>(d_input,conv2_weights_cuda,d_output,inputsize,kernelsize,depth,channels);
    }else if(type == "fc1"){
        convolve3d2kernel<<<grid,block>>>(d_input,fc1_weights_cuda,d_output,inputsize,kernelsize,depth,channels);
    }else if(type == "fc2"){
        convolve3d2kernel<<<grid,block>>>(d_input,fc2_weights_cuda,d_output,inputsize,kernelsize,depth,channels);
    }else{
        convolve3d2kernel<<<grid,block>>>(d_input,conv1_weights_cuda,d_output,inputsize,kernelsize,depth,channels);
    }
    cudaMemcpy(output, d_output, outputsize*outputsize*channels*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&d_input);
    cudaFree(&d_output);
    if(type == "conv2"){
        return output;
    }
    for(int i=0;i<channels;i++){
        for(int j=0;j<outputsize;j++){
            for(int k=0;k<outputsize;k++){
                if(type == "fc1"){
                    output[i*outputsize*outputsize + j*outputsize + k] += fc1_bias[i];
                }else if(type == "fc2"){
                    output[i*outputsize*outputsize + j*outputsize + k] += fc2_bias[i];
                }
            }
        }
    }
    return output;
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

float* pool3d(float* input, int inputsize, int poolsize, int depth){
    int outputsize = inputsize/poolsize;
    float* output = new float[outputsize*outputsize*depth];
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, inputsize*inputsize*depth*sizeof(float));
    cudaMemcpy(d_input, input, inputsize*inputsize*depth*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, outputsize*outputsize*depth*sizeof(float));

    dim3 grid(1,1,depth);
    dim3 block(outputsize,outputsize,1);

    pool3dkernel<<<grid,block>>>(d_input,d_output,inputsize,poolsize,depth);

    cudaMemcpy(output, d_output, outputsize*outputsize*depth*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&d_input);
    cudaFree(&d_output);
    return output;
}

void relu(float* input,int size){
    for(int i=0;i<size;i++){
        input[i] = fmax(0,input[i]);
    }
}

void softmax(float* input,int size){
    float sum = 0.0;
    for(int i=0;i<size;i++){
        sum += (float)exp(input[i]);
    }
    for(int i=0;i<size;i++){
        input[i] = (float)exp(input[i])/sum;
    }
}

void read_weight_file(string filename, float* kernel, float* bias, int dim1,int dim2, int depth) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < depth*dim1*dim1*dim2; i++) {
        fin >> kernel[i];
    }

    for (int i = 0; i < depth; i++) {
        fin >> bias[i];
    }
}

void read_image_file(string filename, float* image, int dim1) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < dim1*dim1; i++) {
        fin >> image[i];
    }
}


void lenet(float* img){
    // float* img = new float[28*28];
    // read_image_file(filename,img,28);
    // struct timespec start, end;
    // long timediff;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    float* out1 = convolve3d1(img,28,5,20);
    float* out2 =  pool3d(out1,24,2,20);
    float* out3 = convolve3d2(out2,12,5,20,50,"conv2");
    addbias(out3,8,50,"conv2");
    float* out4 = pool3d(out3,8,2,50);
    float* out5= convolve3d2(out4,4,4,50,500,"fc1");
    relu(out5,1*1*500);
    float* out6 = convolve3d2(out5,1,1,500,10,"fc2");
    softmax(out6,1*1*10);
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // timediff = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    // std::cout << "Time taken: " << timediff << " milliseconds" << std::endl;

    vector<pair<data_type, int> > predictions(10);
    for (int i = 0; i < predictions.size(); i++) {
        predictions[i] = make_pair(100.0 *out6[i], i);
    }

    sort(predictions.rbegin(), predictions.rend());

    for (int i = 0; i < 5; i++) {
        cout << predictions[i].first << " class " << predictions[i].second << '\n';
    }
    free(out1);
    free(out2);
    free(out3);
    free(out4);
    free(out5);
    free(out6);
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

    float img_matrices[argc - 1][28 * 28];
    for (int i = 1; i < argc; i++) {
        read_image_file(argv[i], img_matrices[i-1], 28);
    }



    struct timespec start, end;
    long timediff;
    clock_gettime(CLOCK_MONOTONIC, &start);



    for(int i=1;i<argc;i++){
        lenet(img_matrices[i-1]);
    }



    clock_gettime(CLOCK_MONOTONIC, &end);
    timediff = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    cout << "Time taken: " << timediff << " milliseconds" << endl;
    
    
    
    free(conv1_weights);
    free(conv1_bias);
    free(conv2_weights);
    free(conv2_bias);
    free(fc1_weights);
    free(fc1_bias);
    free(fc2_weights);
    free(fc2_bias);
    cudaFree(conv1_weights_cuda);
    cudaFree(conv1_bias_cuda);
    cudaFree(conv2_weights_cuda);
    cudaFree(conv2_bias_cuda);
    cudaFree(fc1_weights_cuda);
    cudaFree(fc1_bias_cuda);
    cudaFree(fc2_weights_cuda);
    cudaFree(fc2_bias_cuda);
    return 0;
}
