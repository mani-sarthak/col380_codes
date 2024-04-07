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

using namespace std;


#define data_type float


string conv1_filename = "./trained_weights/conv1.txt";
string conv2_filename = "./trained_weights/conv2.txt";

string fc1_filename = "./trained_weights/fc1.txt";
string fc2_filename = "./trained_weights/fc2.txt";

vector<float> conv1_weights, conv2_weights, fc1_weights, fc2_weights;
vector<float> conv1_bias, conv2_bias, fc1_bias, fc2_bias;
float*** layer_outputs;
float *d_conv1_weights, *d_conv2_weights, *d_fc1_weights, *d_fc2_weights,
    *d_conv1_bias, *d_conv2_bias, *d_fc1_bias, *d_fc2_bias;

__global__ void  convolve3d1kernel(float* input,float* kernel, float* output,int inputsize,int kernelsize,int depth){
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
        output[k*out*out + i*out + j] =sum;
    }
}

vector<data_type> convolve3d1(vector<data_type>&input, float* d_kernel, int inputsize, int kernelsize, int depth){
    int outputsize = inputsize - kernelsize + 1;
    vector<data_type> output(depth*outputsize*outputsize,0);
    float* d_input;
    // float* d_kernel;
    float* d_output;

    cudaMalloc(&d_input, inputsize*inputsize*sizeof(float));
    // cudaMalloc(&d_kernel, kernelsize*kernelsize*depth*sizeof(float));
    cudaMalloc(&d_output, outputsize*outputsize*depth*sizeof(float));

    cudaMemcpy(d_input, input.data(), inputsize*inputsize*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel, kernel.data(), kernelsize*kernelsize*depth*sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(1,1,depth);
    dim3 block(outputsize,outputsize,1);
    convolve3d1kernel<<<grid,block>>>(d_input,d_kernel,d_output,inputsize,kernelsize,depth);
    cudaMemcpy(output.data(), d_output, outputsize*outputsize*depth*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    // cudaFree(d_kernel);
    cudaFree(d_output);
    return output;
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

vector<data_type> convolve3d2(vector<data_type> &input, float* d_kernel, int inputsize, int kernelsize, int depth,int channels){

    int outputsize = inputsize - kernelsize + 1;
    vector<data_type> output(channels*outputsize*outputsize,0);
    float* d_input;
    // float* d_kernel;
    float* d_output;

    cudaMalloc(&d_input, inputsize*inputsize*depth*sizeof(float));
    // cudaMalloc(&d_kernel, kernelsize*kernelsize*depth*channels*sizeof(float));
    cudaMalloc(&d_output, outputsize*outputsize*channels*sizeof(float));

    cudaMemcpy(d_input, input.data(), inputsize*inputsize*depth*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel, kernel.data(), kernelsize*kernelsize*depth*channels*sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(1,1,depth*channels);
    dim3 block(outputsize,outputsize,1);
    convolve3d2kernel<<<grid,block>>>(d_input,d_kernel,d_output,inputsize,kernelsize,depth,channels);
    cudaMemcpy(output.data(), d_output, outputsize*outputsize*channels*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    // cudaFree(d_kernel);
    cudaFree(d_output);
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

vector<data_type> pool3d(vector<data_type>&input, int inputsize, int poolsize, int depth){
    int outputsize = inputsize/poolsize;
    vector<data_type> output(depth*outputsize*outputsize,0);
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, inputsize*inputsize*depth*sizeof(float));
    cudaMalloc(&d_output, outputsize*outputsize*depth*sizeof(float));

    cudaMemcpy(d_input, input.data(), inputsize*inputsize*depth*sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(1,1,depth);
    dim3 block(outputsize,outputsize,1);
    pool3dkernel<<<grid,block>>>(d_input,d_output,inputsize,poolsize,depth);
    cudaMemcpy(output.data(), d_output, outputsize*outputsize*depth*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

vector<float> relu(vector<float>&input){
    vector<float> output(input.size(),0);
    for(int i=0;i<input.size();i++){
        output[i] = fmax(0,input[i]);
    }
    return output;
}

vector<float> softmax(vector<float>&input){
    vector<float> output(input.size(),0);
    float sum = 0.0;
    for(int i=0;i<input.size();i++){
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for(int i=0;i<input.size();i++){
        output[i] /= sum;
    }
    return output;
}

void addbias(vector<float>&input, vector<float>&bias,int size, int depth){
    for(int i=0;i<depth;i++){
        vector<float> temp(size,bias[i]);
        for(int j=0;j<size;j++){
            input[i*size+j] += temp[j];
        }
    }
}


void read_weight_file(string filename, vector<data_type> &kernel, vector<data_type> &bias,
                      float** d_kernel, float** d_bias,
                      int dim1, int dim2, int depth) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);
    kernel.resize(dim1*dim1*dim2*depth);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < depth*dim1*dim1*dim2; i++) {
        fin >> kernel[i];
    }
    bias.resize(depth);
    for (int i = 0; i < depth; i++) {
        fin >> bias[i];
    }

    cudaMalloc(d_kernel, kernel.size() * sizeof(float));
    cudaMemcpy(*d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(d_bias, bias.size() * sizeof(float));
    cudaMemcpy(*d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);

}



void read_image_file(string filename, vector<data_type> &image, int dim1) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);
    image.resize(dim1*dim1);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < dim1*dim1; i++) {
        fin >> image[i];
    }
}


void lenet(string filename){
    vector<float> img;
    read_image_file(filename,img,28);
    struct timespec start, end;
    long timediff;
    clock_gettime(CLOCK_MONOTONIC, &start);
    vector<float> out1 = convolve3d1(img,d_conv1_weights,28,5,20);
    addbias(out1,conv1_bias,24*24,20);
    vector<float> out2 =  pool3d(out1,24,2,20);
    vector<float> out3 = convolve3d2(out2,d_conv2_weights,12,5,20,50);
    addbias(out3,conv2_bias,8*8,50);
    vector<float> out4 = pool3d(out3,8,2,50);
    vector<float> out5 = convolve3d2(out4,d_fc1_weights,4,4,50,500);
    addbias(out5,fc1_bias,1*1,500);
    vector<float> out6 = relu(out5);
    vector<float> out7 = convolve3d2(out6,d_fc2_weights,1,1,500,10);
    addbias(out7,fc2_bias,1*1,10);
    vector<float> out8 = softmax(out7);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timediff = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    std::cout << "Time taken: " << timediff << " milliseconds" << std::endl;

    vector<pair<data_type, int> > predictions(out8.size());
    for (int i = 0; i < predictions.size(); i++) {
        predictions[i] = make_pair(100.0 *out8[i], i);
    }

    sort(predictions.rbegin(), predictions.rend());

    for (int i = 0; i < 5; i++) {
        cout << predictions[i].first << " class " << predictions[i].second << '\n';
    }
}


int main (int argc, char *argv[]) {

    if (argc < 2) {
        cerr << "Filename required as argument\n";
        exit(1);
    }

    read_weight_file(conv1_filename, conv1_weights, conv1_bias, &d_conv1_weights, &d_conv1_bias, 5, 1, 20);
    read_weight_file(conv2_filename, conv2_weights, conv2_bias, &d_conv2_weights, &d_conv2_bias, 5, 20, 50);

    read_weight_file(fc1_filename, fc1_weights, fc1_bias, &d_fc1_weights, &d_fc1_bias, 4, 50, 500);
    read_weight_file(fc2_filename, fc2_weights, fc2_bias, &d_fc2_weights, &d_fc2_bias, 1, 500, 10);

    for (int i = 1; i < argc; i++) {
        lenet(argv[i]);
    }

    cudaFree(d_conv1_weights);
    cudaFree(d_conv2_weights);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc2_weights);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_bias);
    cudaFree(d_fc1_bias);
    cudaFree(d_fc2_bias);


    return 0;
}
