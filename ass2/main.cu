
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

using namespace std;

typedef float data_type;
typedef vector<vector<data_type> > matrix;

vector<vector<matrix> >
    conv1_matrix,
    conv2_matrix,
    fc1_matrix,
    fc2_matrix;

vector<data_type>
    conv1_bias,
    conv2_bias,
    fc1_bias,
    fc2_bias;

string conv1_filename = "./trained_weights/conv1.txt";
string conv2_filename = "./trained_weights/conv2.txt";

string fc1_filename = "./trained_weights/fc1.txt";
string fc2_filename = "./trained_weights/fc2.txt";

void print_vector(const vector<float> &vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

void print_matrix(vector<vector<float> > mat){
    int size = mat.size();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

__global__ void convolutionWithPaddingKernel(float** input, float** kernel,float** output, int inputsize, int kernelsize,int pad ){
    int i= threadIdx.x + blockIdx.x*blockDim.x;
    int j= threadIdx.y + blockIdx.y*blockDim.y;

    if(i<inputsize && j<inputsize){
        for(int x=0;x<kernelsize;x++){
            for(int y=0;y<kernelsize;y++){
                if(i+x>=0 && i+x<(inputsize+2*pad) && j+y>=0 && j+y<(inputsize+2*pad)){
                    atomicAdd(&output[i][j],input[i+x][j+y]*kernel[x][y]);
                }
            }
        }
    } 
}

vector<vector<float> > convolutionWithPadding(vector<vector<float> >& input_matrix, vector<vector<float> >& kernel){
    int inputsize = input_matrix.size();
    int kernelsize = kernel.size();
    int pad = (kernelsize-1)/2;

    vector<vector<float> > input_pad(inputsize + 2*pad, vector<float>(inputsize + 2*pad,0.0));
    for(int i=pad;i<inputsize+pad;i++){
        for(int j=pad;j<inputsize+pad;j++){
            input_pad[i][j] = input_matrix[i-pad][j-pad];
        }
    }

    vector<vector<float> > output(inputsize, vector<float>(inputsize,0.0));
    float **d_input, **d_kernel, **d_output;

    cudaMalloc((void**)&d_input, (inputsize+2*pad)*sizeof(float*));
    cudaMalloc((void**)&d_kernel, kernelsize*sizeof(float*));
    cudaMalloc((void**)&d_output, inputsize*sizeof(float*));

    float* t_input_pad[inputsize + 2*pad],*t_kernel[kernelsize],*t_output[inputsize];

    for(int i=0;i<inputsize+2*pad;i++){
        cudaMalloc((void**)&t_input_pad[i], (inputsize+2*pad)*sizeof(float));
        cudaMemcpy(t_input_pad[i],input_pad[i].data(), (inputsize+2*pad)*sizeof(float), cudaMemcpyHostToDevice);
    }

    for(int i=0;i<kernelsize;i++){
        cudaMalloc((void**)&t_kernel[i], kernelsize*sizeof(float));
        cudaMemcpy(t_kernel[i], kernel[i].data(), kernelsize*sizeof(float), cudaMemcpyHostToDevice);
    }

    for(int i=0;i<inputsize;i++){
        cudaMalloc((void**)&t_output[i], inputsize*sizeof(float));
    }

    cudaMemcpy(d_input, t_input_pad, (inputsize+2*pad)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, t_kernel, kernelsize*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, t_output, inputsize*sizeof(float*), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((inputsize + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputsize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolutionWithPaddingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, inputsize, kernelsize, pad);

    cudaMemcpy(t_output,d_output,inputsize*sizeof(float*),cudaMemcpyDeviceToHost);

    for(int i=0;i<inputsize;i++){
        cudaMemcpy(output[i].data(),t_output[i],inputsize*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(t_output[i]);
    }
    for(int i=0;i<kernelsize;i++){
        cudaFree(t_kernel[i]);
    }
    for(int i=0;i<inputsize+2*pad;i++){
        cudaFree(t_input_pad[i]);
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return output;
}

__global__ void convolutionWithoutPaddingKernel(float** input, float** kernel,float** output, int inputsize, int kernelsize){
    int i= threadIdx.x + blockIdx.x*blockDim.x;
    int j= threadIdx.y + blockIdx.y*blockDim.y;
    int outsize = inputsize-kernelsize+1;

    if(i<outsize && j<outsize){
        for(int x=0;x<kernelsize;x++){
            for(int y=0;y<kernelsize;y++){
                atomicAdd(&output[i][j],input[i+x][j+y]*kernel[x][y]);
            }
        }
    } 
}

vector<vector<float> > convolutionWithoutPadding(vector<vector<float> >& input_matrix, vector<vector<float> >& kernel){
    int inputsize = input_matrix.size();
    int kernelsize = kernel.size();
    int out = inputsize-kernelsize+1;

    vector<vector<float> > output(out, vector<float>(out,0.0));
    float **d_input, **d_kernel, **d_output;

    cudaMalloc((void**)&d_input, inputsize*sizeof(float*));
    cudaMalloc((void**)&d_kernel, kernelsize*sizeof(float*));
    cudaMalloc((void**)&d_output, out*sizeof(float*));

    float* t_input[inputsize],*t_kernel[kernelsize],*t_output[out];

    for(int i=0;i<inputsize;i++){
        cudaMalloc((void**)&t_input[i], (inputsize)*sizeof(float));
        cudaMemcpy(t_input[i],input_matrix[i].data(), (inputsize)*sizeof(float), cudaMemcpyHostToDevice);
    }

    for(int i=0;i<kernelsize;i++){
        cudaMalloc((void**)&t_kernel[i], kernelsize*sizeof(float));
        cudaMemcpy(t_kernel[i], kernel[i].data(), kernelsize*sizeof(float), cudaMemcpyHostToDevice);
    }

    for(int i=0;i<out;i++){
        cudaMalloc((void**)&t_output[i], out*sizeof(float));
    }


    cudaMemcpy(d_input, t_input, inputsize*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, t_kernel, kernelsize*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, t_output, out*sizeof(float*), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((inputsize + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputsize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolutionWithoutPaddingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, inputsize, kernelsize);

    cudaMemcpy(t_output,d_output,out*sizeof(float*),cudaMemcpyDeviceToHost);

    for(int i=0;i<out;i++){
        cudaMemcpy(output[i].data(),t_output[i],out*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(t_output[i]);
    }
    for(int i=0;i<inputsize;i++){
        cudaFree(t_input[i]);
    }
    for(int i=0;i<kernelsize;i++){
        cudaFree(t_kernel[i]);
    }
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return output;
}


__device__ float function_relu(float x){
    return max(0.0f,x);
}

__global__ void reluActivationKernel(float** input,float** output,int n,int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i<n && j<m){
        output[i][j] = function_relu(input[i][j]);
    }
}

void reluActivation(vector<vector<float> >& input_matrix,vector<vector<float> >& output_matrix){
    int n = input_matrix.size();
    int m = input_matrix[0].size();

    float **d_input, **d_output;
    cudaMalloc((void**)&d_input,n*sizeof(float*));
    cudaMalloc((void**)&d_output,n*sizeof(float*));

    float *t_input[n],*t_output[n];

    for (int i = 0; i < n; ++i) {
        cudaMalloc((void**)&t_input[i], m * sizeof(float));
        cudaMalloc((void**)&t_output[i], m * sizeof(float));
        cudaMemcpy(t_input[i], input_matrix[i].data(), m * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_input, t_input, n * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, t_output, n * sizeof(float*), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x,(m+threadsPerBlock.y-1)/threadsPerBlock.y);

    reluActivationKernel<<<numBlocks,threadsPerBlock>>>(d_input,d_output,n,m);

    cudaMemcpy(t_output, d_output, n * sizeof(float*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        cudaMemcpy(output_matrix[i].data(), t_output[i], m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(t_input[i]);
        cudaFree(t_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);


}

__device__ float function_tanh(float x){
    return tanh(x);
}

__global__ void tanhActivationKernel(float** input,float** output,int n,int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i<n && j<m){
        output[i][j] = function_tanh(input[i][j]);
    }
}

void tanhActivation(vector<vector<float> >& input_matrix,vector<vector<float> >& output_matrix){
    int n = input_matrix.size();
    int m = input_matrix[0].size();

    float **d_input, **d_output;
    cudaMalloc((void**)&d_input,n*sizeof(float*));
    cudaMalloc((void**)&d_output,n*sizeof(float*));

    float *t_input[n],*t_output[n];

    for (int i = 0; i < n; ++i) {
        cudaMalloc((void**)&t_input[i], m * sizeof(float));
        cudaMalloc((void**)&t_output[i], m * sizeof(float));
        cudaMemcpy(t_input[i], input_matrix[i].data(), m * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_input, t_input, n * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, t_output, n * sizeof(float*), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x,(m+threadsPerBlock.y-1)/threadsPerBlock.y);

    tanhActivationKernel<<<numBlocks,threadsPerBlock>>>(d_input,d_output,n,m);

    cudaMemcpy(t_output, d_output, n * sizeof(float*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        cudaMemcpy(output_matrix[i].data(), t_output[i], m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(t_input[i]);
        cudaFree(t_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);


}

__global__ void maxPoolingKernel(float* input, float* output,int n,int m,int pool){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<n/pool && j<m/pool){
        float max_value = -FLT_MAX;
        for(int x=0;x<pool;x++){
            for (int y=0;y<pool;y++){
                max_value = max(max_value, input[(i*pool+x)*m + j*pool +y]);
            }
        }
        output[i*(m/pool)+j]=max_value;
    }
}

__global__ void averagePoolingKernel(float* input, float* output,int n,int m,int pool){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<n/pool && j<m/pool){
        float sum = 0.0f;
        for(int x=0;x<pool;x++){
            for (int y=0;y<pool;y++){
                sum+= input[(i*pool+x)*m + j*pool +y];
            }
        }
        output[i*(m/pool)+j]=sum/(pool*pool);
    }
}

vector<vector<float> > maxPooling(vector<vector<float> > &input_matrix,int pool_size){
    int n = input_matrix.size();
    int m = input_matrix[0].size();
    vector<vector<float> > output((n/pool_size) ,vector<float>(m/pool_size));
    vector<float> f_input(n*m);
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            f_input[i*m+j] = input_matrix[i][j];

        }
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n*m*sizeof(float));
    cudaMalloc((void**) &d_output,(n/pool_size)*(m/pool_size)*sizeof(float));

    cudaMemcpy(d_input, f_input.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n+ threadsPerBlock.y - 1) / threadsPerBlock.y);

    maxPoolingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, n, m, pool_size);
    vector<float> f_output((n/pool_size)*(m/pool_size));


    cudaMemcpy(f_output.data(),d_output,(n/pool_size)*(m/pool_size)*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0;i<(n/pool_size);i++){
        for(int j=0;j<(m/pool_size);j++){
            output[i][j] = f_output[i*(m/pool_size)+j];
        }
    }
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

vector<vector<float> > avgPooling(vector<vector<float> > &input_matrix,int pool_size){
    int n = input_matrix.size();
    int m = input_matrix[0].size();
    vector<vector<float> > output((n/pool_size) ,vector<float>(m/pool_size));
    vector<float> f_input(n*m);
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            f_input[i*m+j] = input_matrix[i][j];

        }
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n*m*sizeof(float));
    cudaMalloc((void**) &d_output,(n/pool_size)*(m/pool_size)*sizeof(float));

    cudaMemcpy(d_input, f_input.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n+ threadsPerBlock.y - 1) / threadsPerBlock.y);

    averagePoolingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, n, m, pool_size);
    vector<float> f_output((n/pool_size)*(m/pool_size));

    cudaMemcpy(f_output.data(),d_output,(n/pool_size)*(m/pool_size)*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0;i<(n/pool_size);i++){
        for(int j=0;j<(m/pool_size);j++){
            output[i][j] = f_output[i*(m/pool_size)+j];
        }
    }
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

__global__ void computeExp(float* input, float* output,int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        output[i]=exp(input[i]);
    }
}

__global__ void compSum(float* input, float* output, int size) {
    extern __shared__ float sdata[];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[t] = (i < size) ? input[i] : 0;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            sdata[t] += sdata[t + s];
        }
        __syncthreads();
    }

    if (t == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void softmaxKernel(float* input, float* output, float totalSum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = exp(input[i]) / totalSum;
    }
}

__global__ void sigmoidKernel(float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

vector<data_type> softmax(const vector<float >& input) {
    int n = input.size();
    // int m = input[0].size();
    // int size = n * m;

    float *d_input, *d_exp, *d_sum, *d_prob;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_exp, n * sizeof(float));
    cudaMalloc((void**)&d_sum, n * sizeof(float));
    cudaMalloc((void**)&d_prob, n * sizeof(float));

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    computeExp<<<(n + 255) / 256, 256>>>(d_input, d_exp, n);

    compSum<<<(n + 255) / 256, 256, 256 * sizeof(float)>>>(d_exp, d_sum, n);

    vector<float> sum_f(n);
    cudaMemcpy(sum_f.data(), d_sum, n * sizeof(float), cudaMemcpyDeviceToHost);
    float totalSum = 0.0f;
    for (int i=0;i<sum_f.size();i++) {
        totalSum += sum_f[i];
    }

    softmaxKernel<<<(n + 255) / 256, 256>>>(d_input, d_prob, totalSum, n);

    vector<data_type > output(n,0.0);
    cudaMemcpy(output.data(), d_prob, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_exp);
    cudaFree(d_sum);
    cudaFree(d_prob);

    return output;
}

vector<data_type > sigmoid(const vector<float>& input) {
    int n = input.size();
    // int m = input[0].size();
    // int size = n * m;

    float *d_input, *d_output;
    vector<data_type> output(n,0.0);

    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    sigmoidKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

void read_weight_file(string filename, vector<vector<matrix> > &v, vector<data_type> &bias, int matDim, int dim1, int dim2) {
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);
    v.resize(dim2);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < dim2; i++) {
        v[i].resize(dim1);
        for (int j = 0; j < dim1; j++) {
            v[i][j].resize(matDim);
            for (int k = 0; k < matDim; k++) {
                v[i][j][k].resize(matDim);
                for (int l = 0; l < matDim; l++) {
                    fin >> v[i][j][k][l];
                }
            }
        }
    }

    bias.resize(dim2);
    for (int i = 0; i < dim2; i++) {
        fin >> bias[i];
    }

}

void read_image_file(string filename, matrix &v, int dimension) {
    v.resize(dimension);
    const char* filename_str = filename.c_str();

    ifstream fin(filename_str);

    if (!fin) {
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < dimension; i++) {
        v[i].resize(dimension);
        for (int j = 0; j < dimension; j++) {
            fin >> v[i][j];
        }
    }
}

void sum_matrix(matrix &input1, matrix &input2, matrix &output){
    int n = input1.size();
    int m = input1[0].size();
    assert(n == input2.size());
    assert(m == input2[0].size());
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            output[i][j] = input1[i][j] + input2[i][j];
        }
    }
}

void convolve3d(vector<matrix> &input1, vector<matrix> &input2, data_type bias, matrix &output){
    int k1 = input1.size();
    assert(k1 == input2.size());
    int n1 = input1[0].size();
    int m1 = input1[0][0].size();
    int n2 = input2[0].size();
    int m2 = input2[0][0].size();
    output.resize(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1, 0));
    for (int ind = 0; ind < k1; ind++){
        matrix temp = convolutionWithoutPadding(input1[ind], input2[ind]);
        temp.resize(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1));
        sum_matrix(output, temp, output);
    }
    matrix bias_matrix(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1, bias));
    sum_matrix(output, bias_matrix, output);
}


void convolve_kernels(vector<matrix> &input, vector<vector<matrix> >&kernal, vector<data_type> &bias, vector<matrix> &output){
    int n = input[0].size();
    int k1 = input.size(); 
    int k2 = kernal.size();
    assert(k1 == kernal[0].size());
    assert(k2 == bias.size());
    assert(input[0][0].size() == n);
    output.resize(k2);
    for (int i = 0; i < k2; i++){
        convolve3d(input, kernal[i], bias[i], output[i]);
    }
    assert(output.size() == k2);
}

void pool3d(vector<matrix> &input, int pool_size, vector<matrix> &output){
    int n = input.size();
    output.resize(n);
    for (int i = 0; i < n; i++){
        output[i] = maxPooling(input[i], pool_size);
    }
}

void printing(vector<vector<vector<float> > > input){
    string filename = "output.txt";
    const char* filename_str = filename.c_str();

    // Open the file for writing
    ofstream outFile(filename_str);

    // Check if the file is opened successfully
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open the file." << endl;
        return;
    }

    // Write the data to the file
    for (int i = 0; i < input.size(); ++i) {
        for (int j = 0; j < input[i].size(); ++j) {
            for (int k = 0; k < input[i][j].size(); ++k) {
                outFile << input[i][j][k] << " ";
            }
            outFile << endl;
        }
        outFile << endl;
    }


    // Close the file
    outFile.close();

    cout << "Data has been written to " << filename << endl;
}

void predict(string image_name){

    matrix img_matrix2D;  
    
    vector<matrix> img_matrix3D(1);

    read_image_file(image_name, img_matrix2D, 28);
    img_matrix3D[0] = img_matrix2D;

    vector<matrix> conv1_output, conv2_output, pool1_output, pool2_output;
    vector<matrix> fc1_output, fc2_output;

    convolve_kernels(img_matrix3D, conv1_matrix, conv1_bias, conv1_output);
    // printing(conv1_output);
    pool3d(conv1_output, 2, pool1_output);
    cout << "Pool1 Output: " << pool1_output.size() << " " << pool1_output[0].size() << " " << pool1_output[0][0].size() << endl;
    
    convolve_kernels(pool1_output, conv2_matrix, conv2_bias, conv2_output);
    pool3d(conv2_output, 2, pool2_output);
    cout << "Pool2 Output: " << pool2_output.size() << " " << pool2_output[0].size() << " " << pool2_output[0][0].size() << endl;

    convolve_kernels(pool2_output, fc1_matrix, fc1_bias, fc1_output);
    for (int i = 0; i < fc1_output.size(); i++) {
        // apply_activation(fc1_output[i], relu);
        reluActivation(fc1_output[i], fc1_output[i]);  
    }
    // cout << "FC1 Output: " << fc1_output.size() << " " << fc1_output[0].size() << " " << fc1_output[0][0].size() << endl;
    
    convolve_kernels(fc1_output, fc2_matrix, fc2_bias, fc2_output);
    // cout << "FC2 Output: " << fc2_output.size() << " " << fc2_output[0].size() << " " << fc2_output[0][0].size() << endl;

    vector<data_type> output;
    for (int i = 0; i < fc2_output.size(); i++){
        output.push_back(fc2_output[i][0][0]);
    }

    vector<data_type> probability = softmax(output);

    vector<pair<data_type, int> > predictions(probability.size());
    for (int i = 0; i < predictions.size(); i++) {
        predictions[i] = make_pair(100.0 * probability[i], i);
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

    read_weight_file(conv1_filename, conv1_matrix, conv1_bias, 5, 1, 20);
    read_weight_file(conv2_filename, conv2_matrix, conv2_bias, 5, 20, 50);

    read_weight_file(fc1_filename, fc1_matrix, fc1_bias, 4, 50, 500);
    read_weight_file(fc2_filename, fc2_matrix, fc2_bias, 1, 500, 10);

    for (int i = 1; i < argc; i++) {
        predict(argv[i]);
    }

    return 0;
}
