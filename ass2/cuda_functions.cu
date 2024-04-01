#include <iostream>
#include <vector>
// #include <cuda_runtime.h>
#include "cuda_runtime.h"
#include <limits>
#include <cfloat>
#include "device_launch_parameters.h"
#include<algorithm>
#include<cmath>

using namespace std;


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


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((inputsize + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputsize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolutionWithPaddingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, inputsize, kernelsize, pad);

    cudaMemcpy(t_output,d_output,inputsize*sizeof(float*),cudaMemcpyDeviceToHost);

    for(int i=0;i<inputsize;i++){
        cudaMemcpy(output[i].data(),t_output[i],inputsize*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(t_output[i]);
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
        cudaFree(t_input[i]);

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

vector<float> softmax(const vector<vector<float> >& input) {
    int n = input.size();
    int m = input[0].size();
    int size = n * m;

    float *d_input, *d_exp, *d_sum, *d_prob;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_exp, n * sizeof(float));
    cudaMalloc((void**)&d_sum, n * sizeof(float));
    cudaMalloc((void**)&d_prob, n * sizeof(float));

    cudaMemcpy(d_input, input[0].data(), n * sizeof(float), cudaMemcpyHostToDevice);

    computeExp<<<(size + 255) / 256, 256>>>(d_input, d_exp, n);

    compSum<<<(n + 255) / 256, 256, 256 * sizeof(float)>>>(d_exp, d_sum, n);

    vector<float> sum_f(n);
    cudaMemcpy(sum_f.data(), d_sum, n * sizeof(float), cudaMemcpyDeviceToHost);
    float totalSum = 0.0f;
    for (int i=0;i<sum_f.size();i++) {
        totalSum += sum_f[i];
    }

    softmaxKernel<<<(size + 255) / 256, 256>>>(d_input, d_prob, totalSum, n);

    vector<float > output(n,0.0);
    cudaMemcpy(output.data(), d_prob, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_exp);
    cudaFree(d_sum);
    cudaFree(d_prob);

    return output;
}

vector<float > sigmoid(const vector<vector<float> >& input) {
    int n = input.size();
    int m = input[0].size();
    int size = n * m;

    float *d_input, *d_output;
    vector<float > output(n,0.0);
    cout<<"here1"<<endl;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input[0].data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cout<<"here2"<<endl;


    sigmoidKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cout<<"here3"<<endl;
    // print_matrix(output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cout<<"here4"<<endl;


    return output;
}


int main(){
    vector<vector<float> > inputMatrix;
    for(int i=0;i<5;i++){
        vector<float> row;
        for(int j=0;j<5;j++){
            row.push_back(i*5+j);
        }
        inputMatrix.push_back(row);
    }
    print_matrix(inputMatrix);
    vector<vector<float> > output = avgPooling(inputMatrix,2);
    print_matrix(output);
    vector<vector<float> > output2(inputMatrix.size() ,vector<float>(inputMatrix[0].size()));
    
    reluActivation(inputMatrix,output2);
    print_matrix(output2);

    vector<vector<float> > kernel;
    for(int i=0;i<3;i++){
        vector<float> row;
        for(int j=0;j<3;j++){
            row.push_back(i*3+j);
        }
        kernel.push_back(row);
    }
    print_matrix(kernel);
    vector<vector<float> > output4 = convolutionWithPadding(inputMatrix,kernel);
    print_matrix(output4);

    vector<float > output3 = sigmoid(inputMatrix);
    cout<<"Sigmoid: \n";
    print_vector(output3);

    return 0;
}
