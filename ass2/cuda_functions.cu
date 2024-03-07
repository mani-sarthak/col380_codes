#include <iostream>
#include <vector>
// #include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<algorithm>
#include<cmath>

using namespace std;


__global__ void cudaConvolution(const float* input, const float* kernel, float* output, int inputSize, int kernelSize,int inputSize2, int kernelSize2,int padding) {
    int out= inputSize-kernelSize+1;
    int out2 = inputSize2 - kernelSize2 + 1;
    if(padding){
        out = inputSize;
        out2 = inputSize2;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int pad_size = (kernelSize - 1) / 2; 
    if(i<out && j<out2){
        float sum=0.0;
        for(int x=0;x<kernelSize;x++){
            for(int y=0;y<kernelSize2;y++){
                int x_index = i+x;
                int y_index = j+y;
                if(padding){
                    x_index -= kernelSize -1;
                    y_index -= kernelSize2 -1;
                }
                if(x_index>=0 && x_index<inputSize && y_index>=0 && y_index<inputSize2){
                sum += input[x_index*inputSize+y_index]*kernel[x*kernelSize+y];}
            }
        }
        output[i*out+j] = sum;
    }
}

vector<vector<float> > convolution(vector<vector<float> > &input_matrix, vector<vector<float> > & kernal, bool padding,int threads){
    int n = input_matrix.size();
    int k = kernal.size();
    int out = n-k+1;
    int out2 = input_matrix[0].size() - kernal[0].size() + 1;
    if (padding){
        out = n;
        out2 = input_matrix[0].size();
    }
    int ibytes = n*input_matrix[0].size()*sizeof(float);
    int kbytes = k*kernal[0].size()*sizeof(float);
    int obytes = out*out2*sizeof(float);

    vector<vector<float> > output(out, vector<float>(out2,0.0));
    float *d_input, *d_kernal, *d_output;
    cudamalloc(&d_input, ibytes);
    cudamalloc(&d_kernal, kbytes);
    cudamalloc(&d_output, obytes);

    cudaMemcpy(d_input, input_matrix.data(), ibytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernal, kernal.data(), kbytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((out + threadsPerBlock.x - 1) / threadsPerBlock.x, (out2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaConvolution<<<numBlocks, threadsPerBlock>>>(d_input, d_kernal, d_output, n, k, input_matrix[0].size(), kernal[0].size(),padding);


    cudaMemcpy(output.data(),d_output,obytes,cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernal);
    cudaFree(d_output);
    return output;

}



vector<vector<float> >Convolution_padding(vector<vector<float> > &input_matrix, vector<vector<float> > & kernal,int threads){
    return convolution(input_matrix, kernal, true,threads);
}

vector<vector<float> >Convolution_no_padding(vector<vector<float> > &input_matrix, vector<vector<float> > & kernal,int threads){
    return convolution(input_matrix, kernal, false,threads);
}

__global__ void cudaActivationRelu(float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        input[i] = fmaxf(0.0,input[i]);
    }
}

__global__ void cudaActivationTanh(float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        input[i] = tanhf(input[i]);
    }
}

vector<vector<float> > activation_relu(vector<vector<float> > &input_matrix,int threads){
    int n = input_matrix.size();
    int m = input_matrix[0].size();
    int bytes = n*m*sizeof(float);
    vector<float> output(n*m);

    float *d_input;
    cudamalloc(&d_input, bytes);
    cudaMemcpy(d_input, input_matrix.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock(threads*threads);
    int numBlocks((n*m + threadsPerBlock - 1) / threadsPerBlock);

    cudaActivationRelu<<<numBlocks, threadsPerBlock>>>(d_input, n*m);

    cudaMemcpy(output.data(),d_input,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    std::vector<std::vector<float> > result(n ,vector<float>(m,0.0));
    for (int i = 0; i < input_matrix.size(); ++i) {
        for (int j = 0; j < input_matrix[0].size(); ++j) {
            result[i][j] = output[i * input_matrix[0].size() + j];
        }
    }

    return result;
}

vector<vector<float> > activation_tanh(vector<vector<float> > &input_matrix,int threads){
    int n = input_matrix.size();
    int m = input_matrix[0].size();
    int bytes = n*m*sizeof(float);
    vector<float> output(n*m);

    float *d_input;
    cudamalloc(&d_input, bytes);
    cudaMemcpy(d_input, input_matrix.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock(threads*threads);
    int numBlocks((n*m + threadsPerBlock - 1) / threadsPerBlock);

    cudaActivationTanh<<<numBlocks, threadsPerBlock>>>(d_input, n*m);

    cudaMemcpy(output.data(),d_input,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    std::vector<std::vector<float> > result(n ,vector<float>(m,0.0));
    for (int i = 0; i < input_matrix.size(); ++i) {
        for (int j = 0; j < input_matrix[0].size(); ++j) {
            result[i][j] = output[i * input_matrix[0].size() + j];
        }
    }

    return result;
}

__global void cudaMaxPooling(const float* input, float* output, int n, int m, int pool_size) {
    int out = n/pool_size;
    int out2 = m/pool_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<out && j<out2){
        float max_val = input[i*pool_size*n+j*pool_size];
        for(int x=0;x<pool_size;x++){
            for(int y=0;y<pool_size;y++){
                max_val = fmaxf(max_val, input[(i*pool_size+x)*n+j*pool_size+y]);
            }
        }
        output[i*out+j] = max_val;
    }
}

__global__ void cudaAvgPooling(const float* input, float* output, int n, int m, int pool_size) {
    int out = n/pool_size;
    int out2 = m/pool_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<out && j<out2){
        float sum = 0;
        for(int x=0;x<pool_size;x++){
            for(int y=0;y<pool_size;y++){
                sum += input[(i*pool_size+x)*n+j*pool_size+y];
            }
        }
        output[i*out+j] = sum/(pool_size*pool_size);
    }
}

vector<vector<float> > MaxPooling(vector<vector<float> > &input_matrix,int pool_size,int threads){
    int n = input_matrix.size();
    int m = input_matrix[0].size();
    int out = n/pool_size;
    int out2 = m/pool_size;
    int bytes = out*out2*sizeof(float);
    vector<vector<float> > output(out ,vector<float>(out2,0.0));

    float *d_input, *d_output;
    cudamalloc(&d_input, n*m*sizeof(float));
    cudamalloc(&d_output, bytes);

    cudaMemcpy(d_input, input_matrix.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((out + threadsPerBlock.x - 1) / threadsPerBlock.x, (out2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaMaxPooling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, n, m, pool_size);

    cudaMemcpy(output.data(),d_output,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

vector<vector<float> > AvgPooling(vector<vector<float> > &input_matrix,int pool_size,int threads){
    int n = input_matrix.size();
    int m = input_matrix[0].size();
    int out = n/pool_size;
    int out2 = m/pool_size;
    int bytes = out*out2*sizeof(float);
    vector<vector<float> > output(out ,vector<float>(out2,0.0));

    float *d_input, *d_output;
    cudamalloc(&d_input, n*m*sizeof(float));
    cudamalloc(&d_output, bytes);

    cudaMemcpy(d_input, input_matrix.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((out + threadsPerBlock.x - 1) / threadsPerBlock.x, (out2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaAvgPooling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, n, m, pool_size);

    cudaMemcpy(output.data(),d_output,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}


vector<float> softmax(vector<float> &input_vector){
    float sum = 0.0;
    vector<float> output(input_vector.size(),0.0);
    for(int i=0;i<input_vector.size();i++){
        sum += exp(input_vector[i]);
    }
    for(int i=0;i<input_vector.size();i++){
        output[i] = exp(input_vector[i])/sum;
    }
    return output;
}
    
vector<float> sigmoid(vector<float> &input_vector){
    vector<float> output(input_vector.size(),0.0);
    for(int i=0;i<input_vector.size();i++){
        output[i] = 1/(1+exp(-input_vector[i]));
    }
    return output;
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

int main(int argc, char *argv[]){
    int threads = argv[1];
    vector<vector<float> > input_matrix = {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}};
    vector<vector<float> > kernal = {{1,2,1},{0,0,0},{-1,-2,-1}};
    vector<vector<float> > output_matrix = Convolution_padding(input_matrix, kernal,threads);
    print_matrix(output_matrix);
    output_matrix = Convolution_no_padding(input_matrix, kernal,threads);
    print_matrix(output_matrix);
    print_matrix(relu_activation(output_matrix));
    print_matrix(tanh_activation(output_matrix));

    int pool_size = 2;
    print_matrix(max_pooling(output_matrix, pool_size,threads));
    print_matrix(avg_pooling(output_matrix, pool_size,threads));

    vector<float> input_vector = {1,2,3,4,5};
    vector<float> output_vector = softmax(input_vector);
    for(int i=0;i<output_vector.size();i++){
        cout << output_vector[i] << " ";
    }
    cout << endl;
    vector<float> output_vector = sigmoid(input_vector);
    for(int i=0;i<output_vector.size();i++){
        cout << output_vector[i] << " ";
    }
    cout << endl;


    return 0;
}