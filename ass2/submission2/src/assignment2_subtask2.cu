#include <bits/stdc++.h>
using namespace std;

typedef float data_type;

__global__ void convolve_parallel(float *input_cuda, float *kernel_cuda, float *output_cuda, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n - k + 1 && idy < n - k + 1) {
        float sum = 0.0;
        for (int x = 0; x < k; x++) {
            for (int y = 0; y < k; y++) {
                sum += input_cuda[(idx + x) * n + (idy + y)] * kernel_cuda[x * k + y];
            }
        }
        output_cuda[idx * (n - k + 1) + idy] = sum;
    }
}



__global__ void activate_parallel(float *input_cuda, float *output_cuda, int n, int m, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < m) {
        int index = idx * n + idy;
        if (type == 0) {
            output_cuda[index] = max(0.0, input_cuda[index]);
        } else if (type == 1) {
            output_cuda[index] = tanhf(input_cuda[index]);
        }
    }
}


__global__ void maxPool_parallel(float *input_cuda, float *output_cuda, int inputWidth, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputWidth / poolSize && col < inputWidth / poolSize) {
        float maxVal = 0.0;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                maxVal = fmaxf(maxVal, input_cuda[(row * poolSize + i) * inputWidth + (col * poolSize + j)]);
            }
        }
        output_cuda[row * inputWidth / poolSize + col] = maxVal;
    }
}


__global__ void avgPool_parallel(float *input, float *output, int n, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n / poolSize && col < n / poolSize) {
        float sum = 0.0;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                sum += input[(row * poolSize + i) * n + (col * poolSize + j)];
            }
        }
        output[row * n / poolSize + col] = sum / (poolSize * poolSize);
    }
}

__global__ void normalize_sigmoid(float *input_cuda, float *output_cuda, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output_cuda[idx] = 1/(1+exp(-input_cuda[idx]));
}

void normalize_softmax(float *input, float *output, int n){
    float total = 0.0;
    for(int i=0;i<n;++i) total+=(float)exp(input[i]);
    for(int i=0;i<n;++i) output[i] = exp(input[i])/total;
}



int main(int argc, char **argv){
    stringstream ss;
    for(int i = 1 ; i < argc; ++i){
	    ss << argv[i] << " ";
    }
    int work;
    ss >> work;
    if (work == 1){
        int n, m, p;
        ss >> n >> m >> p;
        int size = n + 2 * p;
        int output_size = size - m + 1;
        data_type *input = new data_type[size * size];
        data_type *kernel = new data_type[m * m];
        data_type *output = new data_type[output_size * output_size];
        for (int i = p; i < n + p; i++){
            for (int j = p; j < n + p; j++){
                ss >> input[i * (size) + j];
            }
        }
        for (int i = 0; i < m; i++){
            for (int j = 0; j < m; j++){
                ss >> kernel[i * m + j];
            }
        }
        data_type *input_cuda, *kernel_cuda, *output_cuda;
        cudaMalloc(&input_cuda, size * size * sizeof(data_type));
        cudaMalloc(&kernel_cuda, m * m * sizeof(data_type));
        cudaMalloc(&output_cuda, output_size * output_size * sizeof(data_type));
        cudaMemcpy(input_cuda, input, size * size * sizeof(data_type), cudaMemcpyHostToDevice);
        cudaMemcpy(kernel_cuda, kernel, m * m * sizeof(data_type), cudaMemcpyHostToDevice);
        
        dim3 block_size(16, 16);
        dim3 grid_size((output_size + block_size.x - 1) / block_size.x, (output_size + block_size.y - 1) / block_size.y);

        convolve_parallel<<<grid_size, block_size>>>(input_cuda, kernel_cuda, output_cuda, size, m);
        cudaMemcpy(output, output_cuda, output_size * output_size * sizeof(data_type), cudaMemcpyDeviceToHost);

        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){
                cout << output[i * output_size + j] << " ";
            }
            cout << endl;
        }

        delete[] input;
        delete[] kernel;
        delete[] output;
        cudaFree(input_cuda);
        cudaFree(kernel_cuda);
        cudaFree(output_cuda);
    }
    else if (work == 2){
        int activation_type;
        ss >> activation_type;
        int n, m;
        ss >> n >> m;
        data_type *input = new data_type[n * m];
        data_type *output = new data_type[n * m];
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                ss >> input[i * m + j];
            }
        }
        data_type *input_cuda, *output_cuda;
        cudaMalloc(&input_cuda, n * m * sizeof(data_type));
        cudaMalloc(&output_cuda, n * m * sizeof(data_type));
        cudaMemcpy(input_cuda, input, n * m * sizeof(data_type), cudaMemcpyHostToDevice);

        dim3 block_size(16, 16);
        dim3 grid_size((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);

        if (activation_type == 0){
            activate_parallel<<<grid_size, block_size>>>(input_cuda, output_cuda, n, m, 0);
        }
        else if (activation_type == 1){
            activate_parallel<<<grid_size, block_size>>>(input_cuda, output_cuda, n, m,  1);
        }

        cudaMemcpy(output, output_cuda, n * m * sizeof(data_type), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                cout << output[i * m + j] << " ";
            }
            cout << endl;
        }
        delete[] input;
        delete[] output;
        cudaFree(input_cuda);
        cudaFree(output_cuda);
    }
    else if (work == 3){
        int pool_type;
        ss >> pool_type;
        int n;
        ss >> n;
        data_type *input = new data_type[n * n];
        int pool_size;
        ss >> pool_size;
        int output_size = n / pool_size;
        data_type *output = new data_type[output_size * output_size];
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                ss >> input[i * n + j];
            }
        }
        data_type *input_cuda, *output_cuda;
        cudaMalloc(&input_cuda, n * n * sizeof(data_type));
        cudaMalloc(&output_cuda, output_size * output_size * sizeof(data_type));
        cudaMemcpy(input_cuda, input, n * n * sizeof(data_type), cudaMemcpyHostToDevice);

        dim3 block_size(16, 16);
        dim3 grid_size((output_size + block_size.x - 1) / block_size.x, (output_size + block_size.y - 1) / block_size.y);

        if (pool_type == 0){
            maxPool_parallel<<<grid_size, block_size>>>(input_cuda, output_cuda, n, pool_size);
        }
        else if (pool_type == 1){
            avgPool_parallel<<<grid_size, block_size>>>(input_cuda, output_cuda, n, pool_size);
        }

        cudaMemcpy(output, output_cuda, output_size * output_size * sizeof(data_type), cudaMemcpyDeviceToHost);
        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){
                cout << output[i * output_size + j] << " ";
            }
            cout << endl;
        }
        delete[] input;
        delete[] output;
        cudaFree(input_cuda);
        cudaFree(output_cuda);
    }
    else if (work == 4){
        int normalization_type;
        ss >> normalization_type;
        int n;
        ss >> n;
        data_type *input = new data_type[n];
        data_type *output = new data_type[n];
        for (int i = 0; i < n; i++){
            ss >> input[i];
        }
        data_type *input_cuda, *output_cuda;
        cudaMalloc(&input_cuda, n * sizeof(data_type));
        cudaMalloc(&output_cuda, n * sizeof(data_type));
        cudaMemcpy(input_cuda, input, n * sizeof(data_type), cudaMemcpyHostToDevice);

        dim3 block_size(16, 16);
        dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
        
        if (normalization_type == 0){
            normalize_sigmoid<<<grid_size, block_size>>>(input_cuda, output_cuda, n);
            cudaMemcpy(output, output_cuda, n * sizeof(data_type), cudaMemcpyDeviceToHost);
        }
        else if (normalization_type == 1){
            normalize_softmax(input, output, n);
        }

        for (int i = 0; i < n; i++){
            cout << output[i] << " ";
        }
        cout << endl;
        delete[] input;
        delete[] output;
        cudaFree(input_cuda);
        cudaFree(output_cuda);
    }   
}