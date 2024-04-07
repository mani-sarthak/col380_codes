#include<bits/stdc++.h>
#include<dirent.h>
#include<time.h>

#define sz(x) static_cast<int>((x).size())
using namespace std;

ofstream our_output;

//We are flattening the 2d matrix and (x,y) transforms to n*x + y
__global__ void convolve_cuda(float *input_cuda, float *kernel_cuda, float *output_cuda, int input_size, int kernel_size, int depth, float *bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if ((idx < input_size - kernel_size + 1) && (idy < input_size - kernel_size + 1) && (idz < depth)) {
        float sum = 0.0;
        for (int x = 0; x < kernel_size; x++) {
            for (int y = 0; y < kernel_size; y++) {
                sum += input_cuda[(idx + x) * input_size + (idy + y)] * kernel_cuda[idz * kernel_size * kernel_size + x * kernel_size + y];
            }
        }
        output_cuda[idz * (input_size - kernel_size + 1) * (input_size - kernel_size + 1) + idx * (input_size - kernel_size + 1) + idy] = (sum + bias[idz]);
    }
}

__global__ void convolve_cuda_3d_channel(float *input_cuda, float *kernel_cuda, float *output_cuda, int input_size, int kernel_size, int depth, int channels){ 

    //depth is 20 and channels is 50
    
    // blockDim.z is going to be 20*50
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    // idz is from 0 to 999
    //depth is the input depth and channels is the number of kernels

    int self_depth = idz % depth;
    int channel_id = idz / depth;
    int output_size = input_size - kernel_size + 1;
    // channels is 50 (for now), idz_channel lies between (0,49)
    // idz_index lies in (0,19)
    
    if ((idx < input_size - kernel_size + 1) && (idy < input_size - kernel_size + 1) && (channel_id < channels)) {
        float sum = 0.0;
        for (int x = 0; x < kernel_size; x++) {
            for (int y = 0; y < kernel_size; y++) {
                sum += input_cuda[(input_size * input_size * self_depth) + (input_size * (idx + x)) + (idy + y)] * kernel_cuda[(kernel_size * kernel_size * depth * channel_id) + (kernel_size * kernel_size * self_depth) + (kernel_size * x) + y];
            }
        }
        // atomic add
        atomicAdd(&output_cuda[(output_size * output_size * channel_id) + (output_size * idx) + idy], sum);
    }
}

void activate(float* input, int n, string type){   
    if(type == "relu"){
        for(int i=0;i<n;i++){
            input[i] = max(0.0,input[i]);
        }
    }
    else if(type == "sigmoid"){
        for(int i=0;i<n;i++){
            input[i] = 1/(1+exp(-input[i]));
        }
    }
    else if(type == "tanh"){
        for(int i=0;i<n;i++){
            input[i] = tanh(input[i]);
        }
    }
}

__global__ void maxPool_cuda(float *input_cuda, float *output_cuda, int input_size, int pool_size,int num_filters) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_id = blockIdx.z * blockDim.z + threadIdx.z;

    int output_size = input_size / pool_size;

    if ((row < output_size) && (col < output_size) && (filter_id < num_filters)) {
        float maxVal = input_cuda[(filter_id * input_size * input_size) + (row * pool_size * input_size) + (col * pool_size)];
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                maxVal = fmaxf(maxVal, input_cuda[(filter_id * input_size * input_size) + (input_size * (row * pool_size + i)) + (col * pool_size + j)]);
            }
        }
        output_cuda[(filter_id * output_size * output_size) + (output_size * row) + col] = maxVal;
    }
}

__global__ void avgPool_parallel(float *input_cuda, float *output_cuda, int n, int pool_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n / pool_size && col < n / pool_size) {
        float sum = 0.0;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                sum += input_cuda[(row * pool_size + i) * n + (col * pool_size + j)];
            }
        }
        output_cuda[row * n / pool_size + col] = sum / (pool_size * pool_size);
    }
}

void output_probability(float* input, int n){   
    float total = 0.0;
    for(int i=0;i<n;++i) total+=(float)exp(input[i]);
    for(int i=0;i<n;++i) input[i] = 100* (exp(input[i])/total);
}

void load_convolution_values(const char* filename, float* values, int num_values) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_values; ++i) {
        if (!(file >> values[i])) {
            std::cerr << "Error reading value at index " << i << " from file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    file.close();
}

//all these pointers are for conv1 layer
float* conv1_values;
float* conv1_filter;
float* conv1_bias;
float* conv1_filter_cuda;
float* conv1_bias_cuda;

void conv1_init()
{
    int k = 5;
    int image_size = 28;
    int output_size = 24; //(n-k+1)

    int num_filters = 20;
    int num_conv_values = 520;

    conv1_values = new float[num_conv_values];
    load_convolution_values("conv1.txt",conv1_values, num_conv_values);
    conv1_filter = new float[k * k * num_filters];
    conv1_bias = new float[num_filters];

    for(int i=0; i < k * k * num_filters; i++) conv1_filter[i] = conv1_values[i];
    for(int i=0; i < num_filters; ++i) conv1_bias[i] = conv1_values[k * k * num_filters + i];

    cudaMalloc(&conv1_filter_cuda, k * k * num_filters * sizeof(float));
    cudaMemcpy(conv1_filter_cuda, conv1_filter, k * k * num_filters * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&conv1_bias_cuda, num_filters * sizeof(float));
    cudaMemcpy(conv1_bias_cuda, conv1_bias, num_filters * sizeof(float), cudaMemcpyHostToDevice);
}

void conv1_finish()
{
    free(conv1_values);
    free(conv1_filter);
    free(conv1_bias);
    cudaFree(&conv1_filter_cuda);
    cudaFree(&conv1_bias_cuda);
}

float* conv1(float* input){
    int k = 5;
    int image_size = 28;
    int output_size = 24; //(n-k+1)

    int num_filters = 20;
    int num_conv_values = 520;
    
    float* output = new float[output_size * output_size * num_filters];

    //make common memory of input image on CUDA
    float* input_cuda;
    cudaMalloc(&input_cuda, image_size * image_size * sizeof(float));
    cudaMemcpy(input_cuda, input, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);

    float* output_cuda;
    cudaMalloc(&output_cuda, output_size * output_size * num_filters * sizeof(float));

    dim3 num_blocks(1,1,20);
    dim3 threads_per_block(24,24,1);
    // calling kernel code
    convolve_cuda<<<num_blocks,threads_per_block>>> (input_cuda, conv1_filter_cuda, output_cuda, image_size, k, num_filters, conv1_bias_cuda);
    
    cudaMemcpy(output, output_cuda, output_size * output_size * num_filters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(&output_cuda);
    cudaFree(&input_cuda);
    
    return output;
}


float* pool1(float* input_pool1){
    int input_size = 24;
    int num_filters = 20;
    int pool_size = 2;
    int output_size = input_size / pool_size;

    float* output_pool1 = new float[output_size * output_size * num_filters];

    float* input_pool1_cuda;
    cudaMalloc(&input_pool1_cuda, input_size * input_size * num_filters * sizeof(float));
    cudaMemcpy(input_pool1_cuda, input_pool1, input_size * input_size * num_filters * sizeof(float), cudaMemcpyHostToDevice);

    float* output_pool1_cuda;
    cudaMalloc(&output_pool1_cuda, output_size * output_size * num_filters * sizeof(float));
    
    dim3 num_blocks(1,1,num_filters);
    dim3 threads_per_block(output_size,output_size,1);
    // calling kernel code
    maxPool_cuda<<<num_blocks,threads_per_block>>> (input_pool1_cuda, output_pool1_cuda, input_size, pool_size, num_filters);

    cudaMemcpy(output_pool1, output_pool1_cuda, output_size * output_size * num_filters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(&input_pool1_cuda);
    cudaFree(&output_pool1_cuda);
    
    return output_pool1;
}

float* conv2_values;
float* conv2_filter;
float* conv2_bias;
float* conv2_filter_cuda;

void conv2_init()
{
    int kernel_size = 5;
    int input_size = 12;
    int output_size = input_size - kernel_size + 1; //8

    int num_filters_input = 20;
    int num_filters_output = 50;
    int num_conv_values = 25050;

    conv2_values = new float[num_conv_values];
    load_convolution_values("conv2.txt",conv2_values, num_conv_values);
    conv2_filter = new float[num_filters_input * num_filters_output * kernel_size * kernel_size];
    conv2_bias = new float[num_filters_output];

    for(int i=0;i<(kernel_size * kernel_size * num_filters_input * num_filters_output);++i) conv2_filter[i] = conv2_values[i];
    for(int i=0;i<num_filters_output;++i) conv2_bias[i] = conv2_values[kernel_size * kernel_size * num_filters_output * num_filters_input + i];

    cudaMalloc(&conv2_filter_cuda,num_filters_input * num_filters_output * kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(conv2_filter_cuda,conv2_filter,num_filters_input * num_filters_output * kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);
}

void conv2_finish()
{
    free(conv2_values);
    free(conv2_filter);
    free(conv2_bias);
    cudaFree(&conv2_filter_cuda);
}

float* conv2(float* input){
    int kernel_size = 5;
    int input_size = 12;
    int output_size = input_size - kernel_size + 1; //8

    int num_filters_input = 20;
    int num_filters_output = 50;
    int num_conv_values = 25050;

    float* output = new float[output_size * output_size * num_filters_output];
    
    for(int i=0;i<num_filters_output;++i){
        for (int j=0;j<output_size;++j){
            for (int k=0;k<output_size;++k){
                output[output_size * output_size * i + output_size * j + k] = conv2_bias[i];
            }
        }
    }

    float* output_cuda;
    cudaMalloc(&output_cuda, output_size * output_size * num_filters_output * sizeof(float));
    cudaMemcpy(output_cuda, output, output_size * output_size * num_filters_output * sizeof(float), cudaMemcpyHostToDevice);

    float* input_cuda;
    cudaMalloc(&input_cuda, input_size * input_size * num_filters_input * sizeof(float));
    cudaMemcpy(input_cuda, input, input_size * input_size * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);

    dim3 num_blocks(1,1,num_filters_input * num_filters_output);
    dim3 threads_per_block(output_size,output_size,1);

    // calling the kernel code
    convolve_cuda_3d_channel<<<num_blocks,threads_per_block>>> (input_cuda, conv2_filter_cuda, output_cuda, input_size, kernel_size, num_filters_input, num_filters_output);
    
    cudaMemcpy(output, output_cuda, output_size * num_filters_output * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(&output_cuda);
    cudaFree(&input_cuda);
    return output;
}

float* pool2(float* input){
    int input_size = 8;
    int num_filters_input = 50;
    int pool_size = 2;
    int output_size = input_size / pool_size;
    int num_filters_output = 50;

    float* input_cuda;
    cudaMalloc(&input_cuda, input_size * input_size * num_filters_input * sizeof(float));
    cudaMemcpy(input_cuda, input, input_size * input_size * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);

    float* output_cuda;
    cudaMalloc(&output_cuda, output_size * output_size * num_filters_output * sizeof(float));

    float* output = new float[output_size * output_size * num_filters_output];

    dim3 number_blocks(1,1,num_filters_output);
    dim3 threads_per_block(output_size,output_size,1);

    // calling the kernel code
    maxPool_cuda<<<number_blocks, threads_per_block>>> (input_cuda, output_cuda, input_size, pool_size, num_filters_output);
    cudaMemcpy(output, output_cuda, output_size * output_size * num_filters_output * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&input_cuda);
    cudaFree(&output_cuda);
    return output;
}

float* fc1_values;
float* fc1_filter;
float* fc1_bias;
float* fc1_filter_cuda;

void fc1_init()
{
    int kernel_size = 4;
    int input_size = 4;
    int output_size = input_size - kernel_size + 1; // 1

    int num_filters_input = 50;
    int num_filters_output = 500;
    int num_conv_values = num_filters_output * kernel_size * kernel_size * num_filters_input + num_filters_output; // 400500

    fc1_values = new float[num_conv_values];
    load_convolution_values("fc1.txt",fc1_values, num_conv_values);
    fc1_filter = new float[num_filters_input * num_filters_output * kernel_size * kernel_size];
    fc1_bias = new float[num_filters_output];

    for(int i=0;i<kernel_size * kernel_size * num_filters_input * num_filters_output;++i) fc1_filter[i] = fc1_values[i];
    for(int i=0;i<num_filters_output;++i) fc1_bias[i] = fc1_values[kernel_size * kernel_size * num_filters_output * num_filters_input + i];

    cudaMalloc(&fc1_filter_cuda, kernel_size * kernel_size * num_filters_output * num_filters_input* sizeof(float));
    cudaMemcpy(fc1_filter_cuda, fc1_filter, kernel_size * kernel_size * num_filters_output * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);
}

void fc1_finish()
{
    free(fc1_values);
    free(fc1_filter);
    free(fc1_bias);
    cudaFree(&fc1_filter_cuda);
}

float* fc1(float* input){
    int kernel_size = 4;
    int input_size = 4;
    int output_size = input_size - kernel_size + 1; // 1

    int num_filters_input = 50;
    int num_filters_output = 500;
    int num_conv_values = num_filters_output * kernel_size * kernel_size * num_filters_input + num_filters_output; // 400500
    // kernel array contains kernel values and bias contains the bias terms

    float* output = new float[output_size * output_size * num_filters_output];
    
    for(int i=0;i<num_filters_output;++i){
        for (int j=0;j<output_size;++j){
            for (int k=0;k<output_size;++k){
                output[output_size * output_size * i + output_size * j + k] = fc1_bias[i];
            }
        }
    }

    float* output_cuda;
    cudaMalloc(&output_cuda, output_size * output_size * num_filters_output * sizeof(float));
    cudaMemcpy(output_cuda, output, output_size * output_size * num_filters_output * sizeof(float), cudaMemcpyHostToDevice);

    float* input_cuda;
    cudaMalloc(&input_cuda, input_size * input_size * num_filters_input * sizeof(float));
    cudaMemcpy(input_cuda, input, input_size * input_size * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);


    dim3 num_blocks(1,1,num_filters_input * num_filters_output);
    dim3 threads_per_block(output_size,output_size,1);

    // calling the kernel code
    convolve_cuda_3d_channel<<<num_blocks,threads_per_block>>> (input_cuda, fc1_filter_cuda, output_cuda, input_size, kernel_size, num_filters_input, num_filters_output);
    

    cudaMemcpy(output, output_cuda, output_size * num_filters_output * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&output_cuda);
    cudaFree(&input_cuda);
    
    // we have output array here of size 500

    // perform the relu activation on the output array
    activate(output, num_filters_output, "relu");

    return output;
}

float* fc2_values;
float* fc2_filter;
float* fc2_bias;
float* fc2_filter_cuda;

void fc2_init()
{
    int kernel_size = 1;
    int input_size = 1;
    int output_size = input_size - kernel_size + 1; // 1

    int num_filters_input = 500;
    int num_filters_output = 10;
    int num_conv_values = num_filters_output * kernel_size * kernel_size * num_filters_input + num_filters_output; // 5010

    fc2_values = new float[num_conv_values];
    load_convolution_values("fc2.txt",fc2_values, num_conv_values);
    fc2_filter = new float[num_filters_input * num_filters_output * kernel_size * kernel_size];
    fc2_bias = new float[num_filters_output];

    for(int i=0;i<kernel_size * kernel_size * num_filters_input * num_filters_output;++i) fc2_filter[i] = fc2_values[i];
    for(int i=0;i<num_filters_output;++i) fc2_bias[i] = fc2_values[kernel_size * kernel_size * num_filters_output * num_filters_input + i];

    cudaMalloc(&fc2_filter_cuda, kernel_size * kernel_size * num_filters_output * num_filters_input* sizeof(float));
    cudaMemcpy(fc2_filter_cuda, fc2_filter, kernel_size * kernel_size * num_filters_output * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);
}

void fc2_finish()
{
    free(fc2_values);
    free(fc2_filter);
    free(fc2_bias);
    cudaFree(&fc2_filter_cuda);
}

float* fc2(float* input){
    int kernel_size = 1;
    int input_size = 1;
    int output_size = input_size - kernel_size + 1; // 1

    int num_filters_input = 500;
    int num_filters_output = 10;
    int num_conv_values = num_filters_output * kernel_size * kernel_size * num_filters_input + num_filters_output; // 5010

    float* output = new float[output_size * output_size * num_filters_output];
    
    for(int i=0;i<num_filters_output;++i){
        for (int j=0;j<output_size;++j){
            for (int k=0;k<output_size;++k){
                output[output_size * output_size * i + output_size * j + k] = fc2_bias[i];
            }
        }
    }

    float* output_cuda;
    cudaMalloc(&output_cuda, output_size * output_size * num_filters_output * sizeof(float));
    cudaMemcpy(output_cuda, output, output_size * output_size * num_filters_output * sizeof(float), cudaMemcpyHostToDevice);

    float* input_cuda;
    cudaMalloc(&input_cuda, input_size * input_size * num_filters_input * sizeof(float));
    cudaMemcpy(input_cuda, input, input_size * input_size * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);


    dim3 num_blocks(1,1,num_filters_input * num_filters_output);
    dim3 threads_per_block(output_size,output_size,1);

    // calling the kernel code
    convolve_cuda_3d_channel<<<num_blocks,threads_per_block>>> (input_cuda, fc2_filter_cuda, output_cuda, input_size, kernel_size, num_filters_input, num_filters_output);
    

    cudaMemcpy(output, output_cuda, output_size * num_filters_output * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&output_cuda);
    cudaFree(&input_cuda);
    
    // we have output array here of size 10

    // perform the softmax activation on the output array

    output_probability(output, num_filters_output);

    return output;
}

void neural_net_structure(float* input,string filename){
    float* conv1_output = conv1(input);
    float* pool1_output = pool1(conv1_output);
    float* conv2_output = conv2(pool1_output);
    float* pool2_output = pool2(conv2_output);
    float* fc1_output = fc1(pool2_output);
    float* final_output_probabilities = fc2(fc1_output);
    
    // cout<<"conv1 output"<<endl;
    // for(int i=0;i<20;++i){
    //   for(int j=0;j<24;++j){
    //     for(int k=0;k<24;++k){
    //       cout<<conv1_output[i*24*24+j*24+k]<<" ";
    //     }
    //     cout<<endl;
    //   }
    //   cout<<endl<<endl;
    // }

    // cout<<"pool1 output"<<endl;
    // for(int i=0;i<20;++i){
    //   for(int j=0;j<12;++j){
    //     for(int k=0;k<12;++k){
    //       cout<<pool1_output[i*12*12+j*12+k]<<" ";
    //     }
    //     cout<<endl;
    //   }
    //   cout<<endl<<endl;
    // }

    // cout<<"conv2 output"<<endl;
    // for(int i=0;i<50;++i){
    //   for(int j=0;j<8;++j){
    //     for(int k=0;k<8;++k){
    //       cout<<conv2_output[i*8*8+j*8+k]<<" ";
    //     }
    //     cout<<endl;
    //   }
    //   cout<<endl<<endl;
    // }
    // cout<<"pool2 output"<<endl;
    // for(int i=0;i<50;++i){
    //   for(int j=0;j<4;++j){
    //     for(int k=0;k<4;++k){
    //       cout<<pool2_output[i*4*4+j*4+k]<<" ";
    //     }
    //     cout<<endl;
    //   }
    //   cout<<endl<<endl;
    // }
    // cout<<"fc1 output"<<endl;
    // for(int i=0;i<500;++i){
    //   cout<<fc1_output[i]<<" ";
    // }
    // cout<<endl;
    // cout<<endl;

    float max_prob = 0.0;
    int ans_number = 0;

    for(int i=0;i<10;++i){
        if(final_output_probabilities[i]>max_prob)
        {
            max_prob = final_output_probabilities[i];
            ans_number = i;
        }
    }

    our_output<<filename<<" "<<ans_number<<endl;

    free(conv1_output);
    free(pool1_output);
    free(conv2_output);
    free(pool2_output);
    free(fc1_output);
    free(final_output_probabilities);
}

int main()
{   // input image of size 28*28
    // int image_size = 28;
    // float* input = new float[image_size*image_size];
    // for(int i=0;i<28;++i){
    //     for(int j=0;j<28;++j){
    //         input[i*28+j] = 1.0;
    //     }
    // }

    conv1_init();
    conv2_init();
    fc1_init();
    fc2_init();
    our_output.open("our_output.txt");
    int image_size = 28;

    string folderPath = "test_output/";

    struct timespec start, end;
    long timediff;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i=0;i<10000;++i) 
    {
        string filename = folderPath;
        stringstream ss;
        ss << i;
        string str = ss.str();
        for(int j=0; j<6-str.length();++j)filename+='0';
        filename+=str;
        filename+=".txt";
        float* input = new float[image_size*image_size];
        load_convolution_values(filename.c_str(),input,image_size*image_size); 
        neural_net_structure(input,filename);
        free(input);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    timediff = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    std::cout << "Time taken: " << timediff << " milliseconds" << std::endl;

    our_output.close();
    conv1_finish();
    conv2_finish();
    fc1_finish();
    fc2_finish();
}