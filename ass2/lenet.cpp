#include <cassert>
#include <iostream>
#include "functions.hpp"

using namespace std;

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
        matrix temp;
        temp.resize(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1));
        convolve(input1[ind], input2[ind], temp);
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
        pool_max(input[i], pool_size, output[i]);
    }
}


void predict(string image_name){

    matrix img_matrix2D;  
    
    vector<matrix> img_matrix3D(1);

    read_image_file(image_name, img_matrix2D, 28);
    img_matrix3D[0] = img_matrix2D;

    vector<matrix> conv1_output, conv2_output, pool1_output, pool2_output;
    vector<matrix> fc1_output, fc2_output;

    convolve_kernels(img_matrix3D, conv1_matrix, conv1_bias, conv1_output);
    pool3d(conv1_output, 2, pool1_output);
    // cout << "Pool1 Output: " << pool1_output.size() << " " << pool1_output[0].size() << " " << pool1_output[0][0].size() << endl;
    
    convolve_kernels(pool1_output, conv2_matrix, conv2_bias, conv2_output);
    pool3d(conv2_output, 2, pool2_output);
    // cout << "Pool2 Output: " << pool2_output.size() << " " << pool2_output[0].size() << " " << pool2_output[0][0].size() << endl;

    convolve_kernels(pool2_output, fc1_matrix, fc1_bias, fc1_output);
    for (int i = 0; i < fc1_output.size(); i++) {
        apply_activation(fc1_output[i], relu);
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
