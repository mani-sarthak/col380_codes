#include <iostream>
#include <fstream>
#include <vector>
#include "functions.hpp"

using namespace std;


void read_single_matrix(string filename, matrix &v, int dimension) {
    v.resize(dimension);

    ifstream fin(filename);

    if (!fin) {
        return -1;
    }

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            fin >> v[i][j];
        }
    }
}


void sumMatrix(matrix &input1, matrix &input2, matrix &output){
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

void convolve3D(vector<matrix> &input1, vector<matrix> &input2, int bias, matrix &output){
    int k1 = input1.size();
    assert(k1 == input2.size());
    int n1 = input1[0].size();
    int m1 = input1[0][0].size();
    int n2 = input2[0].size();
    int m2 = input2[0][0].size();
    output.resize(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1));
    for (int ind = 0; ind < k1; ind++){
        matrix temp;
        temp.resize(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1));
        convolve(input1[ind], input2[ind], temp);
        sumMatrix(output, temp, output);
    }
    matrix bias_matrix(n1 - n2 + 1, vector<data_type>(m1 - m2 + 1, bias));
    sumMatrix(output, bias_matrix, output);
}


void convolveKernels(vector<matrix> &input, vector<vector<matrix> >&kernal, vector<data_type> &bias, vector<matrix> &output){
    int n = input[0].size();
    int k1 = input.size(); 
    int k2 = kernal.size();
    assert(k1 == kernal[0].size());
    assert(k2 == bias.size());
    assert(input[0][0].size() == n);
    output.resize(k2);
    for (int i = 0; i < k2; i++){
        convolve3D(input, kernal[i], bias[i], output[i]);
    }
    assert(output.size() == k2);
}


void pool3D(vector<matrix> &input, int pool_size, vector<matrix> &output){
    int n = input.size();
    output.resize(n);
    for (int i = 0; i < n; i++){
        pool_max(input[i], pool_size, output[i]);
    }
}

void predict(string image_name){
    matrix img_matrix2D;  
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
    
    vector<matrix> img_matrix3D(1);

    read_single_matrix(image_name, img_matrix2D, 28);
    img_matrix3D[0] = img_matrix2D;

    string conv1_filename = "./trained_weights/conv1.txt";
    string conv2_filename = "./trained_weights/conv2.txt";

    readFile(conv1_filename, conv1_matrix, conv1_bias, 5, 1, 20);
    readFile(conv2_filename, conv2_matrix, conv2_bias, 5, 20, 50);


    if (read_single_matrix(img_filename, img_matrix, IMG_DIMENSION)) {
        cerr << "Error opening file " << img_filename << endl;
        return;
    }

    readFile(fc1_filename, fc1_matrix, fc1_bias, 4, 50, 500);
    readFile(fc2_filename, fc2_matrix, fc2_bias, 1, 500, 10);


    vector<matrix> conv1_output, conv2_output, pool1_output, pool2_output;
    vector<matrix> fc1_output, fc2_output;

    // cout << "Conv1 Matrix: " << conv1_matrix.size() << " " << conv1_matrix[0].size() << " " << conv1_matrix[0][0].size() << " " << conv1_matrix[0][0][0].size() << endl;
    // cout << "Conv2 Matrix: " << conv2_matrix.size() << " " << conv2_matrix[0].size() << " " << conv2_matrix[0][0].size() << " " << conv2_matrix[0][0][0].size() << endl;
    // cout << "FC1 Matrix: " << fc1_matrix.size() << " " << fc1_matrix[0].size() << " " << fc1_matrix[0][0].size() << " " << fc1_matrix[0][0][0].size() << endl;
    // cout << "FC2 Matrix: " << fc2_matrix.size() << " " << fc2_matrix[0].size() << " " << fc2_matrix[0][0].size() << " " << fc2_matrix[0][0][0].size() << endl;
    // cout << "Conv1 Bias: " << conv1_bias.size() << endl;
    // cout << "Conv2 Bias: " << conv2_bias.size() << endl;
    // cout << "FC1 Bias: " << fc1_bias.size() << endl;
    // cout << "FC2 Bias: " << fc2_bias.size() << endl;

    convolveKernels(img_matrix3D, conv1_matrix, conv1_bias, conv1_output);
    pool3D(conv1_output, 2, pool1_output);
    // cout << "Pool1 Output: " << pool1_output.size() << " " << pool1_output[0].size() << " " << pool1_output[0][0].size() << endl;
    
    convolveKernels(pool1_output, conv2_matrix, conv2_bias, conv2_output);
    pool3D(conv2_output, 2, pool2_output);
    // cout << "Pool2 Output: " << pool2_output.size() << " " << pool2_output[0].size() << " " << pool2_output[0][0].size() << endl;

    convolveKernels(pool2_output, fc1_matrix, fc1_bias, fc1_output);
    // cout << "FC1 Output: " << fc1_output.size() << " " << fc1_output[0].size() << " " << fc1_output[0][0].size() << endl;
    
    convolveKernels(fc1_output, fc2_matrix, fc2_bias, fc2_output);
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

    return ;
}





int main (int argc, char *argv[]) {
    string filename = "./img.txt";
    predict(filename);
    return 0;
}
