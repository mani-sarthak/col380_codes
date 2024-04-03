#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include "functions.hpp"

using namespace std;

void convolve(matrix &input, matrix &kernel, matrix &output, bool pad) {
    int n = input.size();
    int m = input[0].size();
    int k = kernel.size();
    int l = kernel[0].size();
    int o_n = n - k + 1;
    int o_m = m - l + 1;

    matrix input_padded = input;
    if (pad) {
        int p = (k - 1) / 2;
        input_padded.resize(n + 2 * p, vector<data_type>(n + 2 * p, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                input_padded[i + p][j + p] = input[i][j];
            }
        }
        o_n = n;
        o_m = m;
        n += 2 * p;
    }

    output.assign(o_n, vector<data_type>(o_n, 0));

    for (int i = 0; i <= n - k; ++i) {
        for (int j = 0; j <= m - l; ++j) {
            for (int x = 0; x < k; ++x) {
                for (int y = 0; y < l; ++y) {
                    output[i][j] += input_padded[i + x][j + y] * kernel[x][y];
                }
            }
        }
    }

}

void convolve(matrix &input, matrix &kernel, matrix &output){
    convolve(input, kernel, output, false);
}

void convolve_and_pad(matrix &input, matrix &kernel, matrix &output){
    convolve(input, kernel, output, true);
}

data_type relu(data_type inp){
    return max(inp, (data_type)0);
}

data_type tanh_activation(data_type inp){
    return tanh(inp);
}

void apply_activation(matrix &mat, function<data_type(data_type)> activation){
    int size = mat.size();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            mat[i][j] = activation(mat[i][j]);
        }
    }
}

void change_matrix_entry(data_type &inp){

}

void init_matrix(matrix &mat){
    int size = mat.size();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            mat[i][j] = (data_type)rand() / (data_type)RAND_MAX;
            change_matrix_entry(mat[i][j]);
        }
    }
}


void initialise(matrix &input, int n){
    input.resize(n, vector<data_type>(n));
    init_matrix(input);
}


void print_matrix(matrix &mat){
    int size = mat.size();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


void print_vector(vector<data_type> &v){
    for (int i = 0; i < v.size(); i++){
        cout << v[i] << " ";
    }
    cout << endl;
}

void pool_max(matrix &input, int pool_size, matrix &output){
    int n = input.size();
    int m = input[0].size();
    int o_n = n / pool_size;
    int o_m = m / pool_size;
    output.resize(o_n, vector<data_type>(o_m));
    for (int i = 0; i < o_n; i++){
        for (int j = 0; j < o_m; j++){
            data_type max_val = -1e9;
            for (int x = i*pool_size; x < (i+1)*pool_size; x++){
                for (int y = j*pool_size; y < (j+1)*pool_size; y++){
                    max_val = max(max_val, input[x][y]);
                }
            }
            output[i][j] = max_val;
        }
    }
}

void pool_avg(matrix &input, int pool_size, matrix &output){
    int n = input.size();
    int m = input[0].size();
    int o_n = n / pool_size;
    int o_m = m / pool_size;
    assert(output.size() == o_n);
    assert(output[0].size() == o_m);
    for (int i = 0; i < o_n; i++){
        for (int j = 0; j < o_m; j++){
            data_type sum = 0;
            for (int x = i*pool_size; x < (i+1)*pool_size; x++){
                for (int y = j*pool_size; y < (j+1)*pool_size; y++){
                    sum += input[x][y];
                }
            }
            output[i][j] = sum / (pool_size * pool_size);
        }
    }
}

vector<data_type> softmax(vector<data_type> &inp){
    vector<data_type> res;
    data_type sum = 1e-9;
    for (int i=0; i<inp.size(); i++){
        sum += exp(inp[i]);
    }
    for (int i=0; i<inp.size(); i++){
        res.push_back(exp(inp[i]) / sum);
    }
    return res;
}

vector<data_type> sigmoid(vector<data_type> &inp){
    vector<data_type> res;
    for (int i=0; i<inp.size(); i++){
        res.push_back(1 / (1 + exp(-inp[i])));
    }
    return res;
}

void apply_normalisation(vector<data_type> &inp, vector<data_type> &out, function<vector<data_type> (vector<data_type>)> normalisation){
    out = normalisation(inp);
}

// read ((matDim x matDim) * dim1 ) * dim2 matrices from file
// eg. ((5 * 5) x 20 ) x 50; v.size() = 50, v[0].size() = 20
// also have biases of dim2 size
// read_file("./trained_weights/conv2.txt", v, bias, 5, 20, 50);
void read_weight_file(string filename, vector<vector<matrix> > &v, vector<data_type> &bias, int matDim, int dim1, int dim2) {

    ifstream fin(filename);
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
    ifstream fin(filename);

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
