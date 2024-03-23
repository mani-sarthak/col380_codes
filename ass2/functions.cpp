#include <iostream>
#include <cassert>
#include <cmath>
#include "functions.hpp"

using namespace std;

int INP_N, KER_N, FINAL_N;

void convolve(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output){
    int n = input.size();
    int m = input[0].size();
    int k = kernel.size();
    int l = kernel[0].size();
    int o_n = n + k - 1;
    int o_m = m + l - 1;
    vector<vector<data_type> > temp(o_n, vector<data_type>(o_m));
    for (int x = 0; x < o_n; x++){
        for (int y = 0; y < o_m; y++){
            temp[x][y] = 0;
            for (int u=0; u<=x; u++){
                for (int v=0; v <= y; v++){
                    if (u < n && v < m && x-u < k && y-v < l) 
                    temp[x][y] += (input[u][v] * kernel[x-u][y-v]);
                }
            }
        }
    }
    if (true){
        assert(output.size() == n - k + 1);
        assert(output[0].size() == m - l + 1);
        for (int i = 0; i < n - k + 1; i++){
            for (int j = 0; j < m - l + 1; j++){
                output[i][j] = temp[i + k - 1][j + l - 1];
            }
        }
    }
    else {
        int pad_size = (kernel.size() - 1) / 2;
        assert(output.size() == o_n - 2*pad_size );
        assert(output[0].size() == o_m - 2*pad_size );
        for (int i = 0; i < n ; i++){
            for (int j = 0; j < m ; j++){
                output[i][j] = temp[i + pad_size][j + pad_size];
            }
        }
    }
}

void convolve_and_pad(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output){
    convolve(input, kernel, output);
}

data_type relu(data_type inp){
    return max(inp, (data_type)0);
}

data_type tanh_activation(data_type inp){
    return tanh(inp);
}

void applyActivation(vector<vector<data_type> > &mat, function<data_type(data_type)> activation){
    int size = mat.size();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            mat[i][j] = activation(mat[i][j]);
        }
    }
}

void changeMatrixEntry(data_type &inp){

}

void init_matrix(vector<vector<data_type> > &mat){
    int size = mat.size();
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            mat[i][j] = (data_type)rand() / (data_type)RAND_MAX;
            changeMatrixEntry(mat[i][j]);
        }
    }
}


void initialise(vector<vector<data_type> > &input, int n){
    input.resize(n, vector<data_type>(n));
    init_matrix(input);
}


void print_matrix(vector<vector<data_type> > &mat){
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

void pool_max(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output){
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

void pool_avg(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output){
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

void applyNormalisation(vector<data_type> &inp, vector<data_type> &out, function<vector<data_type> (vector<data_type>)> normalisation){
    out = normalisation(inp);
}

// read ((matDim x matDim) * dim1 ) * dim2 matrices from file
// eg. ((5 * 5) x 20 ) x 50; v.size() = 50, v[0].size() = 20
// also have biases of dim2 size
// readFile("./trained_weights/conv2.txt", v, bias, 5, 20, 50);
void readFile(string filename, vector<vector<matrix> > &v, vector<data_type> &bias, int matDim, int dim1, int dim2) {
    ifstream fin(filename);
    v.resize(dim2);
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
    string temp;
    fin >> temp;
    assert(temp == "");
}
