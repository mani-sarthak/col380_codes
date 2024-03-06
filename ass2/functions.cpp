#include <iostream>
#include <cassert>
#include <cmath>
#include "functions.hpp"

using namespace std;

int INP_N, KER_N, FINAL_N;

void convolve(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output, bool SHRINK){
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
    if (SHRINK){
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
    convolve(input, kernel, output, false);
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

void pool_max(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output){
    int n = input.size();
    int m = input[0].size();
    int o_n = n / pool_size;
    int o_m = m / pool_size;
    assert(output.size() == o_n);
    assert(output[0].size() == o_m);
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
    data_type sum = 0;
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

void fetchSize(int argc, char* argv[]){
    if (argc == 1){
        INP_N = 10;
        KER_N = 3;
        // FINAL_N = INP_N - KER_N + 1;
        FINAL_N = INP_N;
    }
    else if (argc ==2){
        INP_N = atoi(argv[1]);
        KER_N = 3;
        assert(INP_N - KER_N + 1  > 0);
        FINAL_N = INP_N - KER_N + 1;
    }
    else if (argc == 3){
        INP_N = atoi(argv[1]);
        KER_N = atoi(argv[2]);
        assert(INP_N - KER_N + 1  > 0);
        FINAL_N = INP_N - KER_N + 1;
    }
    else if (argc == 4){
        INP_N = atoi(argv[1]);
        KER_N = atoi(argv[2]);
        assert(INP_N - KER_N + 1  > 0);
        assert(KER_N % 2 == 1);
        bool SHRINK = (atoi(argv[3]) != 0) ;
        if (SHRINK){
            FINAL_N = INP_N;
        }
        else{
            FINAL_N = INP_N - KER_N + 1;
        }
    }
    else{
        cout << "Invalid number of arguments" << endl;
    }
}

// int main(int argc, char* argv[]){
//
//     fetchSize(argc, argv);
//     vector<vector<data_type> > input, kernel, output, pool;
//     initialise(input, INP_N);
//     initialise(kernel, KER_N);
//     initialise(output, FINAL_N);
//     for (int i=0; i<kernel.size(); i++){
//         for (int j=0; j<kernel.size(); j++){
//             kernel[i][j] = 1;
//         }
//     }
//
//
//     print_matrix(input);
//     print_matrix(kernel);
//
//     if (FINAL_N == INP_N - KER_N + 1){
//         convolve(input, kernel, output);
//     }
//     else{
//         convolve_and_pad(input, kernel, output);
//     }
//     print_matrix(output);
//
//     
//
//     applyActivation(output, tanh_activation);
//     print_matrix(output);
//
//     int pool_size = 2;
//     initialise(pool, FINAL_N / pool_size);
//     pool_max(output, pool_size, pool);
//     print_matrix(pool);
//
//     return 0;
// }
