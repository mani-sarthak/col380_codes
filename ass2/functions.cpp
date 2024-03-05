#include <bits/stdc++.h>
using namespace std;


#define data_type float
// contains the serial functions

int INP_N, KER_N, FINAL_N;

//convolution of matrix
// NEED MODIFY THIS FOR PADDING
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
    assert(output.size() == n - k + 1);
    assert(output[0].size() == m - l + 1);
    for (int i = 0; i < n - k + 1; i++){
        for (int j = 0; j < m - l + 1; j++){
            output[i][j] = temp[i + k - 1][j + l - 1];
        }
    }
}



void pad(vector<vector<data_type> > &input, int pad_size, vector<vector<data_type> > &output){
    int n = input.size();
    int m = input[0].size();
    int o_n = n + 2 * pad_size;
    int o_m = m + 2 * pad_size;
    output.resize(o_n, vector<data_type>(o_m));
    for (int i = 0; i < o_n; i++){
        for (int j = 0; j < o_m; j++){
            if (i < pad_size || j < pad_size || i >= n + pad_size || j >= m + pad_size){
                output[i][j] = 0;
            }
            else{
                output[i][j] = input[i - pad_size][j - pad_size];
            }
        }
    }
}


void convolve_and_pad(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output){
    vector<vector<data_type> > padded;
    assert(kernel.size() % 2 == 1);
    assert(kernel.size() == kernel[0].size());
    int pad_size = (kernel.size() - 1) / 2;
    pad(input, pad_size, padded);
    cout << padded.size() << " " << padded[0].size() << endl;
    cout << kernel.size() << " " << kernel[0].size() << endl;
    cout << input.size() << " " << input[0].size() << endl;
    convolve(padded, kernel, output);
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

// initialize the matrices

void changeMatrixEntry(data_type &inp){
    inp = (inp - 0.5) * 2;
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


void initialise(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output){
    input.resize(INP_N, vector<data_type>(INP_N));
    kernel.resize(KER_N, vector<data_type>(KER_N));
    output.resize(FINAL_N, vector<data_type>(FINAL_N));
    init_matrix(input);
    init_matrix(kernel);
}


// print the matrix
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

void fetchSize(int argc, char* argv[]){
    if (argc == 1){
        INP_N = 5;
        KER_N = 3;
        FINAL_N = INP_N - KER_N + 1;
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
    // else if (argc == 4){
    //     INP_N = atoi(argv[1]);
    //     KER_N = atoi(argv[2]);
    //     assert(INP_N - KER_N + 1  > 0);
    //     bool flag = atoi(argv[3]);
    //     if (flag){
    //         FINAL_N = INP_N;
    //     }
    //     else{
    //         FINAL_N = INP_N - KER_N + 1;
    //     }
    // }
    else{
        cout << "Invalid number of arguments" << endl;
    }
}

void pool_max(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output){
    
}

void pool_avg(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output){
    
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

int main(int argc, char* argv[]){

    fetchSize(argc, argv);
    vector<vector<data_type> > input, kernel, output;
    initialise(input, kernel, output);
    for (int i=0; i<kernel.size(); i++){
        for (int j=0; j<kernel.size(); j++){
            kernel[i][j] = 1;
        }
    }


    print_matrix(input);
    print_matrix(kernel);


    // Do the computation

    convolve(input, kernel, output);
    print_matrix(output);
    convolve_and_pad(input, kernel, output);
    print_matrix(output);

    // 


    
    

    applyActivation(output, tanh_activation);
    print_matrix(output);


    return 0;
}