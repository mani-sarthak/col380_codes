#include <bits/stdc++.h>
using namespace std;


typedef float data_type;
typedef vector<vector<data_type> > matrix;


void convolve(matrix &input, matrix &kernel, matrix &output, int pad) {
    int n = input.size();
    int m = input[0].size();
    int k = kernel.size();
    int l = kernel[0].size();
    int o_n = n + (2 * pad) - k + 1;
    int o_m = m + (2 * pad) - l + 1;

    matrix input_padded = input;
    if (pad) {
        int p = pad;
        input_padded.resize(n + 2 * p, vector<data_type>(m + 2 * p, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                input_padded[i + p][j + p] = input[i][j];
            }
        }
    }

    output.resize(o_n, vector<data_type>(o_m, 0));

    for (int i = 0; i < o_n; ++i) {
        for (int j = 0; j < o_m; ++j) {
            for (int m = 0; m < k; ++m) {
                for (int n = 0; n < l; ++n) {
                    output[i][j] += input_padded[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }

}


data_type relu(data_type inp){
    return max(inp, (data_type)0);
}

data_type tanh_activation(data_type inp){
    return tanh(inp);
}


void apply_activation(matrix &mat, function<data_type(data_type)> activation){
    int n = mat.size();
    int m = mat[0].size();
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            mat[i][j] = activation(mat[i][j]);
        }
    }
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
    output.resize(o_n, vector<data_type>(o_m));
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


vector<data_type> softmax(vector<data_type> &inp) {
    vector<data_type> res;
    data_type sum = 1e-9;
    for (int i = 0; i < inp.size(); i++) {
        sum += exp(inp[i]);
    }
    for (int i = 0; i < inp.size(); i++) {
        res.push_back(exp(inp[i]) / sum);
    }
    return res;
}

vector<data_type> sigmoid(vector<data_type> &inp) {
    vector<data_type> res;
    for (int i = 0; i < inp.size(); i++) {
        res.push_back(1 / (1 + exp(-inp[i])));
    }
    return res;
}



int main(int argc, char *argv[]){
    stringstream ss;
    for(int i = 1 ; i < argc; ++i){
	    ss << argv[i] << " ";
    }
    
    int work;
    ss >> work;
    if (work == 1){
        int n, m , p;
        ss >> n >> m >> p;
        int output_size = n + 2*p - m + 1;
        matrix input(n, vector<data_type>(n));
        matrix kernel(m, vector<data_type>(m));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ss >> input[i][j];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                ss >> kernel[i][j];
            }
        }
        matrix output;
        // convolve sahi karo
        convolve(input, kernel, output, p);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                cout << output[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (work == 2){
        int activation_type;
        ss >> activation_type;
        int n, m;
        ss >> n >> m;
        matrix input(n, vector<data_type>(m));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ss >> input[i][j];
            }
        }
        switch (activation_type)
        {
        case 0:
            apply_activation(input, relu);
            break;
        case 1:
            apply_activation(input, tanh_activation);
            break;
        default:
            break;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cout << input[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (work == 3){
        int pool_type;
        ss >> pool_type;
        int n;
        ss >> n;
        matrix input(n, vector<data_type>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ss >> input[i][j];
            }
        }
        int pool_size;
        ss >> pool_size;
        matrix output;
        switch (pool_type)
        {
        case 0:
            pool_max(input, pool_size, output);
            break;
        case 1:
            pool_avg(input, pool_size, output);
            break;
        default:
            break;
        }
        int o_n = n / pool_size;
        for (int i = 0; i < o_n; ++i) {
            for (int j = 0; j < o_n; ++j) {
                cout << output[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (work == 4){
        int normalization_type;
        ss >> normalization_type;
        int n;
        ss >> n;
        vector<data_type> input(n);
        for (int i = 0; i < n; ++i) {
            ss >> input[i];
        }
        vector<data_type> output;

        switch (normalization_type)
        {
        case 0:
            output = sigmoid(input);
            break;
        case 1:
            output = softmax(input);
            break;
        default:
            break;
        }
        for (int i = 0; i < n; ++i) {
            cout << output[i] << " ";
        }
        cout << endl;

    }

}