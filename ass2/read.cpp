#include <bits/stdc++.h>
#include "functions.hpp"

using namespace std;

#define data_type float
#define matrix vector<vector<data_type> >

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



int main(){
    vector<vector<matrix> > v;
    vector<data_type> bias;
    readFile("./trained_weights/fc1.txt", v, bias, 4, 50, 500);

    print_matrix(v[0][0]);
    print_matrix(v[0][1]);
    print_matrix(v[1][0]);
    print_vector(bias);
    return 0;
}