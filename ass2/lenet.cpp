#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

typedef float data_type;
typedef vector<vector<data_type>> matrix;

const int IMG_DIMENSION = 28;

void read_matrix_from_file(ifstream &fin, matrix &v, int dimension) {

    v.resize(dimension);

    for (int i = 0; i < dimension; i++) {
        v[i].resize(dimension);
        for (int j = 0; j < dimension; j++) {
            fin >> v[i][j];
        }
    }

}

void read_single_matrix(string filename, matrix &v, int rows, int columns) {

    v.resize(rows);
    ifstream fin(filename);

    for (int i = 0; i < rows; i++) {
        v[i].resize(columns);
        for (int j = 0; j < columns; j++) {
            fin >> v[i][j];
        }
    }

}

void read_multiple_matrices(string filename, vector<vector<matrix>> &v,
                            vector<data_type> &biases, int output_count,
                            int input_count, int dimension) {

    int bias_count = output_count;
    v.resize(output_count);
    biases.resize(bias_count);
    ifstream fin(filename);

    for (int i = 0; i < output_count; i++) {
        v[i].resize(input_count);
        for (int j = 0; j < input_count; j++) {
            read_matrix_from_file(fin, v[i][j], dimension);
        }
    }

    for (int i = 0; i < bias_count; i++) {
        fin >> biases[i];
    }
}

void print_vector(vector<data_type> &v) {
    for (int it = 0; it < v.size(); it++) {
        cout << v[it] << " \n"[it == v.size()-1];
    }
}

void print_matrix(matrix &v) {
    for (int i = 0; i < v.size(); i++) {
        print_vector(v[i]);
    }
}

void print_multiple_matrices(vector<vector<matrix>> &v) {
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            print_matrix(v[i][j]);
        }
    }
}


int main (int argc, char *argv[]) {

    matrix img_matrix;
    vector<vector<matrix>>
        conv1_matrix,
        conv2_matrix,
        fc1_matrix,
        fc2_matrix;

    vector<data_type>
        conv1_bias,
        conv2_bias,
        fc1_bias,
        fc2_bias;

    string img_filename = "./img.txt";
    read_single_matrix(img_filename, img_matrix, IMG_DIMENSION, IMG_DIMENSION);
    // print_matrix(img_matrix);

    string conv1_filename = "./trained_weights/conv1.txt";
    string conv2_filename = "./trained_weights/conv2.txt";

    read_multiple_matrices(conv1_filename, conv1_matrix, conv1_bias, 20, 1, 5);
    read_multiple_matrices(conv2_filename, conv2_matrix, conv2_bias, 50, 20, 5);

    string fc1_filename = "./trained_weights/fc1.txt";
    string fc2_filename = "./trained_weights/fc2.txt";

    read_multiple_matrices(fc1_filename, fc1_matrix, fc1_bias, 500, 50, 4);
    read_multiple_matrices(fc2_filename, fc2_matrix, fc2_bias, 10, 500, 1);



    return 0;

}
