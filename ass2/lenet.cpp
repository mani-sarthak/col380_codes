#include <iostream>
#include <fstream>
#include <vector>
#include "functions.hpp"

using namespace std;

typedef vector<vector<data_type>> matrix;

const int IMG_DIMENSION = 28;
const int LAYER_ONE_NODES = 20;
const int LAYER_TWO_NODES = 20;
const int LAYER_THREE_NODES = 50;
const int LAYER_FOUR_NODES = 50;
const int LAYER_FIVE_NODES = 500;
const int LAYER_SIX_NODES = 10;

void read_matrix_from_file(ifstream &fin, matrix &v, int dimension) {

    v.resize(dimension);

    for (int i = 0; i < dimension; i++) {
        v[i].resize(dimension);
        for (int j = 0; j < dimension; j++) {
            fin >> v[i][j];
        }
    }

}

void read_single_matrix(string filename, matrix &v, int dimension) {

    v.resize(dimension);
    ifstream fin(filename);

    for (int i = 0; i < dimension; i++) {
        v[i].resize(dimension);
        for (int j = 0; j < dimension; j++) {
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
    read_single_matrix(img_filename, img_matrix, IMG_DIMENSION);
    // print_matrix(img_matrix);

    string conv1_filename = "./trained_weights/conv1.txt";
    string conv2_filename = "./trained_weights/conv2.txt";

    read_multiple_matrices(conv1_filename, conv1_matrix, conv1_bias, 20, 1, 5);
    read_multiple_matrices(conv2_filename, conv2_matrix, conv2_bias, 50, 20, 5);

    string fc1_filename = "./trained_weights/fc1.txt";
    string fc2_filename = "./trained_weights/fc2.txt";

    read_multiple_matrices(fc1_filename, fc1_matrix, fc1_bias, 500, 50, 4);
    read_multiple_matrices(fc2_filename, fc2_matrix, fc2_bias, 10, 500, 1);

    vector<matrix> layer1(LAYER_ONE_NODES, matrix(24, vector<data_type>(24)));
    for (int i = 0; i < LAYER_ONE_NODES; i++) {
        convolve(img_matrix, conv1_matrix[i][0], layer1[i], true);
        for (int j = 0; j < 24; j++) {
            for (int k = 0; k < 24; k++) {
                layer1[i][j][k] += conv1_bias[i];
            }
        }
    }

    vector<matrix> layer2(LAYER_TWO_NODES, matrix(12, vector<data_type>(12)));
    for (int i = 0; i < LAYER_TWO_NODES; i++) {
        pool_max(layer1[i], 2, layer2[i]);
    }

    vector<matrix> layer3;
    for (int i = 0; i < LAYER_THREE_NODES; i++) {
        matrix output(8, vector<data_type>(8, conv2_bias[i]));
        for (int j = 0; j < LAYER_TWO_NODES; j++) {
            matrix curr_output(8, vector<data_type>(8));
            convolve(layer2[j], conv2_matrix[i][j], curr_output, true);
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    output[x][y] += curr_output[x][y];
                }
            }
        }
        layer3.push_back(output);
    }

    vector<matrix> layer4(LAYER_THREE_NODES, matrix(4, vector<data_type>(4)));
    for (int i = 0; i < LAYER_FOUR_NODES; i++) {
        pool_max(layer3[i], 2, layer4[i]);
    }

    vector<matrix> layer5;
    for (int i = 0; i < LAYER_FIVE_NODES; i++) {
        matrix output(1, vector<data_type>(1, fc1_bias[i]));
        for (int j = 0; j < LAYER_FOUR_NODES; j++) {
            matrix curr_output(1, vector<data_type>(1));
            convolve(layer4[j], fc1_matrix[i][j], curr_output, true);
            // applyActivation(curr_output, relu);
            output[0][0] += curr_output[0][0];
        }
        applyActivation(output, relu);
        layer5.push_back(output);
    }

    vector<data_type> layer6 = fc2_bias;
    for (int i = 0; i < LAYER_SIX_NODES; i++) {
        for (int j = 0; j < LAYER_FIVE_NODES; j++) {
            matrix curr_output(1, vector<data_type>(1));
            convolve(layer5[j], fc2_matrix[i][j], curr_output, true);
            layer6[i] += curr_output[0][0];
        }
    }

    vector<data_type> final = softmax(layer6);

    vector<pair<data_type, int>> predictions(final.size());

    for (int i = 0; i < final.size(); i++) {
        predictions[i] = { 100.0 * final[i], i };
    }

    sort(predictions.rbegin(), predictions.rend());

    for (int i = 0; i < 5; i++) {
        cout << predictions[i].first << " class " << predictions[i].second << '\n';
    }

    return 0;

}
