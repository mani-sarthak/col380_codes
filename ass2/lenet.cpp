#include <iostream>
#include <fstream>
#include <vector>
#include "functions.hpp"

using namespace std;

const int IMG_DIMENSION = 28;
const int LAYER_ZERO_NODES = 1;
const int LAYER_ONE_NODES = 20;
const int LAYER_TWO_NODES = 20;
const int LAYER_THREE_NODES = 50;
const int LAYER_FOUR_NODES = 50;
const int LAYER_FIVE_NODES = 500;
const int LAYER_SIX_NODES = 10;
const int LAYER_ONE_KERNEL_SIZE = 5;
const int LAYER_THREE_KERNEL_SIZE = 5;
const int LAYER_FIVE_KERNEL_SIZE = 4;
const int LAYER_SIX_KERNEL_SIZE = 1;

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

void read_matrix_from_file(ifstream &fin, matrix &v, int dimension) {

    initialise(v, dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            fin >> v[i][j];
        }
    }

}

int read_single_matrix(string filename, matrix &v, int dimension) {

    initialise(v, dimension);
    ifstream fin(filename);

    if (!fin) {
        return -1;
    }

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            fin >> v[i][j];
        }
    }
    return 0;

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

void predict_number(string img_filename) {

    matrix img_matrix;

    if (read_single_matrix(img_filename, img_matrix, IMG_DIMENSION)) {
        cerr << "Error opening file " << img_filename << endl;
        return;
    }

    // Layer 0 - image matrix 28x28

    // Layer 1 - convolution with 5x5
    // 1 | 28x28 -> 20 | 24x24
    vector<matrix> layer1(LAYER_ONE_NODES, matrix(24, vector<data_type>(24)));
    for (int i = 0; i < LAYER_ONE_NODES; i++) {
        convolve(img_matrix, conv1_matrix[i][0], layer1[i]);
        for (int j = 0; j < 24; j++) {
            for (int k = 0; k < 24; k++) {
                layer1[i][j][k] += conv1_bias[i];
            }
        }
    }

    // Layer 2 - max pooling with stride 2
    // 20 | 24x24 -> 20 | 12x12
    vector<matrix> layer2(LAYER_TWO_NODES, matrix(12, vector<data_type>(12)));
    for (int i = 0; i < LAYER_TWO_NODES; i++) {
        pool_max(layer1[i], 2, layer2[i]);
    }

    // Layer 3 - convolution with 5x5
    // 20 | 12x12 -> 50 | 8x8
    vector<matrix> layer3;
    for (int i = 0; i < LAYER_THREE_NODES; i++) {
        matrix output(8, vector<data_type>(8, conv2_bias[i]));
        matrix curr_output(8, vector<data_type>(8));
        for (int j = 0; j < LAYER_TWO_NODES; j++) {
            convolve(layer2[j], conv2_matrix[i][j], curr_output);
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    output[x][y] += curr_output[x][y];
                }
            }
        }
        layer3.push_back(output);
    }

    // Layer 4 - max pooling with stride 2
    // 50 | 8x8 -> 50 | 4x4
    vector<matrix> layer4(LAYER_THREE_NODES, matrix(4, vector<data_type>(4)));
    for (int i = 0; i < LAYER_FOUR_NODES; i++) {
        pool_max(layer3[i], 2, layer4[i]);
    }

    // Layer 5 - inner product
    // 50 | 4x4 -> 500 | 1x1
    vector<matrix> layer5;
    for (int i = 0; i < LAYER_FIVE_NODES; i++) {
        matrix output(1, vector<data_type>(1, fc1_bias[i]));
        // matrix curr_output(1, vector<data_type>(1));
        for (int j = 0; j < LAYER_FOUR_NODES; j++) {
            // convolve(layer4[j], fc1_matrix[i][j], curr_output);
            // output[0][0] += curr_output[0][0];
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    output[0][0] += layer4[j][x][y] * fc1_matrix[i][j][x][y];
                }
            }
        }
        // apply relu to 500 output matrices
        applyActivation(output, relu);
        layer5.push_back(output);
    }

    // Layer 6 - inner product
    // 500 | 1x1 -> 10 | 1x1
    vector<data_type> layer6 = fc2_bias;
    for (int i = 0; i < LAYER_SIX_NODES; i++) {
        // matrix curr_output(1, vector<data_type>(1));
        for (int j = 0; j < LAYER_FIVE_NODES; j++) {
            // convolve(layer5[j], fc2_matrix[i][j], curr_output);
            // layer6[i] += curr_output[0][0];
            layer6[i] += layer5[j][0][0] * fc2_matrix[i][j][0][0];
        }
    }

    // apply softmax for final probability
    vector<data_type> probability = softmax(layer6);

    vector<pair<data_type, int>> predictions(probability.size());

    for (int i = 0; i < predictions.size(); i++) {
        predictions[i] = { 100.0 * probability[i], i };
    }

    sort(predictions.rbegin(), predictions.rend());

    for (int i = 0; i < 5; i++) {
        cout << predictions[i].first << " class " << predictions[i].second << '\n';
    }
}


int main (int argc, char *argv[]) {

    if (argc < 2) {
        cerr << "Image filename(s) required in argument\n";
        exit(1);
    }

    string conv1_filename = "./trained_weights/conv1.txt";
    string conv2_filename = "./trained_weights/conv2.txt";

    read_multiple_matrices(conv1_filename, conv1_matrix, conv1_bias,
                           LAYER_ONE_NODES, LAYER_ZERO_NODES, LAYER_ONE_KERNEL_SIZE);
    read_multiple_matrices(conv2_filename, conv2_matrix, conv2_bias,
                           LAYER_THREE_NODES, LAYER_TWO_NODES, LAYER_THREE_KERNEL_SIZE);

    string fc1_filename = "./trained_weights/fc1.txt";
    string fc2_filename = "./trained_weights/fc2.txt";

    read_multiple_matrices(fc1_filename, fc1_matrix, fc1_bias,
                           LAYER_FIVE_NODES, LAYER_FOUR_NODES, LAYER_FIVE_KERNEL_SIZE);
    read_multiple_matrices(fc2_filename, fc2_matrix, fc2_bias,
                           LAYER_SIX_NODES, LAYER_FIVE_NODES, LAYER_SIX_KERNEL_SIZE);

    for (int i = 1; i < argc; i++) {
        predict_number(argv[i]);
    }

    return 0;

}
