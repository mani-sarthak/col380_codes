#include <functional>
#include <vector>

using namespace std;

typedef float data_type;
typedef vector<vector<data_type>> matrix;

void convolve(matrix &input, matrix &kernel, matrix &output, bool SHRINK = true);

void convolve_and_pad(matrix &input, matrix &kernel, matrix &output);

data_type relu(data_type inp);

data_type tanh_activation(data_type inp);

void applyActivation(matrix &mat, function<data_type(data_type)> activation);

void changeMatrixEntry(data_type &inp);

void init_matrix(matrix &mat);

void initialise(matrix &input, int n);

void print_matrix(matrix &mat);

void pool_max(matrix &input, int pool_size, matrix &output);

void pool_avg(matrix &input, int pool_size, matrix &output);

vector<data_type> softmax(vector<data_type> &inp);

vector<data_type> sigmoid(vector<data_type> &inp);

void applyNormalisation(vector<data_type> &inp, vector<data_type> &out, function<vector<data_type> (vector<data_type>)> normalisation);

void fetchSize(int argc, char* argv[]);
