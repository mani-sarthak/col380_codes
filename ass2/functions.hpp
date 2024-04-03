#include <functional>
#include <vector>
#include <string>
using namespace std;

typedef float data_type;
typedef vector<vector<data_type> > matrix;

void convolve(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output, bool pad);

void convolve(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output);

void convolve_and_pad(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output);

data_type relu(data_type inp);

data_type tanh_activation(data_type inp);

void apply_activation(vector<vector<data_type> > &mat, function<data_type(data_type)> activation);

void change_matrix_entry(data_type &inp);

void init_matrix(vector<vector<data_type> > &mat);

void initialise(vector<vector<data_type> > &input, int n);

void print_matrix(vector<vector<data_type> > &mat);

void print_vector(vector<data_type> &v);

void pool_max(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output);

void pool_avg(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output);

vector<data_type> softmax(vector<data_type> &inp);

vector<data_type> sigmoid(vector<data_type> &inp);

void apply_normalisation(vector<data_type> &inp, vector<data_type> &out, function<vector<data_type> (vector<data_type>)> normalisation);

void read_weight_file(string filename, vector<vector<matrix> > &v, vector<data_type> &bias, int matDim, int dim1, int dim2);

void read_image_file(string filename, matrix &v, int dimension);

void sum_matrix(matrix &input1, matrix &input2, matrix &output);

void convolve3d(vector<matrix> &input1, vector<matrix> &input2, data_type bias, matrix &output);

void convolve_kernels(vector<matrix> &input, vector<vector<matrix> >&kernal, vector<data_type> &bias, vector<matrix> &output);

void pool3d(vector<matrix> &input, int pool_size, vector<matrix> &output);
