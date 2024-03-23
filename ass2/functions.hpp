#include <bits/stdc++.h>
using namespace std;

#define data_type double
#define matrix vector<vector<data_type> >


void convolve(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output);


void convolve_and_pad(matrix &input, matrix &kernel, matrix &output);

data_type relu(data_type inp);

data_type tanh_activation(data_type inp);

void applyActivation(matrix &mat, function<data_type(data_type)> activation);

void changeMatrixEntry(data_type &inp);

void init_matrix(matrix &mat);

void initialise(matrix &input, int n);

void print_matrix(matrix &mat);

void print_vector(vector<data_type> &v);

void pool_max(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output);


void pool_avg(matrix &input, int pool_size, matrix &output);

vector<data_type> softmax(vector<data_type> &inp);

vector<data_type> sigmoid(vector<data_type> &inp);

void applyNormalisation(vector<data_type> &inp, vector<data_type> &out, function<vector<data_type> (vector<data_type>)> normalisation);

void readFile(string filename, vector<vector<matrix> > &v, vector<data_type> &bias, int matDim, int dim1, int dim2);