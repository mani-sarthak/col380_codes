#include <functional>
#include <vector>

using namespace std;

typedef float data_type;

void convolve(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output, bool SHRINK = true);

void convolve_and_pad(vector<vector<data_type> > &input, vector<vector<data_type> > &kernel, vector<vector<data_type> > &output);

data_type relu(data_type inp);

data_type tanh_activation(data_type inp);

void applyActivation(vector<vector<data_type> > &mat, function<data_type(data_type)> activation);

void changeMatrixEntry(data_type &inp);

void init_matrix(vector<vector<data_type> > &mat);

void initialise(vector<vector<data_type> > &input, int n);

void print_matrix(vector<vector<data_type> > &mat);

void pool_max(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output);

void pool_avg(vector<vector<data_type> > &input, int pool_size, vector<vector<data_type> > &output);

vector<data_type> softmax(vector<data_type> &inp);

vector<data_type> sigmoid(vector<data_type> &inp);

void applyNormalisation(vector<data_type> &inp, vector<data_type> &out, function<vector<data_type> (vector<data_type>)> normalisation);

void fetchSize(int argc, char* argv[]);
