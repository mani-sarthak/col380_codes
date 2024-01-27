#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>

using namespace std;

#define eps 1e-4
#define endl '\n'
#define check 111

// print the matrix of doubles
void printMatrix(const vector<vector<double> >& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << val << "\t";
        }
        cout << endl;
    }
}

//initialisation
void initialise(double** L, double** U, int* P, int n){
    for (int i=0; i<n; i++){
        P[i] = i;
        for (int j=i; j<n; j++){
            U[i][j] = check;
        }
        L[i][i] = 1;
        for (int j=0; j<i; j++){
            L[i][j] = check;
        }
    }
}

// LU Decomposition

void LU_Decomposition(double** A, double** L, double** U, int* P, int n, int num_threads) {

    omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

    for (int k = 0; k < n; k++) {
        double max = 0.0;
        int k_pivot = -1;

        // Finding the pivot
        for (int i = k; i < n; i++) {
            if (abs(A[i][k]) > max) {
                max = abs(A[i][k]);
                k_pivot = i;
            }
        }
        if (k_pivot == -1) {
            throw runtime_error("Singular matrix");
        }

        // Pivoting
        if (k_pivot != k) {
            swap(P[k_pivot], P[k]);
            swap(A[k_pivot], A[k]);
            for (int j = 0; j < k; j++) {
                swap(L[k_pivot][j], L[k][j]);
            }
        }

        U[k][k] = A[k][k];

        // Parallelize the update of L and U
        #pragma omp parallel for shared(A, L, U, k, n)
        for (int i = k + 1; i < n; i++) {
            L[i][k] = A[i][k] / U[k][k];
            U[k][i] = A[k][i];
        }

        // Parallelize the update of matrix A
        #pragma omp parallel for collapse(2) shared(A, L, U, k, n)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }
}


// Function to multiply two matrices
vector<vector<double> > multiply(const vector<vector<double> >& A, const vector<vector<double> >& B) {
    int n = A.size();
    vector<vector<double> > result(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j]; // look for cache coherence in this.
            }
        }
    }
    return result;
}

// Function to apply permutation vector to a matrix
vector<vector<double> > permute(const vector<vector<double> >& A, const vector<int>& P) {
    int n = A.size();
    vector<vector<double> > PA = vector<vector<double> >(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        PA[i] = A[P[i]];
    }
    return PA;
}

// Function to check if two matrices are equal
bool areMatricesEqual(const vector<vector<double> >& A, const vector<vector<double> >& B) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(A[i][j] - B[i][j]) > eps) return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        cerr << "Input filename and numthread required as argument\n";
        exit(1);
    }
    ifstream fin(argv[1]);
    int num_threads = stoi(argv[2]);

    int n;
    fin >> n;

    double** A = (double**)malloc(sizeof(double*) * n);
    double** L = (double**)malloc(sizeof(double*) * n);
    double** U = (double**)malloc(sizeof(double*) * n);
    int* P = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        A[i] = (double*)malloc(sizeof(double) * n);
        L[i] = (double*)malloc(sizeof(double) * n);
        U[i] = (double*)malloc(sizeof(double) * n);
    }

    // read input matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fin >> A[i][j];
        }
    }

    // prepare initialisation
    initialise(L, U, P, n);

    auto start_time = chrono::high_resolution_clock::now();
    LU_Decomposition(A, L, U, P, n, num_threads);
    auto end_time = chrono::high_resolution_clock::now();
	cout << "Execution time: " << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() << " seconds" << endl;
    
    
    // vector<vector<double> > PA = permute(A, P); 
    // vector<vector<double> > LU = multiply(L, U);
    // bool equal = areMatricesEqual(PA, LU);
    // cout << "Are PA and LU equaln with epsilon tolerance ? \n" << (equal ? "Yes" : "No") << endl;

    return 0;
}
