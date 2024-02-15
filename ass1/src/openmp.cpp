#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>

using namespace std;

#define eps 1e-9
#define endl '\n'
#define check 111

double norm_val = 0.0;

void printMatrix(double** mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << mat[i][j] << " \n"[j == n-1];
        }
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
        double max_element = 0.0;
        int k_pivot = -1;

        // Finding the pivot
        for (int i = k; i < n; i++) {
            double x = abs(A[i][k]);
            if (x > max_element) {
                max_element = x;
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
            double* L_k_row = L[k];
            double* L_pivot_row = L[k_pivot];
            for (int j = 0; j < k; j++) {
                swap(L_pivot_row[j], L_k_row[j]);
            }
        }

        double* U_row = U[k];
        double* A_row = A[k];
        double U_diag = A_row[k];
        U_row[k] = U_diag;

        for (int i = k + 1; i < n; i++){
            L[i][k] = A[i][k] / U_diag;
            U_row[i] = A_row[i];
        }

        // Parallelize the update of matrix A
        #pragma omp parallel for collapse(2) shared(A, L, U, k, n)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= L[i][k] * U_row[j];
            }
        }
    }
}


// Function to multiply two matrices
double** multiply(double** A, double** B, int n) {
    double** result = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i) {
        result[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j]; // look for cache coherence in this.
            }
        }
    }
    return result;
}

// Function to apply permutation vector to a matrix
double** permute(double** A, int* P, int n) {
    double** PA = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i) {
        PA[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            PA[i][j] = A[P[i]][j];
        }
    }
    return PA;
}

// Function to check if two matrices are equal
bool areMatricesEqual(double** A, double** B, int n) {
    for (int j = 0; j < n; ++j) {
        double col_err = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = abs(A[i][j] - B[i][j]);
            col_err += x * x;
            if (x > eps) {
                return false;
            }
        }
        norm_val += sqrt(col_err);
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
    double** A_copy = (double**)malloc(sizeof(double*) * n);
    double** L = (double**)malloc(sizeof(double*) * n);
    double** U = (double**)malloc(sizeof(double*) * n);
    int* P = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        A[i] = (double*)malloc(sizeof(double) * n);
        A_copy[i] = (double*)malloc(sizeof(double) * n);
        L[i] = (double*)malloc(sizeof(double) * n);
        U[i] = (double*)malloc(sizeof(double) * n);
    }

    // read input matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fin >> A[i][j];
            A_copy[i][j] = A[i][j];
        }
    }

    // prepare initialisation
    initialise(L, U, P, n);

    auto start_time = chrono::high_resolution_clock::now();
    LU_Decomposition(A, L, U, P, n, num_threads);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Execution time: " << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() << " seconds" << endl;


    // double** PA = permute(A_copy, P, n);
    // double** LU = multiply(L, U, n);
    // bool equal = areMatricesEqual(PA, LU, n);
    // cout << "Matrix PA\n" << fixed << setprecision(6);
    // printMatrix(PA);
    // cout << "\nMatrix L\n";
    // printMatrix(L, n);
    // cout << "\nMatrix U\n";
    // printMatrix(U, n);
    // cout << "\nAre PA and LU equaln with epsilon tolerance ? \n" << (equal ? "Yes" : "No") << endl;
    // cout << scientific << "Norm value: " << norm_val << endl;

    return 0;
}
