#include <bits/stdc++.h>
#include <omp.h>

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
void initialise(vector<vector<double> > &L, vector<vector<double> > &U, vector<int> &P){
    int n = L.size();
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

void LU_Decomposition(vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, vector<int>& P, int num_threads) {
    int n = A.size();

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



void LU_Decomposition1(vector<vector<double> > &A, vector<vector<double> >& L, vector<vector<double> >& U, vector<int>& P) {
    int n = A.size();

    for (int k = 0; k < n; k++) {
        double max = 0.0;
        int k_pivot = -1;

        // Finding the pivot (This part is not parallelized due to data dependencies)
        for (int i = k; i < n; i++) {
            if (abs(A[i][k]) > max) {
                max = abs(A[i][k]);
                k_pivot = i;
            }
        }
        if (k_pivot == -1) {
            throw runtime_error("Singular matrix");
        }

        // Pivoting (This part is not parallelized due to sequential nature)
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

int main() {
    int n;
    cin >> n;

    vector<vector<double> > A(n, vector<double>(n)), A2(n, vector<double>(n));
    vector<vector<double> > L(n, vector<double>(n, 0));
    vector<vector<double> > U(n, vector<double>(n, 0));
    vector<int> P(n);

    // read input matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> A[i][j];
            A2[i][j] = A[i][j];
        }
    }

    // prepare initialisation
    initialise(L, U, P);

    LU_Decomposition(A2, L, U, P, 4);
    vector<vector<double> > PA = permute(A, P); 
    vector<vector<double> > LU = multiply(L, U);
    bool equal = areMatricesEqual(PA, LU);
    cout << "Are PA and LU equaln with epsilon tolerance ? \n" << (equal ? "Yes" : "No") << endl;

    return 0;
}
