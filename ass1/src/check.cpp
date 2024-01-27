#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

using namespace std;

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

int main() {
    int n, num_threads;

    cout << "Enter the size of the matrix: ";
    cin >> n;

    cout << "Enter the number of threads: ";
    cin >> num_threads;

    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<vector<double>> U(n, vector<double>(n, 0));
    vector<int> P(n);

    cout << "Enter matrix A row by row:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    try {
        LU_Decomposition(A, L, U, P, num_threads);

        cout << "L matrix:" << endl;
        for (const auto& row : L) {
            for (const auto& elem : row) {
                cout << elem << " ";
            }
            cout << endl;
        }

        cout << "U matrix:" << endl;
        for (const auto& row : U) {
            for (const auto& elem : row) {
                cout << elem << " ";
            }
            cout << endl;
        }

        cout << "Permutation vector:" << endl;
        for (const auto& elem : P) {
            cout << elem << " ";
        }
        cout << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
