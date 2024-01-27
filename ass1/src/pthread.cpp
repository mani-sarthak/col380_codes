#include <pthread.h>
#include <bits/stdc++.h>
#include <chrono>


using namespace std;

#define eps 1e-9
#define endl '\n'
#define check 111

int NUM_THREADS;
int N;

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

// Structure to pass data to threads
typedef struct {
    int thread_id;
    int n;
    double** A;
    double** L;
    double** U;
    int* P;
    int k;
} ThreadData;


void* matrix_update(void* threadarg) {
    ThreadData* my_data = (ThreadData*) threadarg;
    int tid = my_data->thread_id;
    int n = my_data->n;
    int k = my_data->k;
    double** A = my_data->A;
    double** L = my_data->L;
    double** U = my_data->U;
    int start = tid*(n-(k+1))/NUM_THREADS;
    int rows_to_process = (n-(k+1))/NUM_THREADS;
    if (tid == NUM_THREADS-1){
        rows_to_process = n - (k+1) - start;
    }

    // check for loop invariant here
    for(int i=(k+1) + start; i < (k+1)+(tid+1)*(n-(k+1))/NUM_THREADS; i++){		
    // for(int i=(k+1) + start; i < (k+1)+start + rows_to_process; i++){										
        for(int j=k+1; j < n; j++){
            A[i][j] -= L[i][k]*U[k][j];   
        }
    }

    return NULL;
}

void LU_Decomposition_Parallel(double** A, double** L, double** U, int* P, int n) {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data_array[NUM_THREADS];


    // serial code for pivot_search
    for (int k=0; k < n; k++){
        double max = 0.0;
        int k_pivot = -1;
        for (int i=k; i<n; i++){
            if (abs(A[i][k]) > max){
                max = abs(A[i][k]);
                k_pivot = i;
            }
        }
        if (k_pivot == -1){
            throw runtime_error("Singular matrix");
        }

        if (k_pivot != k){
            swap(P[k_pivot], P[k]);
            swap(A[k_pivot], A[k]);
            for (int j=0; j<k; j++){
                swap(L[k_pivot][j], L[k][j]);
            }
        }
        
        
        U[k][k] = A[k][k];
        for (int i=k+1; i<n; i++){
            L[i][k] = A[i][k] / U[k][k];
            U[k][i] = A[k][i];
        }
            
        // parallel section O(n^3)
        for (int i=0; i<NUM_THREADS; i++){
            thread_data_array[i].thread_id = i;
            thread_data_array[i].k = k;
            thread_data_array[i].n = n;
            thread_data_array[i].A = A;
            thread_data_array[i].L = L;
            thread_data_array[i].U = U;
            thread_data_array[i].P = P;

            int rc = pthread_create(&threads[i], NULL, matrix_update, (void*)(thread_data_array + i));
            if (rc != 0) {
                cout << "Error:unable to create thread," << rc << endl;
                exit(-1);
            }
            // for (int j=start; j<end; j++){
            //     A[i][j] = A[i][j] - L[i][k] * U[k][j];
            // }
        }    
        for (int i=0; i<NUM_THREADS; i++){
            pthread_join(threads[i], NULL);
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
            if (abs(A[i][j] - B[i][j]) > eps) return false;
        }
        norm_val += sqrt(col_err);
    }
    return true;
}


int main(int argc, char** argv) {
    // Allocate and initialize matrices A, L, U, and permutation array P
    if (argc < 3) {
        cerr << "Input filename and numthread required as argument\n";
        exit(1);
    }
    ifstream fin(argv[1]);
    int num_threads = stoi(argv[2]);
    NUM_THREADS = num_threads;

    int n;
    fin >> n;
    N = n;

    
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
    initialise(L, U, P, n);

    // Call LU_Decomposition_Parallel
    auto start_time = chrono::high_resolution_clock::now();
    LU_Decomposition_Parallel(A, L, U, P, n);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Execution time: " << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() << " seconds" << endl;
    
    // double** PA = permute(A_copy, P, n);
    // double** LU = multiply(L, U, n);
    // bool equal = areMatricesEqual(PA, LU, n);
    // cout << "Matrix PA\n" << fixed << setprecision(6);
    // printMatrix(PA, n);
    // cout << "\nMatrix L\n";
    // printMatrix(L, n);
    // cout << "\nMatrix U\n";
    // printMatrix(U, n);
    // cout << "\nAre PA and LU equaln with epsilon tolerance ? \n" << (equal ? "Yes" : "No") << endl;
    // cout << scientific << "Norm value: " << norm_val << endl;


    return 0;
}
