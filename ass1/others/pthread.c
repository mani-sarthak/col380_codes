#include <math.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>


#define eps 1e-9
#define endl '\n'
#define check 111

int NUM_THREADS;

double norm_val = 0.0;

void printMatrix(double** mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
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
        double x = L[i][k];
        double* A_row = A[i];
        double* U_row = U[k];
        for(int j=k+1; j < n; j++){
            A_row[j] -= x * U_row[j];   
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
            if (fabs(A[i][k]) > max){
                max = fabs(A[i][k]);
                k_pivot = i;
            }
        }
        if (k_pivot == -1){
            exit(1);
            // throw runtime_error("Singular matrix");
        }

        if (k_pivot != k){
            int tmp_int = P[k];
            P[k] = P[k_pivot];
            P[k_pivot] = tmp_int;
            double* tmp_ptr = A[k];
            A[k] = A[k_pivot];
            A[k_pivot] = tmp_ptr;
            for (int j=0; j<k; j++){
                double tmp_double = L[k][j];
                L[k][j] = L[k_pivot][j];
                L[k_pivot][j] = L[k][j];
            }
        }

        U[k][k] = A[k][k];
        double U_diag = U[k][k];
        double* U_row = U[k];
        double* A_row = A[k];
        for (int i=k+1; i<n; i++){
            L[i][k] = A[i][k] / U_diag;
            U_row[i] = A_row[i];
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
                printf("Error:unable to create thread\n");
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
int areMatricesEqual(double** A, double** B, int n) {
    for (int j = 0; j < n; ++j) {
        double col_err = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = fabs(A[i][j] - B[i][j]);
            col_err += x * x;
            if (x > eps) {
                return 0;
            }
        }
        norm_val += sqrt(col_err);
    }
    return 1;
}


int main(int argc, char** argv) {

    if (argc < 3) {
        printf("Input filename and numthread required as argument\n");
        exit(1);
    }

    FILE* input_file = fopen(argv[1], "r");
    if (input_file == NULL) {
        printf("Invalid input filename\n");
        exit(1);
    }

    int num_threads = atoi(argv[2]);
    NUM_THREADS = num_threads;

    int n;
    fscanf(input_file, "%d\n", &n);

    // Allocate and initialize matrices A, L, U, and permutation array P
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
            fscanf(input_file, "%lf", &A[i][j]);
            A_copy[i][j] = A[i][j];
        }
    }
    initialise(L, U, P, n);

    // Call LU_Decomposition_Parallel
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);
    LU_Decomposition_Parallel(A, L, U, P, n);
    gettimeofday(&end_time, 0);
    long seconds = end_time.tv_sec - start_time.tv_sec;
    long microseconds = end_time.tv_usec - start_time.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf("Time measured: %.6f seconds.\n", elapsed);

    // double** PA = permute(A_copy, P, n);
    // double** LU = multiply(L, U, n);
    // int equal = areMatricesEqual(PA, LU, n);
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
