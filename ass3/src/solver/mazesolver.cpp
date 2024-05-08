#include <bits/stdc++.h>
#include <mpi.h>
#include "dfs.hpp"
#include "dijkstra.hpp"
#include "mazesolver.hpp"
using namespace std;





pair<bool, vector<vector<int> > > solving_algorithm(int type, pair<int, int> start, pair<int, int> end, vector<vector<int> > &maze){
    pair<bool, vector<vector<int> > > ans;
    if (type == 0){
        ans =  solve_dfs(start, end, maze);
    }
    else if (type == 1){
        ans =  solve_dijkstra(start, end, maze);
    }
    return ans;
}

void solve_grid(int type, vector<vector<int> > &grid, int rank, vector<vector<int> >& answer){

    
    int size = grid.size()/NPROCS;
    vector<vector<int> > subgrid(size, vector<int>(size, 0));
    for (int i =0; i<size; i++){
        for (int j=0; j<size; j++){
            if (rank == 0) subgrid[i][j] = grid[i][j];
            else if (rank == 1) subgrid[i][j] = grid[i][j+size];
            else if (rank == 2) subgrid[i][j] = grid[i+size][j];
            else if (rank == 3) subgrid[i][j] = grid[i+size][j+size];
        }
    }
    vector<vector<int> > output(size, vector<int>(size, 0));
    pair<int, int> src, dest;
    pair<bool, vector<vector<int> > > out;

    if (rank == 0) {
        bool found = false;
        for (int i = 0; i < MAZE_DIMENSION / 2 && !found; i++) {
            src = make_pair(i, MAZE_DIMENSION / 2 - 1);
            for (int j = 0; j < MAZE_DIMENSION / 2; j++) {
                dest = make_pair(MAZE_DIMENSION / 2 - 1, j);
                out = solving_algorithm(type, src, dest, subgrid);
                if (out.first) {
                    output = out.second;
                    found = true;
                    break;
                }
            }
        }
    }
    if (rank == 1) {
        src = make_pair(0, MAZE_DIMENSION / 2 - 1);
        for (int i = 0; i < MAZE_DIMENSION / 2; i++) {
            dest = make_pair(i, 0);
            out = solving_algorithm(type, src, dest, subgrid);
            if (out.first) {
                output = out.second;
                break;
            }
        }
    }
    if (rank == 2) {
        dest = make_pair(MAZE_DIMENSION / 2 - 1, 0);
        for (int i = 0; i < MAZE_DIMENSION / 2; i++) {
            src = make_pair(0, i);
            out = solving_algorithm(type, src, dest, subgrid);
            if (out.first) {
                output = out.second;
                break;
            }
        }
    }
    if (rank == 0) output = solving_algorithm(type, make_pair(12, size-1), make_pair(size-1, 26), subgrid).second;
    if (rank == 1) output = solving_algorithm(type, make_pair(0, size-1), make_pair(12, 0), subgrid).second;
    if (rank == 2) output = solving_algorithm(type, make_pair(0, 26), make_pair(size-1, 0), subgrid).second;
    if (rank == 3) output = subgrid;
    int* arr = new int[size*size];
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            arr[i*size + j] = output[i][j];
        }
    }
    if (rank != 3) MPI_Send(arr, size*size, MPI_INT, 3, 0, MPI_COMM_WORLD);
    else {
        int* arr0 = new int[size*size];
        int* arr1 = new int[size*size];
        int* arr2 = new int[size*size];
        MPI_Recv(arr0, size*size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(arr1, size*size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(arr2, size*size, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                answer[i][j] = arr0[i*size + j];
                answer[i][j+size] = arr1[i*size + j];
                answer[i+size][j] = arr2[i*size + j];
                answer[i+size][j+size] = arr[i*size + j];
            }
        }
        // cout << endl << endl;
    }
}
