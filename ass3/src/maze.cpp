#include <bits/stdc++.h>
#include "generator/mazegenerator.hpp"
#include "solver/mazesolver.hpp"
#include <mpi.h>

using namespace std;

vector<vector<int> > answer(MAZE_DIMENSION, vector<int>(MAZE_DIMENSION, 0));

void modify_maze(vector<vector<int> > &grid){
    for (int i = 0; i < grid.size(); i++){
        for (int j = 0; j < grid[i].size(); j++){
            grid[i][j] ^= 1;
            // cout << grid[i][j] << " ";
        }
        // cout << endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    int comm_sz, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char generator_type[20]; // For storing algorithm type (-g option)
    char search_type[20];    // For storing search type (-s option)

    if (argc != 5){
        cout << "Usage: mpirun -np 4 ./maze.out -g <generator_type> -s <search_type>" << endl;
        cout << "generator_type: bfs, kruskal" << endl;
        cout << "search_type: dfs, dijkastra" << endl;
    }
    else {
        strcpy(generator_type, argv[2]);
        strcpy(search_type, argv[4]);
    }
    
    vector<vector<int> > grid;


    // generate the maze
    if (strcmp(generator_type, "bfs") == 0){
        gen_grid(0, grid, rank);
    }
    else if (strcmp(generator_type, "kruskal") == 0){
        gen_grid(1, grid, rank);
    }
    else {
        cout << "Invalid generator type" << endl;
    }
    
    // print the maze
    if (rank == 0) modify_maze(grid);
    MPI_Barrier(MPI_COMM_WORLD);

    // solve the maze
    if (strcmp(search_type, "dfs") == 0){
        solve_grid(0, grid, rank, answer);
    }
    else if (strcmp(search_type, "dijkstra") == 0){
        solve_grid(1, grid, rank, answer);
    }
    else {
        cout << "Invalid search type" << endl;
    }

    if (rank == 3){
        // priniting the output as per specification
        for (int i=0; i<MAZE_DIMENSION; i++){
            for (int j=0; j<MAZE_DIMENSION; j++){
                if (answer[i][j] == 0) cout << "*";
                else if (answer[i][j] == 1) cout << " ";
                else {
                    if (i == 0 && j == MAZE_DIMENSION-1) cout << "S";
                    else if (i == MAZE_DIMENSION-1 && j == 0) cout << "E";
                    else cout << "P";
                }
            }
            cout << endl;
        }
    }
    MPI_Finalize();
    return 0;
}
