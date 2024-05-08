#ifndef MAZE_SOLVER_HPP
#define MAZE_SOLVER_HPP

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

#define MAZE_DIMENSION 64
#define NPROCS 2
#define MST_DIMENSION MAZE_DIMENSION / (2 * NPROCS)
#define WEIGHT_RANGE 71
#define INF 1e7

void solve_grid(int type, vector<vector<int> > &grid, int rank, vector<vector<int> >& answer);

#endif