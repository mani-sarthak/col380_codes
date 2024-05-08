#ifndef MAZE_GENERATOR_HPP
#define MAZE_GENERATOR_HPP

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;


#define MAZE_DIMENSION 64
#define NPROCS 2
#define MST_DIMENSION MAZE_DIMENSION / (2 * NPROCS)
#define WEIGHT_RANGE 71
#define INF 1e7


struct cell{
    int x, y;
    int left, right, up, down;
};


void gen_grid(int type, vector<vector<int> > &grid, int rank);
int get_randint();
void initialiseGraph(vector<vector<cell> >&graph, int rank);
void print_graph(vector<vector<cell> > &graph);
void print_adjacency_matrix(vector<vector<int> > &adjacency_matrix);
void print_mst(vector<vector<int> > &mst);
vector<int> mapVertex(int idx);
void gen_grid(vector<vector<int> >&adj, vector<vector<int> >&grid);
void generateAdjacencyMatrix(vector<vector<cell> > &graph, vector<vector<int> > &adjacency_matrix);


#endif
