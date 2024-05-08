#include<bits/stdc++.h>
#include<mpi.h>
#include "bfs.hpp"
#include "kruskal.hpp"
#include "mazegenerator.hpp"
using namespace std;





int get_randint(){
    return rand() % WEIGHT_RANGE + 1;
}


// initialise the graph with random weights to get a random MST
void initialiseGraph(vector<vector<cell> >&graph, int rank){
    srand(time(0)+rank);
    for(int i=0; i<graph.size(); i++){
        for(int j=0; j<graph[0].size(); j++){
            int curr = i*MST_DIMENSION + j;
            int right = i*MST_DIMENSION + j+1;
            int down = (i+1)*MST_DIMENSION + j;
            graph[i][j].x = i;
            graph[i][j].y = j;
            graph[i][j].right = get_randint();
            if (j != graph[0].size()-1) {
                graph[i][j+1].left = get_randint();
            }
            graph[i][j].down = get_randint();
            if (i != graph.size()-1) {
                graph[i+1][j].up = get_randint();
            }
        }
    }
    // removing the edge cases
    for(int i = 0; i < graph.size(); i++) graph[i][graph[0].size()-1].right = 0;
    for(int i = 0; i < graph[0].size(); i++) graph[graph.size()-1][i].down = 0;
}


void print_graph(vector<vector<cell> > &graph){
    for (int i=0; i<graph.size(); i++){
        for (int j=0; j<graph[0].size(); j++){
            cout << "x: " << graph[i][j].x << " y: " << graph[i][j].y << " left: " << graph[i][j].left << " right: " << graph[i][j].right << " up: " << graph[i][j].up << " down: " << graph[i][j].down << endl;
        }
    }
}


void print_adjacency_matrix(vector<vector<int> > &adjacency_matrix){
    for (int i=0; i<MST_DIMENSION*MST_DIMENSION; i++){
        for (int j=0; j<MST_DIMENSION*MST_DIMENSION; j++){
            cout << adjacency_matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void print_mst(vector<vector<int> > &mst){
    for (int i=0; i<mst.size(); i++){
        cout << i << " : ";
        for (int j=0; j<mst[i].size(); j++){
            cout << mst[i][j] << " ";
        }
        cout << endl;
    }
}


vector<int> mapVertex(int idx){
    vector<int> mapp(MST_DIMENSION*MST_DIMENSION);
    int cnt = 0;
    if (idx == 0){
        for(int i=0; i<MST_DIMENSION; i++){
            for(int j=0; j<MST_DIMENSION; j++){
                mapp[cnt++] = i*MST_DIMENSION*2 + j;
            }
        }
    }
    else if (idx == 1){
        for(int i=0; i<MST_DIMENSION; i++){
            for(int j=MST_DIMENSION; j<2*MST_DIMENSION; j++){
                mapp[cnt++] = i*MST_DIMENSION*2 + j;
            }
        }
    }
    else if (idx == 2){
        for(int i=MST_DIMENSION; i<2*MST_DIMENSION; i++){
            for(int j=0; j<MST_DIMENSION; j++){
                mapp[cnt++] = i*MST_DIMENSION*2 + j;
            }
        }
    }
    else if (idx == 3){
        for(int i=MST_DIMENSION; i<2*MST_DIMENSION; i++){
            for(int j=MST_DIMENSION; j<2*MST_DIMENSION; j++){
                mapp[cnt++] = i*MST_DIMENSION*2 + j;
            }
        }
    }
    else {
        assert(false);
    }
    return mapp;
}


void gen_grid_from_adjList(vector<vector<int> >&adj, vector<vector<int> >&grid){
    int sz = 2*MST_DIMENSION;
    int graph_dimension = sz*sz;
    grid.resize(2*sz, vector<int>(2*sz, 1));
    for(int i=0; i<graph_dimension; i++){
        for(int j=0; j<adj[i].size(); j++){
           int x1 = i/sz, y1 = i%sz;
           int x2 = adj[i][j]/sz, y2 = adj[i][j]%sz;
           grid[2*x1][2*y1]=0;
           grid[2*x2][2*y2]=0;
           grid[x1+x2][y1+y2]=0;
        }
    }

    // edge cases
    grid[0][2*sz-1] = 0;
    grid[2*sz-1][0] = 0;

    // // add some non blocking cells
    for (int i = 0; i < 2 * sz; i++) {
        if (grid[MAZE_DIMENSION/2 - 1][i] == 0) {
            continue;
        }
        int non_wall_neighbours = 0;
        non_wall_neighbours += (i > 0 && grid[MAZE_DIMENSION/2 - 1][i-1] == 0);
        non_wall_neighbours += ((i < 2 * sz - 1) && grid[MAZE_DIMENSION/2 - 1][i+1] == 0);
        non_wall_neighbours += (grid[MAZE_DIMENSION/2 - 2][i] == 0);
        non_wall_neighbours += (grid[MAZE_DIMENSION/2][i] == 0);
        if (non_wall_neighbours <= 1) {
            grid[MAZE_DIMENSION/2 - 1][i] = 0;
        }
    }

    for (int i = 0; i < 2 * sz; i++) {
        if (grid[i][MAZE_DIMENSION/2 - 1] == 0) {
            continue;
        }
        int non_wall_neighbours = 0;
        non_wall_neighbours += (i > 0 && grid[i-1][MAZE_DIMENSION/2 - 1] == 0);
        non_wall_neighbours += ((i < 2 * sz - 1) && grid[i+1][MAZE_DIMENSION/2 - 1] == 0);
        non_wall_neighbours += (grid[i][MAZE_DIMENSION/2 - 2] == 0);
        non_wall_neighbours += (grid[i][MAZE_DIMENSION/2] == 0);
        if (non_wall_neighbours <= 1) {
            grid[i][MAZE_DIMENSION/2 - 1] = 0;
        }
    }


    // for (int i = 0; i<2*sz; i++){
    //     if (grid[i][2*sz-2]) {
    //         float rand = get_randint();
    //         rand /= WEIGHT_RANGE;
    //         if (rand < 0.8) grid[i][2*sz-1] = 0;
    //     }
    //     if (grid[2*sz-2][i]) {
    //         float rand = get_randint();
    //         rand /= WEIGHT_RANGE;
    //         if (rand < 0.8) grid[2*sz-1][i] = 0;
    //     }
    // }
}


void generateAdjacencyMatrix(vector<vector<cell> > &graph, vector<vector<int> > &adjacency_matrix){
    for (int i=0; i < graph.size(); i++){
        for (int j=0; j<graph[0].size(); j++){
            int curr = i*graph.size() + j;
            adjacency_matrix[curr][curr] = 0;
            if (graph[i][j].right) {
                adjacency_matrix[curr][curr+1] = graph[i][j].right;
            }
            if (graph[i][j].down) {
                adjacency_matrix[curr][curr+graph[0].size()] = graph[i][j].down;
            }
            if (graph[i][j].left) {
                adjacency_matrix[curr][curr-1] = graph[i][j].left;
            }
            if (graph[i][j].up) {
                adjacency_matrix[curr][curr-graph[0].size()] = graph[i][j].up;
            }
        }
    }
}



// simply take the parameter and depending on that generate using the respective algorithm
void gen_grid(int type, vector<vector<int> >& grid, int rank){
    const int ARR_SIZE = MST_DIMENSION * MST_DIMENSION * 2;
    vector<vector<int> > mst;
    if (type == 0){
        mst = getMST_BFS(rank);
    }
    else if (type == 1){
        mst = getMST_Kruskal(rank);
    }
    else {
        cout << "Invalid generator type" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    vector<int> mapp = mapVertex(rank);
    int* edges = new int[ARR_SIZE];
    int cnt = 0;
    for (int i=0; i<mst.size(); i++){
        for (int j=0; j<mst[i].size(); j++){
            if (i < mst[i][j]){
                edges[cnt++] = mapp[i];
                edges[cnt++] = mapp[mst[i][j]];
            }
        }
    }
    int* arr = new int[MAZE_DIMENSION*MAZE_DIMENSION];
    if (rank != 0){
        MPI_Send(edges, ARR_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
        grid.resize(MAZE_DIMENSION, vector<int>(MAZE_DIMENSION, 0));
        MPI_Bcast(arr, grid.size()*grid.size(), MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < grid.size(); i++){
            for (int j = 0; j < grid[i].size(); j++){
                grid[i][j] = arr[i * grid.size() + j];
            }
        }
    }
    else {
        int* edges1, *edges2, *edges3;
        edges1 = new int[ARR_SIZE];
        edges2 = new int[ARR_SIZE];
        edges3 = new int[ARR_SIZE];
        MPI_Recv(edges1, ARR_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(edges2, ARR_SIZE, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(edges3, ARR_SIZE, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vector<vector<int> > final_edges((2*MST_DIMENSION)*(2*MST_DIMENSION));
        for (int i=0; i<cnt; i+=2){
            final_edges[edges[i]].push_back(edges[i+1]);
            final_edges[edges[i+1]].push_back(edges[i]);
            final_edges[edges1[i]].push_back(edges1[i+1]);
            final_edges[edges1[i+1]].push_back(edges1[i]);
            final_edges[edges2[i]].push_back(edges2[i+1]);
            final_edges[edges2[i+1]].push_back(edges2[i]);
            final_edges[edges3[i]].push_back(edges3[i+1]);
            final_edges[edges3[i+1]].push_back(edges3[i]);
        }
        // print_mst(final_edges);
        
        // now connect the dissconected MSTs
        // connecting p0 and p1
        int x1 = (get_randint() % MST_DIMENSION);
        int y1 = MST_DIMENSION-1;
        x1 = 6; // right edge
        final_edges[x1*(2*MST_DIMENSION)+y1].push_back(x1*(2*MST_DIMENSION)+y1+1);
        final_edges[x1*(2*MST_DIMENSION)+y1+1].push_back(x1*(2*MST_DIMENSION)+y1);

        // connecting p0 and p2
        int x2 = MST_DIMENSION-1;
        int y2 = (get_randint() % MST_DIMENSION);
        y2 = 13; // down edge
        final_edges[x2*(2*MST_DIMENSION)+y2].push_back((x2+1)*(2*MST_DIMENSION)+y2);
        final_edges[(x2+1)*(2*MST_DIMENSION)+y2].push_back(x2*(2*MST_DIMENSION)+y2);

        // connecting p1 and p3
        int x3 = (get_randint() % MST_DIMENSION);
        int y3 = MST_DIMENSION-1;
        x3 = MST_DIMENSION;
        final_edges[x3*(2*MST_DIMENSION)+y3].push_back(x3*(2*MST_DIMENSION)+y3+1);
        final_edges[x3*(2*MST_DIMENSION)+y3+1].push_back(x3*(2*MST_DIMENSION)+y3);

        // print_mst(final_edges);
        gen_grid_from_adjList(final_edges, grid);
        
        for (int i = 0; i < grid.size(); i++){
            for (int j = 0; j < grid[i].size(); j++){
                arr[i * grid.size() + j] = 1^grid[i][j];
            }
        }
        MPI_Bcast(arr, grid.size()*grid.size(), MPI_INT, 0, MPI_COMM_WORLD);
        // print_grid(grid);
    }
}