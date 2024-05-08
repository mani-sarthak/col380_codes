#include<bits/stdc++.h>
#include<mpi.h> 
#include "mazegenerator.hpp"
using namespace std;




// create the MST from the adjacency matrix
void createMST_BFS(vector<vector<int > > &adj_matrix, vector<vector<int> > &mst){
    int graph_dimension = MST_DIMENSION*MST_DIMENSION;
    vector<int> d(graph_dimension, INF);
    vector<int> parent(graph_dimension, 0);
    vector<int> present(graph_dimension, 0);
    // initialise
    int src = 0;
     d[src] = 0;
     present[src] = 1;
     for (int i=0; i<graph_dimension; i++){
          if (i != src){
                d[i] = adj_matrix[src][i];
          }
     }
     // running loop n-1 times
     for (int i=0; i<graph_dimension-1; i++){
          int min = INF;
          int min_index = -1;
          for (int j=0; j<graph_dimension; j++){
                if (!present[j] && d[j] < min){
                     min = d[j];
                     min_index = j;
                }
          }
          present[min_index] = 1;
          for (int j=0; j<graph_dimension; j++){
                if (!present[j] && adj_matrix[min_index][j] < d[j]){
                     d[j] = adj_matrix[min_index][j];
                     parent[j] = min_index;
                }
          }
     }

    // representing MST as adjacency list
    mst.resize(graph_dimension);
    for(int i =0; i < graph_dimension; i++){
        if (parent[i] == i) continue;
        mst[parent[i]].push_back(i);
        mst[i].push_back(parent[i]);
    }
}




vector<vector<int> > getMST_BFS(int rank){
   vector<vector<cell> > graph(MST_DIMENSION , vector<cell>(MST_DIMENSION));
    initialiseGraph(graph, rank);
    // print_graph(graph);

    vector<vector<int> > adj_matrix(MST_DIMENSION * MST_DIMENSION, vector<int>(MST_DIMENSION * MST_DIMENSION, INF));
    generateAdjacencyMatrix(graph, adj_matrix);
    // print_adjacency_matrix(adj_matrix);

    vector<vector<int> > mst;
    createMST_BFS(adj_matrix, mst);
    return mst;
}
