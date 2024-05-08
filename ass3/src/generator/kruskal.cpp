#include <bits/stdc++.h>
#include <mpi.h>
#include "mazegenerator.hpp"
using namespace std;



struct Edge {
    int src, dest, weight;
};

// Disjoint Set (Union-Find) data structure
class DisjointSet {
private:
    vector<int> parent, rank;

public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unionSets(int x, int y) {
        int xRoot = find(x);
        int yRoot = find(y);

        if (xRoot == yRoot)
            return;

        if (rank[xRoot] < rank[yRoot])
            parent[xRoot] = yRoot;
        else if (rank[xRoot] > rank[yRoot])
            parent[yRoot] = xRoot;
        else {
            parent[yRoot] = xRoot;
            rank[xRoot]++;
        }
    }
};

bool compareEdges(const Edge& a, const Edge& b) {
    return a.weight < b.weight;
}

// Function to find MST using Kruskal's algorithm
vector<vector<int> > kruskalMST(vector<Edge>& edges, int n) {
    sort(edges.begin(), edges.end(), compareEdges);

    vector<vector<int> > mst(n);

    DisjointSet dsu(n);

    for (int i= 0; i<edges.size(); i++){
        Edge edge = edges[i];
        int src = edge.src;
        int dest = edge.dest;

        int srcRoot = dsu.find(src);
        int destRoot = dsu.find(dest);

        if (srcRoot != destRoot) {
            mst[src].push_back(dest);
            mst[dest].push_back(src);
            dsu.unionSets(srcRoot, destRoot);
        }
    }

    return mst;
}

// given a graph return the edges in the graph
vector<Edge> generateEdges(vector<vector<cell> > &graph){
    vector<Edge> edges;
    for (int i=0; i<MST_DIMENSION; i++){
        for (int j=0; j<MST_DIMENSION; j++){
            if (graph[i][j].right){
                Edge edge;
                edge.src = i*MST_DIMENSION + j;
                edge.dest = i*MST_DIMENSION + j + 1;
                edge.weight = graph[i][j].right;
                edges.push_back(edge);
            }
            if (graph[i][j].down){
                Edge edge;
                edge.src = i*MST_DIMENSION + j;
                edge.dest = (i+1)*MST_DIMENSION + j;
                edge.weight = graph[i][j].down;
                edges.push_back(edge);
            }
        }
    }
    return edges;
}

vector<vector<int> > getMST_Kruskal(int rank){
    vector<vector<cell> > graph(MST_DIMENSION , vector<cell>(MST_DIMENSION));
    initialiseGraph(graph, rank);
    vector<Edge> edges;
    edges = generateEdges(graph);
    vector<vector<int> > mst =  kruskalMST(edges, MST_DIMENSION*MST_DIMENSION);
    return mst;
}