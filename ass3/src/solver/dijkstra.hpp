#ifndef DIJKSTRA
#define DIJKSTRA

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

pair<bool, vector<vector<int> > > solve_dijkstra(pair<int, int> src, pair<int, int> dest, vector<vector<int> >& maze);

#endif