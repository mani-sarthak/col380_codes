#include <bits/stdc++.h>
#include <mpi.h>
#include "mazesolver.hpp"
using namespace std;



const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};
pair<bool, vector<vector<int> > > solve_dijkstra(pair<int, int> src, pair<int, int> dest, vector<vector<int> >& mat) {

    int n = mat.size();
    if (mat[src.first][src.second] != 1 || mat[dest.first][dest.second] != 1) {
        return make_pair(false, mat);
    }
    // initialize distances
    vector<vector<int> > dist(n, vector<int>(n, INF));
    dist[src.first][src.second] = 0;

    queue<pair<int, int> > q;
    q.push(make_pair(src.first, src.second));
    bool found = false;

    // dijkstra (equivalent to bfs here)
    while (!q.empty()) {
        pair<int, int> u = q.front();
        q.pop();

        if (u == dest) {
            found = true;
            break;
        }

        for (int i = 0; i < 4; i++) {
            int x = u.first + dx[i];
            int y = u.second + dy[i];

            if (x >= 0 && x < n && y >= 0 && y < n && mat[x][y] == 1) {
                int new_dist = dist[u.first][u.second] + 1;
                if (new_dist < dist[x][y]) {
                    dist[x][y] = new_dist;
                    q.push(make_pair(x, y));
                }
            }
        }

    }

    if (!found) {
        return make_pair(false, mat);
    }

    // copy original matrix
    vector<vector<int> > result = mat;

    // go from destination to source and mark points 2 on the path
    pair<int, int> curr = dest;
    while (curr != src) {
        result[curr.first][curr.second] = 2;

        for (int i = 0; i < 4; i++) {
            int x = curr.first + dx[i];
            int y = curr.second + dy[i];

            if (x >= 0 && x < n && y >= 0 && y < n &&
                dist[x][y] + 1 == dist[curr.first][curr.second]) {
                curr = make_pair(x, y);
                break;
            }
        }
    }
    result[src.first][src.second] = 2;

    return make_pair(true, result);

}
