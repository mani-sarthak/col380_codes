#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};


bool is_valid(int x, int y, int rows, int cols) {
    return x >= 0 && x < rows && y >= 0 && y < cols;
}

pair<bool, vector<vector<int> > > solve_dfs(pair<int,int> start, pair<int,int> end, vector<vector<int> > &maze){
    int rows = maze.size();
    int cols = maze[0].size();
    vector<vector<int> > visited(rows, vector<int>(cols, 0));
    stack<pair<int,int> > s;
    if (maze[start.first][start.second] != 1 || maze[end.first][end.second] != 1) {
        return make_pair(false, maze);
    }

    s.push(start);
    vector<vector<int> > path = maze;

    while(!s.empty()){
        pair<int,int> curr = s.top();
        s.pop();
        if(visited[curr.first][curr.second]){
            path[curr.first][curr.second] = maze[curr.first][curr.second];
            continue;
        }
        path[curr.first][curr.second] = 2;
        visited[curr.first][curr.second] = 1;
        if(curr == end){
            return make_pair(true, path);
            break;
        }
        s.push(curr);
        for(int i=0;i<4;i++){
            int nx = curr.first + directions[i][0];
            int ny = curr.second + directions[i][1];
            if(is_valid(nx,ny,rows,cols) && !visited[nx][ny] && maze[nx][ny]){
                s.push(make_pair(nx,ny));
            }
        }
    }
    return make_pair(false, path);

}
