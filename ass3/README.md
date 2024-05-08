**Readme**

To compile the program, use one of the following commands:

```bash
make maze
```
or simply:
```bash
make
```

This will generate an executable named `maze.out` in the parent directory.

To use the executable for solving mazes, employ the following command:

```bash
mpirun -np 4 ./maze.out -g [bfs/kruskal] -s [dfs/dijkstra]
```

Replace `[bfs/kruskal]` with your preferred algorithm for generating the maze (either breadth-first search or Kruskal's algorithm), and `[dfs/dijkstra]` with your choice of solving algorithm (depth-first search or Dijkstra's algorithm).

For more details on design decisions and implementation, please refer to the attached report.
