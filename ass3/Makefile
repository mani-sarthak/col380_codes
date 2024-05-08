GENERATORS = src/generator/bfs.cpp src/generator/kruskal.cpp src/generator/mazegenerator.cpp
SOLVERS = src/solver/dijkstra.cpp src/solver/dfs.cpp src/solver/mazesolver.cpp
MAZE = src/maze.cpp

maze:
	mpic++ -o maze.out $(MAZE) $(GENERATORS) $(SOLVERS)

clean:
	rm -f maze.out
