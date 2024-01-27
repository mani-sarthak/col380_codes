#include <iostream>
#include <vector>
#include <random>

int main() {
    int n;
    std::cin >> n;
    std::cout << n << "\n";

    // Create a 2D vector (matrix) of size n x n
    std::vector<std::vector<double> > matrix(n, std::vector<double>(n));

    // Fill the matrix with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0); // Range can be adjusted

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    // Display the matrix
    for (const auto &row : matrix) {
        for (double elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
