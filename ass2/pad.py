import numpy as np
from scipy.signal import convolve2d

def apply_convolution(input_matrix, kernel):
    # Calculate padding size
    padding_size = (np.array(kernel.shape) - 1) // 2

    # Apply padding to the input matrix
    padded_input = np.pad(input_matrix, ((padding_size[0],), (padding_size[1],)), mode='constant')

    # Apply convolution
    output_matrix = convolve2d(padded_input, kernel, mode='valid')

    return output_matrix

# Example usage
input_matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

# Example 3x3 convolutional kernel
conv_kernel = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

output_matrix = apply_convolution(input_matrix, conv_kernel)

print("Input Matrix:")
print(input_matrix)
print("\nConvolution Kernel:")
print(conv_kernel)
print("\nOutput Matrix (with same size):")
print(output_matrix)
