import numpy as np

def get_top_two_indices(arr):
    # Get the indices of the array sorted in ascending order
    sorted_indices = np.argsort(arr)

    # Get the index of the largest number (last index after sorting)
    largest_index = sorted_indices[-1]

    # Get the index of the second largest number (second-to-last index after sorting)
    second_largest_index = sorted_indices[-2]

    return largest_index, second_largest_index

# Example usage:
my_array = np.array([5, 10, 3, 8, 7])
largest_index, second_largest_index = get_top_two_indices(my_array)

print("Index of the largest number:", largest_index)
print("Index of the second largest number:", second_largest_index)
