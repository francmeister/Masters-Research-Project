# # import numpy as np

# # # Example integer numpy array
# # int_array = np.array([0, 2, 1, 2])

# # # Determine the number of classes (based on max value in int_array + 1)
# # num_classes = 3#np.max(int_array)

# # # Generate one-hot encoding using np.eye and indexing
# # one_hot_encoded = np.eye(num_classes)[int_array]
# # #one_hot_encoded = one_hot_encoded.reshape(1,len(one_hot_encoded)*len(one_hot_encoded[0]))
# # print(one_hot_encoded)

# import numpy as np

# # Example numpy array
# arr = np.array([1, 3, 7, 2, 5])

# # Find the index of the largest number
# index_of_max = np.argmax(arr)

# print(index_of_max)
import numpy as np

# Example arrays
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

# Element-wise equality
are_equal = np.array_equal(a, b)  # Returns True if arrays are identical
print(are_equal)


