'''data_transformations.py
Daniel Yu
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Spring 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    # Compute the minimum and maximum values for each variable
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Compute the range for each variable
    ranges = max_vals - min_vals

    # Subtract the minimum value and divide by the range for each variable
    normalized_data = (data - min_vals) / ranges

    return normalized_data


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    # Calculate the mean of each feature (column)
    means = np.mean(data, axis=0)
    
    # Subtract the mean from each data point along each feature
    centered_data = data - means
    
    return centered_data


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    # Convert angle from degrees to radians
    radians = np.radians(degrees)
    
    # Construct the rotation matrix based on the specified axis
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(radians), -np.sin(radians)],
                         [0, np.sin(radians), np.cos(radians)]])
    elif axis == 'y':
        return np.array([[np.cos(radians), 0, np.sin(radians)],
                         [0, 1, 0],
                         [-np.sin(radians), 0, np.cos(radians)]])
    elif axis == 'z':
        return np.array([[np.cos(radians), -np.sin(radians), 0],
                         [np.sin(radians), np.cos(radians), 0],
                         [0, 0, 1]])

