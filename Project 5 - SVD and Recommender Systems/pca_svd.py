'''pca_cov.py
Performs principal component analysis using the singular value decomposition of the dataset
Daniel Yu
CS 252: Mathematical Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import pandas as pd
import pca

from data_transformations import normalize


class PCA_SVD(pca.PCA):
    '''Perform principal component analysis using the singular value decomposition (SVD)

    NOTE: In your implementations, only the following "high level" `scipy`/`numpy` functions can be used:
    - `np.linalg.svd`
    The numpy functions that you have been using so far are fine to use.
    '''
    
    def fit(self, vars, normalize_dataset=False):
        '''Performs PCA on the data variables `vars` using SVD instead of the covariance matrix

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE:
        - This method should closely mirror the structure of your implementation in the `PCA` class, except there
        should NOT be a covariance matrix computed here!
        - Make sure you compute all the same instance variables that `fit` does in `PCA`.
        - Leverage the methods that you already implemented as much as possible to do computations.
        '''
        pass
        
        self.vars = vars
        self.A = self.data[vars].to_numpy()

        if normalize_dataset:
            self.orig_means = np.mean(self.A, axis=0)
            self.orig_mins = np.min(self.A, axis=0)
            self.orig_maxs = np.max(self.A, axis=0)
            self.A = (self.A - self.orig_means) / (self.orig_maxs - self.orig_mins)
            self.normalized = True
        else:
            self.normalized = False

        U, S, Vt = np.linalg.svd(self.A, full_matrices=False)
        
        self.e_vals = S ** 2 / (self.A.shape[0] - 1)
        self.e_vecs = Vt.T
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

