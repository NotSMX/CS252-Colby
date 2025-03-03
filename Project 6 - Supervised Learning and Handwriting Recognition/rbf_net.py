'''rbf_net.py
Radial Basis Function Neural Network
Daniel Yu
CS 252: Mathematical Data Analysis and Visualization
Spring 2024
'''
import numpy as np

import kmeans
from classifier import Classifier

class RBF_Net(Classifier):
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Call the superclass constructor
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        super().__init__(num_classes=num_classes)
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None
        self.k = num_hidden_units

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_wts(self):
        '''Returns the hidden-output layer weights and bias

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(num_hidden_units+1, num_classes).
        '''
        return self.wts


    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        self.k = centroids.shape[0]
        cluster_dists = np.zeros(self.k)
        for i in range(self.k):
            indices = np.where(cluster_assignments == i)[0]
            if indices.size == 0:
                cluster_dists[i] = 0
            else:
                centroid = centroids[i]
                distances = kmeans_obj.dist_pt_to_centroids(data[indices], centroid)
                cluster_dists[i] = np.mean(distances)

        return cluster_dists
        pass

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kmeans_obj = kmeans.KMeans(data)
        kmeans_obj.cluster_batch(self.k, n_iter = 5)
        self.prototypes = kmeans_obj.centroids
        self.sigmas = self.avg_cluster_dist(data, kmeans_obj.centroids, kmeans_obj.data_centroid_labels, kmeans_obj)

    def pseudo_inverse(self, A):
        '''Uses the SVD to compute the pseudo-inverse of the data matrix `A`
        
        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        
        Returns
        -----------
        ndarray. shape=(num_features, num_data_samps). The pseudoinverse of `A`.

        NOTE:
        - You CANNOT use np.linalg.pinv here!! Implement it yourself with SVD :)
        - Skip this until we cover the topic in lecture
        '''
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        s_inv = np.diag(1 / s)
        return np.dot(vh.T, np.dot(s_inv, u.T))


    def linear_regression(self, A, y):
        '''Performs linear regression using the SVD-based solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE:
        - Remember to handle the intercept
        - You should use your own SVD-based solver here, but if you get here before we cover this in lecture, use
        scipy.linalg.lstsq for now.
        '''
        A_bias = np.hstack((np.ones((A.shape[0], 1)), A))
        return np.dot(self.pseudo_inverse(A_bias), y)

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        k_mean = kmeans.KMeans(data)
        epsilon = 1e-8
        activations = np.zeros((data.shape[0], self.k))
        for i in range(data.shape[0]):
            distances = k_mean.dist_pt_to_centroids(data[i, :], self.prototypes)
            ha = np.exp(-(distances ** 2) / ((2*(self.sigmas) ** 2) + epsilon))
            activations[i,:] = ha.reshape(1,-1)
        
        return activations

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        num_samples = hidden_acts.shape[0]
        output_acts = np.zeros((num_samples, self.num_classes))
        hidden_acts_bias = np.hstack((np.ones((num_samples, 1)), hidden_acts))
        for i in range(num_samples):
            output_acts[i] = np.dot(hidden_acts_bias[i], self.wts)
        return output_acts

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data)
        hidden_acts = self.hidden_act(data)
        self.wts = np.zeros((self.k + 1, self.num_classes))
        for c in range(self.num_classes):
            y_c = (y == c).astype(int)
            self.wts[:, c] = self.linear_regression(hidden_acts, y_c)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        hidden_acts = self.hidden_act(data)
        output_acts = self.output_act(hidden_acts)
        return np.argmax(output_acts, axis=1)

