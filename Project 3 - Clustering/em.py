'''em.py
Cluster data using the Expectation-Maximization (EM) algorithm with Gaussians
Daniel Yu
CS 252: Mathematical Data Analysis Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from IPython.display import display, clear_output


class EM():
    def __init__(self, data=None):
        '''EM object constructor.
        See docstrings of individual methods for what these variables mean / their shapes

        (Should not require any changes)
        '''
        self.k = None
        self.centroids = None
        self.cov_mats = None
        self.responsibilities = None
        self.pi = None
        self.data_centroid_labels = None

        self.loglikelihood_hist = None

        self.data = data
        self.num_samps = None
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def gaussian(self, pts, mean, sigma):
        '''
        Evaluates a multivariate Gaussian distribution described by
        mean `mean` and covariance matrix `sigma` at the (x, y) points `pts`

        Parameters:
        -----------
        pts: ndarray. shape=(num_samps, num_features).
            Data samples at which we want to evaluate the Gaussian
            Example for 2D: shape=(num_samps, 2)
        mean: ndarray. shape=(num_features,)
            Mean of Gaussian (i.e. mean of one cluster). Same dimensionality as data
            Example for 2D: shape=(2,) for (x, y)
        sigma: ndarray. shape=(num_features, num_features)
            Covariance matrix of a Gaussian (i.e. covariance of one cluster).
            Example for 2D: shape=(2,2). For standard deviations (sigma_x, sigma_y) and constant c,
                Covariance matrix: [[sigma_x**2, c*sigma_x*sigma_y],
                                    [c*sigma_x*sigma_y, sigma_y**2]]

        Returns:
        -----------
        ndarray. shape=(num_samps,)
            Multivariate gaussian evaluated at the data samples `pts`
        '''
        result = []
        # samps = pts.shape[0]
        features = pts.shape[1]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)
        norm_const = 1.0 / ((2 * np.pi) ** (features/2) * np.sqrt(np.abs(det_sigma)))
        for pt in pts:
            diff = pt - mean
            exponent = -0.5 * np.dot(diff, np.dot(inv_sigma, diff))
            result.append(norm_const * np.exp(exponent))

        return np.array(result)

    def initalize(self, k):
        '''Initialize all variables used in the EM algorithm.

        Parameters:
        -----------
        k: int. Number of clusters.

        Returns
        -----------
        None

        TODO:
        - Set k as an instance variable.
        - Initialize the log likelihood history to an empty Python list.
        - Initialize the centroids to random data samples
            shape=(k, num_features)
        - Initialize the covariance matrices to the identity matrix
        (1s along main diagonal, 0s elsewhere)
            shape=(k, num_features, num_features)
        - Initialize the responsibilities to an ndarray of 1/k.
            shape=(k, num_samps)
        - Initialize the pi array (proportion of points assigned to each cluster) so that each cluster
        is equally likely.
            shape=(k,)
        '''
        pass
        self.k = k
        self.loglikelihood_hist = []
        self.num_samps = self.data.shape[0]
        self.num_features = self.data.shape[1]

        # Initialize the centroids to random data samples
        self.centroids = np.ndarray([k, self.num_features])
        for i in range(k):
            self.centroids[i,:] = self.data[np.random.randint(self.num_samps), :]

        # Initialize the covariance matrices to the identity matrix
        self.cov_mats = np.array([np.eye(self.num_features) for _ in range(k)])

        # Initialize responsibilities array
        self.responsibilities = np.full((self.k,self.num_samps), 1/k)

        # Initialize the pi array so that each cluster is equally likely
        self.pi = np.full(k, 1/k)

    def e_step(self):
        '''Expectation (E) step in the EM algorithm.
        Set self.responsibilities, the probability that each data point belongs to each of the k clusters.
        i.e. leverages the Gaussian distribution.

        NOTE: Make sure that you normalize so that the probability that each data sample belongs
        to any cluster equals 1.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.responsibilities: ndarray. shape=(k, num_samps)
            The probability that each data point belongs to each of the k clusters.
        '''
        pass
        # Update each cluster
        for i in range(self.k):
            # Compute the Gaussian for this cluster
            # print(self.pi[i] * self.gaussian(self.data, self.centroids[i], self.cov_mats[i]))
            self.responsibilities[i, :] = self.pi[i] * self.gaussian(self.data, self.centroids[i], self.cov_mats[i])

        # Normalize so that the probability that each data sample belongs to any cluster equals 1
        # print(self.responsibilities, 'divided by', np.sum(self.responsibilities, axis=0), '=', self.responsibilities / np.sum(self.responsibilities, axis=0))
        # print( np.sum(self.responsibilities, axis=0))
        self.responsibilities = self.responsibilities / np.sum(self.responsibilities, axis=0)
       
        # print(self.responsibilities)
        return self.responsibilities

    def m_step(self):
        '''Maximization (M) step in the EM algorithm.
        Set self.centroids, self.cov_mats, and self.pi, the parameters that define each Gaussian
        cluster center and spread, as well as the degree to which data points "belong" to each cluster

        TODO:
        - Compute the proportion of data points that belong to each cluster.
        - Compute the mean of each cluster. This is the mean over all data points, but weighting
        the data by the probability that they belong to that cluster.
        - Compute the covariance matrix of each cluster. Use the usual equation (for all the data),
        but before summing across data samples, make sure to weight each data samples by the
        probability that they belong to that cluster.

        NOTE: When computing the covariance matrix, use the updated cluster centroids for
        the CURRENT time step.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.centroids: ndarray. shape=(k, num_features)
            Mean of each of the k Gaussian clusters
        self.cov_mats: ndarray. shape=(k, num_features, num_features)
            Covariance matrix of each of the k Gaussian clusters
            Example of a covariance matrix for a single cluster (2D data): [[1, 0.2], [0.2, 1]]
        self.pi: ndarray. shape=(k,)
            Proportion of data points belonging to each cluster.
        '''
        pass
        # Update each cluster
        for i in range(self.k):
            # Compute the proportion of data points that belong to this cluster
            self.pi[i] = np.sum(self.responsibilities[i, :,]) / self.num_samps

            # Compute the mean of this cluster
            self.centroids[i] = np.sum(self.responsibilities[i, :, np.newaxis] * self.data, axis=0) / np.sum(self.responsibilities[i, :,])

            # Compute the covariance matrix of this cluster
            diff = self.data - self.centroids[i]
            self.cov_mats[i] = np.dot((self.responsibilities[i, :, np.newaxis] * diff).T, diff) / np.sum(self.responsibilities[i, :,])

        return self.centroids, self.cov_mats, self.pi

    def log_likelihood(self):
        '''Compute the sum of the log of the Gaussian probability of each data sample in each cluster
        Used to determine whether the EM algorithm is converging.

        Parameters:
        -----------
        None

        Returns
        -----------
        float. Summed log-likelihood of all data samples

        NOTE: Remember to weight each cluster's Gaussian probabilities by the proportion of data
        samples that belong to each cluster (pi).
        '''
        pass
        log_likelihood = 0.0
        for i in range(self.num_samps):
            summation = 0
            for c in range(self.k):
                prob = self.pi[c] * self.gaussian(self.data[i].reshape(1,2), self.centroids[c], self.cov_mats[c])
                # print(self.data[i].reshape(1,2),self.centroids[c],self.cov_mats[c])
                # print(self.gaussian(self.data[i].reshape(1,2), self.centroids[c], self.cov_mats[c]))  
                summation += prob  
            log_likelihood += (np.log(float(summation)))
        return log_likelihood

    def cluster(self, k, max_iter=100, stop_tol=1e-3, verbose=False, animate=False):
        '''Main method used to cluster data using the EM algorithm
        Perform E and M steps until the change in the loglikelihood from last step to the current
        step <= `stop_tol` OR we reach the maximum number of allowed iterations (`max_iter`).

        Parameters:
        -----------
        k: int. Number of clusters.
        max_iter: int. Max number of iterations to allow the EM algorithm to run.
        stop_tol: float. Stop running the EM algorithm if the change of the loglikelihood from the
        previous to current step <= `stop_tol`.
        verbose: boolean. If true, print out the current iteration, current log likelihood,
            and any other helpful information useful for debugging.

        Returns:
        -----------
        self.loglikelihood_hist: Python list. The log likelihood at each iteration of the EM algorithm.

        NOTE: Reminder to initialize all the variables before running the EM algorithm main loop.
            (Use the method that you wrote to do this)
        NOTE: At the end, print out the total number of iterations that the EM algorithm was run for.
        NOTE: The log likelihood is a NEGATIVE float, and should increase (approach 0) if things are
            working well.
        '''
        pass
        # Initialize variables
        self.initalize(k)

        # Initialize log likelihood history
        self.loglikelihood_hist = []

        # Start the EM algorithm main loop
        for iteration in range(max_iter):
            # E-step: Update responsibilities
            self.e_step()

            # M-step: Update parameters
            self.m_step()

            # Compute log likelihood
            log_likelihood = self.log_likelihood()

            # Append current log likelihood to history
            self.loglikelihood_hist.append(log_likelihood)
            
            # Check convergence
            if iteration > 0 and abs(log_likelihood - self.loglikelihood_hist[-2]) <= stop_tol:
                break

            # Print verbose information if required
            if verbose:
                print(f"Iteration {iteration + 1}: Log Likelihood = {log_likelihood}")

            # Visualize clustering process if required
            if animate:
                clear_output(wait=True)
                self.plot_clusters(self.data)
                plt.pause(0.01)

        # Print total number of iterations
        print(f"Total number of iterations: {iteration + 1}")

        return self.loglikelihood_hist

    def find_outliers(self, thres=0.05):
        '''Find outliers in a dataset using clustering by EM algorithm

        Parameters:
        -----------
        thres: float. Value >= 0
            Outlier defined as data samples assigned to a cluster with probability of belonging to
            that cluster < thres

        Returns:
        -----------
        Python lists of ndarrays. len(Python list) = len(cluster_inds).
            Example if k = 2: [(array([ 0, 17]),), (array([20, 26]),)]
                The Python list has 2 entries. Each entry is a ndarray.
            Within each ndarray, indices of `self.data` of detected outliers according to that cluster.
                For above example: data samples with indices 20 and 26 are outliers according to
                cluster 2.
        '''
        pass
        # Initialize outliers list
        outliers = []
        hardcluster = np.argmax(self.responsibilities, axis=0)
        # Loop through clusters
        orig_inds = np.arange(len(self.data))
        for c in range(self.k):
            curr_cluster_data = self.data[hardcluster == c]
            cluster_ind_map = orig_inds[hardcluster == c]
            cluster_outliers = []
            cluster_probs = self.gaussian(curr_cluster_data, self.centroids[c], self.cov_mats[c])
            is_outlier = cluster_probs < thres
            # Find index of original sample that corresponds to these outliers
            # Append outliers to outliers list
            outliers.append(np.array(cluster_ind_map[is_outlier]))

        return outliers

    def estimate_log_probs(self, xy_points):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = np.log(self.gaussian(xy_points, self.centroids[c], self.cov_mats[c]))
        probs += np.log(self.pi[:, np.newaxis])
        return -logsumexp(probs, axis=0)

    def get_sample_points(self, data, res):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        data_min = np.min(data, axis=0) - 0.5
        data_max = np.max(data, axis=0) + 0.5
        x_samps, y_samps = np.meshgrid(np.linspace(data_min[0], data_max[0], res),
                                       np.linspace(data_min[1], data_max[1], res))
        plt_samps_xy = np.c_[x_samps.ravel(), y_samps.ravel()]
        return plt_samps_xy, x_samps, y_samps

    def plot_clusters(self, data, res=100, show=True):
        '''Method to call to plot the clustering of `data` using the EM algorithm

        (Should not require any changes)
        '''
        # Plot points assigned to each cluster in a different color
        cluster_hard_assignment = np.argmax(self.responsibilities, axis=0)
        for c in range(self.k):
            curr_clust = data[cluster_hard_assignment == c]
            plt.plot(curr_clust[:, 0], curr_clust[:, 1], '.', markersize=7)

        # Plot centroids of each cluster
        plt.plot(self.centroids[:, 0], self.centroids[:, 1], '+k', markersize=12)

        # Get grid of (x,y) points to sample the Gaussian clusters
        xy_points, x_samps, y_samps = self.get_sample_points(data, res=res)

        # Evaluate the sample points at each cluster Gaussian. For visualization, take max prob
        # value of the clusters at each point
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = self.gaussian(xy_points, self.centroids[c], self.cov_mats[c])
        probs /= probs.max(axis=1, keepdims=True)
        probs = probs.sum(axis=0)
        probs = np.reshape(probs, [res, res])

        # Make heatmap for cluster probabilities
        plt.contourf(x_samps, y_samps, probs, cmap='viridis')
        if show:
            plt.show()
