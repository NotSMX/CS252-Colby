'''kmeans.py
Performs K-Means clustering
Daniel Yu
CS 251/2: Data Analysis and Visualization
Spring 2024
'''
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        pass

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        difference = (pt_1 - pt_2) ** 2
        sum = np.sum(difference)
        distances = np.sqrt(sum)
        total_distance = np.sum(distances)
    
        return float(total_distance)
        pass

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        self.centroids = centroids
        difference = (self.centroids - pt) ** 2
        sum = np.sum(difference, axis = 1)
        distances = np.sqrt(sum)
    
        return distances
        pass

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k
        self.centroids = np.ndarray([k, self.num_features])
        for i in range(k):
            self.centroids[i,:] = self.data[np.random.randint(self.num_samps), :]
        return self.centroids
        pass

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')
        
        self.k = k
        self.centroids = self.initialize(k)
        count = 0
        diff = float('inf')

        while count <= max_iter and np.abs(np.mean(diff)) >= tol:
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, diff = self.update_centroids(self.k, self.data_centroid_labels, self.centroids)
            count = count + 1
            self.inertia = self.compute_inertia()

        if verbose:
            print(f"Number of iterations that K-means was run for: {count}")
            
        return self.inertia, count

    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best_inertia = float('inf')
        for i in range(1, n_iter+1):
            
            # print(k)
            kmeans, count = self.cluster(k)
            # print(self.cluster(k))
            if kmeans < best_inertia:
                self.inertia = kmeans
                best_inertia = kmeans
                best_centroids = self.centroids
                best_centroid_labels = self.data_centroid_labels
            # print(f"Number of Iteration(s): {i} inertia: {best_inertia}")
            if verbose:
                  print(f"Number of Iteration(s): {i} inertia: {best_inertia}")

        self.centroids = best_centroids
        self.data_centroid_labels = best_centroid_labels
        pass

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        ndarray = np.ndarray([self.num_samps,])
        for x in range(self.data.shape[0]):
            list = []
            for y in centroids:
                list.append(self.dist_pt_to_pt(self.data[x,:], y))
            min = np.argmin(list)
            ndarray[x] = min
        return ndarray
        pass

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = np.ndarray([k, self.num_features])
        centroid_diff = np.ndarray([k, self.num_features])

        for x in range(k):
        # the indices that are assigned to the cluster
            indices = np.where(data_centroid_labels == x)[0]
            if len(indices) > 0:
                new_centroids[x] = np.mean(self.data[indices], axis=0)
            else:
                new_centroids[x] = self.data[np.random.randint(self.num_samps)]
        centroid_diff = new_centroids - prev_centroids

        return new_centroids, centroid_diff

        pass

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        mse = 0.0
        for i in range(len(self.data)):
            mse += np.square(self.dist_pt_to_pt(self.data[i],  self.centroids[int(self.data_centroid_labels[i])]))
        self.inertia = (1/len(self.data)) * mse


        return float(self.inertia)
        pass

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

        cluster_colors = [colors[int(label)] for label in self.data_centroid_labels.astype(int)]
        plt.scatter(self.data[:, 0], self.data[:,1], c=cluster_colors)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x', s=100)
        plt.title("Clusters Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        pass

    def elbow_plot(self, max_k, n_iter = 1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: int. Number of iterations that are run for

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertias = []

        for x in range(1, max_k+1):
            self.cluster_batch(x, n_iter)
            inertias.append(self.inertia)
        x = np.arange(1, max_k+1)
        plt.plot(x, inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title("Elbow Plot")
        plt.xticks(x)

        pass

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        pass
        for i in range(self.data.shape[0]):
                distances = self.dist_pt_to_centroids(self.data[i], self.centroids)
                index = np.argmin(distances)
                self.data[i] = self.centroids[index]

    '''EXTENSIONS BEYOND HERE'''

    def clusteriter(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')
        
        count = 0
        diff = float('inf')

        while count <= max_iter and np.abs(np.mean(diff)) >= tol:
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, diff = self.update_centroids(self.k, self.data_centroid_labels, self.centroids)
            count = count + 1
            self.inertia = self.compute_inertia()
            yield(self.centroids, self.data_centroid_labels, count)
    
    def animate(self, i, fig, ax) :
        colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        ax.cla()
        cluster_colors = [colors[int(label)] for label in self.data_centroid_labels.astype(int)]
        ax.scatter(self.data[:, 0], self.data[:,1], c=cluster_colors)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x', s=100)
        ax.text(0.05,0.95,'Frame: '+ str(i[2]), horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes)
        return [fig]


    def animate_clusters(self, k=2, max_iter = 1000):
        self.k = k
        self.initialize(k)
        fig = plt.figure(figsize=(8,5))
        ax = plt.axes()
        ani = animation.FuncAnimation(fig, self.animate, fargs=(fig, ax), frames = self.clusteriter(k=k, max_iter=max_iter), interval=200, blit=True, cache_frame_data=False)
        ani.save('kmeans.gif',writer='Pillow')
            
