import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt

# Import dataset
faces_df = pd.read_csv('frey-faces.csv', skiprows=4, sep=' ', header=None)

cols = ['0']
paramCols = [i for i in range(1, len(faces_df.columns))]
cols.extend(paramCols)

faces_df.columns = cols

cluster_array = np.array(faces_df)

# Initial constants and variables
k = 10  # Set k for choice of number of clusters

i = 0
first_run = True
cluster_vars = [99999]

# Functions
# Assign datapoint to closest centroid
def create_clusters(centroids, cluster_array):
    clusters = []
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append((sum((centroid - cluster_array[i])**2))**0.5)
        clusters.append(np.argmin(distances))
    return clusters

# Fit centroid to new cluster mean
def fit_centroids(clusters, cluster_array):
    new_centroids = []
    # Create a new df where datapoints are designated to clusters
    cluster_df = pd.concat([pd.DataFrame(cluster_array), pd.DataFrame(clusters, columns=['cluster'])], axis=1)

    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster'] == c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

# Calculate variance within each cluster
def centroid_variance(clusters, cluster_array):
    sum_squares = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),pd.DataFrame(clusters, columns=['cluster'])], axis=1)

    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster'] == c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        mean_repmat = np.matlib.repmat(cluster_mean, current_cluster.shape[0],1)
        sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat)**2)))
    return sum_squares


# Main program
# Initiate centroids in mean of dataset and include a small vector offset for each additional centroid
mean_centroids = cluster_array.mean(axis=0)
mean_centroids =[mean_centroids]
centroids = np.array(mean_centroids)

if k>1:
    for num in range(k):
        centroids = np.append(centroids, np.array(mean_centroids)+num, axis=0)

# Create the first clusters based on initial centroids
clusters = create_clusters(centroids, cluster_array)
initial_clusters = clusters

# Fit clusters to dataset until there is no more change in variance (convergence)
while first_run or (round(cluster_vars[i-1]-cluster_vars[i], 5) != 0):
    i += 1
    centroids = fit_centroids(clusters, cluster_array)
    clusters = create_clusters(centroids, cluster_array)

    cluster_var = np.mean(centroid_variance(clusters, cluster_array))
    cluster_vars.append(cluster_var)
    first_run = False


# Create an array of final centroids and a dataframe of datapoints
centroids = np.array(centroids)
cluster_choice_df = pd.DataFrame(clusters)

# Reshape data points to a 28x20 grayscale picture and show centroids together with the five best and worst matches.
for r in range(k):
    centroid = centroids[1, :]
    idx = cluster_choice_df.index[cluster_choice_df[0] == r].to_list()
    cluster_choice_array = faces_df.loc[idx, :].to_numpy()
    closest_face_index = []
    border_face_index = []
    dist = []

    for i in range(len(cluster_choice_array)):
        dist.append(np.sum((cluster_choice_array[i, :] - centroid) ** 2))

    dist = np.argsort(dist)
    closest_face_index = np.array(dist[:5])
    border_face_index = np.array(dist[-5:])

    closest_face_array = np.array(cluster_choice_array[closest_face_index])
    border_face_array = np.array(cluster_choice_array[border_face_index])

    axes = []
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(28, 20))


    b = np.reshape(centroids[r, :], (28, 20))
    axs[0, 2].imshow(b, cmap='gray')
    best = {}
    worst = {}

    for a in range(5):
        best[a] = np.reshape(closest_face_array[a, :], (28, 20))
        worst[a] = np.reshape(border_face_array[a, :], (28, 20))

    for i in range(5):
        axs[1, i].imshow(best[i], cmap='gray')
        axs[-1, i].set_title("Border " + str(i))
        axs[2, i].imshow(worst[i], cmap='gray')
        axs[-2, i].set_title("Closest  " + str(i))

    plt.show()

