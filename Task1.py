import numpy as np
import pandas as pd
import plotly.express as px

# Import dataset
city_distances_df = pd.read_csv('city-inner-sweden.csv', skiprows=1, sep=' ', header=None)
training_set = np.array(city_distances_df)

names_df = pd.read_csv('city-names-sweden.csv', header=None)

# Find eigenvalues and eigenvector
evals, evecs = np.linalg.eigh(training_set)

# Store the values for the two biggest eigenvalues to use as x- and y-axis
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]
evals = evals[:2]
evecs = evecs[:, :2]

# Calculate the position of the datapoints with the same relation as the ones in dataset
L = np.diag(np.sqrt(evals))
pos_val = evecs[:, :2]
prod = np.matmul(pos_val, L)

# Save the array as a dataframe before plotting
too_plot = pd.DataFrame(data=prod)
too_plot = pd.concat([too_plot, names_df], axis=1)
too_plot.columns =['x', 'y', 'Names']

# Plot datapoints, do some thouch-up to help interpretability of placement compared to actual map
fig = px.scatter(too_plot, x="y", y="x", text="Names")
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(autorange="reversed", range=[600, -600])
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Sweden', title_x=0.5)
fig.show()

