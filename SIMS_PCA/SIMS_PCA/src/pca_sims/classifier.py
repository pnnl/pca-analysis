import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Get the mass vs. species data
data = pd.read_csv("SIMS_PCA/SIMS_PCA/src/train_data.csv")


# ----------------------------Generate testing data----------------------------
test_size = 1000
# Get how many classes we need from the original data
output_dim = data.shape[0]
print("Num output classes: ", output_dim)

# Select a bunch of random rows in the data
np.random.seed(2)
masses = []
species = []
for i in range(test_size):
    rand_index = np.random.randint(output_dim)
    masses.append(data.at[rand_index, "Precise Mass"])
    # Use one-hot encodig of species strings to integers.
    species.append(rand_index)
masses = np.array(masses)

# TODO Accuracy goes down from 100% --> 0% as this uncertainty gets too small. Possibly use:
#      1) More accurate uncertainties?
#      2) A warning / default classification when vector becomes all 0s ([0,0,0,0,...0])?
# Generate noise for the same number of rows'. Multiply by an uncertainty factor representative of real data noise (e.g., usually around 10-100 ppm, or 1e-5 to 1e-4).
unc = 1e-4
noise = unc * np.random.standard_normal(size = test_size)

# Combine the noise and randomly selected masses to create our training data
x_data = masses + noise*masses
y_data = species
# print("x_data: ", x_data)
# print("y_data: ", y_data)


# Make the known species masses into a row, then repeat for the length of the training data to prepare for mass array operations
# in the next cell
precise_masses = np.reshape(np.array(data["Precise Mass"]), (1,-1))
precise_masses = precise_masses.repeat(test_size,0)
print(precise_masses)


# TODO Accuracy goes down from 100% --> 0% as this uncertainty gets too small. Possibly use:
#      1) More accurate uncertainties?
#      2) A warning / default classification when vector becomes all 0s ([0,0,0,0,...0])?
# Store uncertainty and unknown species to be classified
unc = 0.001

# Calculate standard deviations and resulting probabilities for each mass according to its uncertainty pdf.
# We transform the x_data from 1 x n to n x 1, then broadcast it to n x 48 while subtracting precise_masses and dividing
# by the uncertainty to get the number of standard deviations each measured mass is from each of the 48 known species
# masses, then use norm.cdf to convert to probabilities.
prob_matrix = np.zeros((test_size,len(data)))
prob_matrix = (np.reshape(x_data, (-1,1)) - precise_masses) / (1000 * unc)  # TODO = 1e-1. This (overall) factor currently results in 8% accuracy @ 1e-5, 98% accuracy @ 1e-3, 97% accuracy @ 1e14, and 8% accuracy @ 1e17.
prob_matrix = (norm.cdf(-abs(prob_matrix)))

# Calculate relative probabilities by scaling row sum to 1
rel_prob_matrix = 1/np.reshape(np.sum(prob_matrix,1),(-1,1)) * prob_matrix
# Print out rows from the relative probability matrix to give us an idea of the output
print(np.round(rel_prob_matrix[0:3,:],4))


# Now let's compare our array with the real labels and determine the classification accuracy of the Gaussian Process
y_pred = np.argmax(rel_prob_matrix, 1)
species_pred = [data.at[y, 'Species'] for y in y_pred]
species_act = [data.at[y, 'Species'] for y in y_data]

# print("\nPredicted: \n", species_pred)
# print("\nActual: \n", species_act)

class_acc = (y_pred == y_data).mean()*100
print("\nClassification Accuracy: ", str(class_acc) + "%")


# Show some incorrect classifications and what they should be
y_pred, y_data, species_pred, species_act = np.array(y_pred), np.array(y_data), np.array(species_pred), np.array(species_act)
prediction_mask = (y_pred != y_data)

print("\nIncorrect Predictions: \n", species_pred[prediction_mask])
print("\nCorrect Species: \n", species_act[prediction_mask])

print("\nIndices of Incorrect Predictions: \n", [i for i in range(len(prediction_mask)) if prediction_mask[i]])