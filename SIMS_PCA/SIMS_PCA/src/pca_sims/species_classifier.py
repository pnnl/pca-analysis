"""This class allows us to assign a list of chemical species as a string given an input list of atomic weights."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


class species_classifier:

    # Constructor; takes the following parameters:
    #       data - The mass vs. species data (either positive or negative ions)
    def __init__(self, data_ref, data_test):
        data_ref = pd.read_csv(data_ref)

        # Get how many masses need to be tested
        test_size = data_test.shape[0]
        # Get how many classes we need from the original data
        output_dim = data_ref.shape[0]
        print("Num output classes: ", output_dim)

        # Select a bunch of random rows in the data
        np.random.seed(2)
        masses_test = data_test['raw_mass']

        self.test_size = test_size
        self.masses_ref = data_ref['Precise Mass']
        self.species_ref = data_ref['Species']
        self.masses_test = data_test['raw_mass']

    # Read the data, form representative Gaussian distributions around each peak, 
    # Returns: a matrix with each row representing the sample number and the columns in each row containing the probabilities. 
    #          All entries in a row add to a total probability of 1.
    def calculate_probabilities(self):
        masses_ref = self.masses_ref
        masses_test = self.masses_test
        test_size = self.test_size
        
        # Make the known species masses into a row, then repeat for the length of the training data to prepare for mass array operations
        # in the next cell
        precise_masses = np.reshape(np.array(masses_ref), (1,-1))
        precise_masses = precise_masses.repeat(test_size,0)
        print("\nPrecise_masses: \n", precise_masses)
        print("\nmasses_test: \n", masses_test)


        # TODO Accuracy goes down from 100% --> 0% as this uncertainty gets too small. Possibly use:
        #      1) More accurate uncertainties?
        #      2) A warning / default classification when vector becomes all 0s ([0,0,0,0,...0])?
        # Store uncertainty and unknown species to be classified
        unc = 0.001

        # Calculate standard deviations and resulting probabilities for each mass according to its uncertainty pdf.
        # We transform the x_data from 1 x n to n x 1, then broadcast it to n x 48 while subtracting precise_masses and dividing
        # by the uncertainty to get the number of standard deviations each measured mass is from each of the 48 known species
        # masses, then use norm.cdf to convert to probabilities.
        prob_matrix = np.zeros((test_size,len(masses_ref)))
        prob_matrix = (np.reshape(masses_test, (-1,1)) - precise_masses) / (1000 * unc)
        prob_matrix = (norm.cdf(-abs(prob_matrix)))

        # Calculate relative probabilities by scaling row sum to 1
        self.rel_prob_matrix = 1/np.reshape(np.sum(prob_matrix,1),(-1,1)) * prob_matrix
        # Print out rows from the relative probability matrix to give us an idea of the output
        print("\n -------------------------------First four rows of rel_prob_matrix: \n", np.round(self.rel_prob_matrix[0:4,:],3), "\n-------------------------------")

        return self.rel_prob_matrix
    

    # Save a list of just the top n probabilities in each row of the probability matrix along with their corresponding species and reference document masses
    def identify_top_n_species(self, n):
        masses_ref = self.masses_ref
        species_ref = self.species_ref
        rel_prob_matrix = self.rel_prob_matrix
        top_n_list = []

        # Round probabilities to 3 decimal places for succinctness later and order probabilities from greatest to least
        for row in rel_prob_matrix:
            ind_n = np.flip(np.argsort(row)[-n:])
            top_n_ref_masses = masses_ref[ind_n]
            top_n_species_names = species_ref[ind_n]
            top_n_probs = np.round(row[ind_n], 3)
            # Add our tuple of 3 lists: n masses (float), n names (string), n probabilities (float)
            # Ex:  [1.0073, 2.0152, 6.0146], ['H+', 'H2+', '6Li+'], [0.761, 0.239, 0.000]
            top_n_list.append([list(top_n_ref_masses), list(top_n_species_names), list(top_n_probs)])

        print("\n---------top_n_list: \n\n", top_n_list[0:6])

        return top_n_list


    # Print out accuracy statistics and a list of incorrect classifications.
    # def measure_accuracy(self, rel_prob_matrix):
    #     data = self.data_ref
    #     x_data = self.x_data
    #     y_data = self.y_data
    #     test_size = self.test_size
        
    #     # Now let's compare our array with the real labels and determine the classification accuracy of the Gaussian Process
    #     y_pred = np.argmax(rel_prob_matrix, 1)
    #     species_pred = [data.at[y, 'Species'] for y in y_pred]
    #     species_act = [data.at[y, 'Species'] for y in y_data]

    #     # print("\nPredicted: \n", species_pred)
    #     # print("\nActual: \n", species_act)

    #     class_acc = (y_pred == y_data).mean()*100
    #     print("\nClassification Accuracy: ", str(class_acc) + "%")


    #     # Show some incorrect classifications and what they should be
    #     y_pred, y_data, species_pred, species_act = np.array(y_pred), np.array(y_data), np.array(species_pred), np.array(species_act)
    #     prediction_mask = (y_pred != y_data)

    #     print("\nIncorrect Predictions: \n", species_pred[prediction_mask])
    #     print("\nCorrect Species: \n", species_act[prediction_mask])

    #     print("\nIndices of Incorrect Predictions: \n", [i for i in range(len(prediction_mask)) if prediction_mask[i]])