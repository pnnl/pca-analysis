"""This class allows us to calculate the most likely chemical species classifications from given input lists of reference and test atomic weights."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


class species_classifier:

    # Constructor; takes the following parameters:
    #       data - The mass vs. species data (either positive or negative ions)
    def __init__(self, data:pd.DataFrame, doc_mass_list:list, species_list:list):
        # TODO Find a more efficient way of not altering the original DataFrame
        self.data = data.copy(deep=True)
        
        # Get how many masses need to be tested
        test_size = self.data.shape[0]
        # Get how many classes we need from the original data
        output_dim = self.data.shape[0]
        print("Num output classes: ", output_dim)

        self.test_size = test_size
        print(doc_mass_list)
        print(species_list)
        self.masses_document = np.array(doc_mass_list)
        self.species_document = np.array(species_list)
        self.masses_test = self.data['raw_mass'].to_numpy()

        # Get approximate uncertainty from the data by comparing the test and document masses in the mass id file (data_test). We can exclude
        # the document masses that have multiple potential classifications since they might artificially inflate the uncertainty depending on
        # which one we select.
        a = self.masses_test
        b = self.data['document_mass'].to_numpy()

        for i in range(len(b)):
            if type(b[i]) != list:
                pass
            elif len(b[i]) != 1:
                b[i] = np.nan
            else:
                b[i] = b[i][0]

        residues = np.array(a - b)
        residues = residues[~pd.isnull(residues)]
        print(">>>>>>> Residues: ", residues)
        self.uncertainty = np.std(abs(residues))
        # self.uncertainty = 0.03
        print(">>>>>>> Uncertainty: ", self.uncertainty)

    # Read the data, form representative Gaussian distributions around each peak, 
    # Returns: a matrix with each row representing the sample number and the columns in each row containing the probabilities. 
    #          All entries in a row add to a total probability of 1.
    def calculate_probabilities(self):
        masses_document = self.masses_document
        masses_test = self.masses_test
        test_size = self.test_size
        
        # Make the known species masses into a row, then repeat for the length of the training data to prepare for mass array operations
        # in the next cell
        precise_masses = np.reshape(masses_document, (1,-1))
        precise_masses = precise_masses.repeat(test_size,0)
        print("\nPrecise_masses: \n", precise_masses)
        print("\nmasses_test: \n", masses_test)

        # Calculate standard deviations and resulting probabilities for each mass according to its uncertainty pdf.
        # We transform the x_data from 1 x n to n x 1, then broadcast it to n x 48 while subtracting precise_masses and dividing
        # by the uncertainty to get the number of standard deviations each measured mass is from each of the 48 known species
        # masses, then use norm.cdf to convert to probabilities.
        prob_matrix = np.zeros((test_size,len(masses_document)))
        prob_matrix = (np.reshape(masses_test, (-1,1)) - precise_masses) / (self.uncertainty)
        prob_matrix = (norm.cdf(-abs(prob_matrix)))

        # Calculate relative probabilities by scaling row sum to 1
        self.rel_prob_matrix = 1/np.reshape(np.sum(prob_matrix,1),(-1,1)) * prob_matrix
        # Print out rows from the relative probability matrix to give us an idea of the output
        print("\n -------------------------------First four rows of rel_prob_matrix: \n", np.round(self.rel_prob_matrix[0:4,:],3), "\n-------------------------------")

        return self.rel_prob_matrix
    

    # Save a list of just the top n probabilities in each row of the probability matrix along with their corresponding 
    # species and reference document masses. However, we'll only keep elements that are within +/-0.5 of the Precise Mass
    # value to ensure we aren't giving useless predictions, so there may be fewer than n elements.
    def identify_top_n_species(self, n):
        masses_document = self.masses_document
        species_document = self.species_document
        masses_test = self.masses_test
        rel_prob_matrix = self.rel_prob_matrix
        top_n_list = []

        # TODO Also cut off species with probability less than 1%
        # Increment that tells us which test mass to compare with our precise masses each loop
        row_index = 0
        # Round probabilities to 3 decimal places for succinctness later and order probabilities from greatest to least
        for row in rel_prob_matrix:
            # Get the indices of the most likely candidates in decreasing order of probability; throw out candidates too far away from the target
            ind_n = np.flip(np.argsort(row)[-n:])
            ind_n = ind_n[abs(masses_document[ind_n] - masses_test[row_index]) < 0.5]

            top_n_ref_masses = masses_document[ind_n]
            top_n_species_names = species_document[ind_n]
            top_n_probs = np.round(row[ind_n], 3)
            # Add our list of 3 lists: n masses (float), n names (string), n probabilities (float)
            # Ex:  [1.0073, 2.0152], ['H+', 'H2+'], [0.761, 0.239]
            top_n_list.append([list(top_n_ref_masses), list(top_n_species_names), list(top_n_probs)])

            row_index += 1

        print("\n---------top_n_list: \n\n", top_n_list[:50])

        return top_n_list