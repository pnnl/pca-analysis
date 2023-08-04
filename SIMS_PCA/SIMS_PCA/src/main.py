"""The main script for performing PCA-SIMS analysis"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import patches
import logging
# Disable matplotlib font logging (it outputs unnecessary info about missing fonts)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from pca_sims.pca_sims import pca_sims

# The main PCA folder (if storing your data on Windows, this is probably
# /mnt/c/Users/<INSERT USERNAME>/OneDrive - PNNL/Documents/pca/SIMS_PCA/SIMS_PCA).
pcaDir = "/home/cswelch/pca/SIMS_PCA/SIMS_PCA"

# Output folder
outDir = os.path.join(pcaDir, 'output_sample')

# Document positive and negative mass
f_doc_positive_mass = os.path.join(pcaDir, "sims-data", "positive_doc_mass_record.csv")
f_doc_negative_mass = os.path.join(pcaDir, "sims-data", "negative_doc_mass_record.csv")

# TODO Improve the end-user interface for changing the next two variables?
# Indicates to rest of code whether we are handling positive or negative ions
positive_or_negative_ion = 'positive'

# SIMS data
f_rawsims_data = os.path.join(pcaDir, 'sims-data/OriginalData/DATA_POSITIVE_20020202_Zihua_soils.TXT')

# SIMS metadata
f_metadata = os.path.join(pcaDir, 'sims-data/OriginalData/metadata.txt')

# Initialize the pca_sims instance
pcasims = pca_sims(f_rawsims_data, f_metadata, pcaDir, outDir, positive_or_negative_ion)


# TODO Implement the update side of this if statement
# Take user input to decide whether we would like to do PCA (usually done on the first pass) or update the
# document-based classifications with calibrated data (usually done on later passes).
do_update = input('-------->Would you like to update values using measured masses (y/n)? If not, I will ' +
                  'assume you want to do PCA. \n')

if do_update.strip() == 'y':
    print('-------->Updating Peak Assignments Using Measured Masses...')

    pcasims.update_classifications()

    print('-------->Done.')
else:
    # Perform PCA
    pcasims.perform_pca()

    # Identify chemical components based on existing document mass; use user-specified string positive_or_negative_ion
    # to distinguish whether we should analyze using data from the .csv file containing + or - ions
    if positive_or_negative_ion == 'positive':
        pcasims.identify_components_from_file(f_doc_positive_mass)
    elif positive_or_negative_ion == 'negative':
        pcasims.identify_components_from_file(f_doc_negative_mass)

    # Assign IDs and probabilities to the PCA data using the components found above
    pcasims.classify_species()

    # Rule-based analysis
    pcasims.perform_rule_based_analysis()

    # Plot PCA result
    pcasims.plot_pca_result()

    # Generate the report
    pcasims.generate_report()

    print('-------->Data Exporting...')
    print('\n\n\n-------->Congratulations!')
    print('-------->Please Check Results In The output_sample Folder.')
