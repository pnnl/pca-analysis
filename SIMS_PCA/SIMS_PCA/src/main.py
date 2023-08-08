"""The main script for performing PCA-SIMS analysis"""

import os
import sys
import logging
# Disable matplotlib font logging (it outputs unnecessary info about missing fonts)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from pca_sims.pca_sims import pca_sims

# The main PCA folder (if storing your data on Windows, this is probably
# /mnt/c/Users/<INSERT USERNAME>/OneDrive - PNNL/Documents/pca/SIMS_PCA/SIMS_PCA).
pcaDir = "/home/cswelch/pca/SIMS_PCA/SIMS_PCA"

# Output folder
outDir = os.path.join(pcaDir, 'output_sample')

# TODO Improve the end-user interface for positive_or_negative_ion, f_rawsims_data, and f_metadata?
# Indicates to rest of code whether we are handling positive or negative ions
positive_or_negative_ion = 'negative'

# SIMS data
f_rawsims_data = os.path.join(pcaDir, 'sims-data/OriginalData/DATA_NEGATIVE_20020202_Zihua_soils.TXT')

# SIMS metadata
f_metadata = os.path.join(pcaDir, 'sims-data/OriginalData/metadata.txt')

# SIMS-PCA report
f_report = os.path.join(pcaDir, 'output_sample/report.docx')

# Document positive and negative mass
f_doc_positive_mass = os.path.join(pcaDir, "sims-data", "positive_doc_mass_record.csv")
f_doc_negative_mass = os.path.join(pcaDir, "sims-data", "negative_doc_mass_record.csv")

# User input specifies which document of stored masses should be used for the pca simulations
if positive_or_negative_ion == 'positive':
    f_doc_mass = f_doc_positive_mass
elif positive_or_negative_ion == 'negative':
    f_doc_mass = f_doc_negative_mass
else:
    print('***Error! Invalid input for positive_or_negative_ion; choose \'positive\' or \'negative\'***')
    sys.exit()

# Initialize the pca_sims instance
pcasims = pca_sims(f_rawsims_data, f_metadata, f_doc_mass, pcaDir, outDir, positive_or_negative_ion)


# TODO Implement the update side of this if statement
# Take user input to decide whether we would like to do PCA (usually done on the first pass) or update the
# document-based classifications with calibrated data (usually done on later passes).
do_update = input('-------->Would you like to update values using measured masses (y/n)? If not, I will ' +
                  'assume you want to do PCA. \n')

if do_update.strip() == 'y':
    print('-------->Updating Peak Assignments Using Measured Masses...')

    pcasims.update_classifications(f_doc_mass, f_report)

    print('-------->Done.')
else:
    # Perform PCA
    pcasims.perform_pca()

    # Identify chemical components based on existing document mass; use user-specified string positive_or_negative_ion
    # to distinguish whether we should analyze using data from the .csv file containing + or - ions
    pcasims.identify_components_from_file()

    # Rule-based analysis
    pcasims.perform_rule_based_analysis()

    # Plot PCA result
    pcasims.plot_pca_result()

    # Generate the report
    pcasims.generate_report(f_report=f_report)

    print('-------->Data Exporting...')
    print('\n\n\n-------->Congratulations!')
    print('-------->Please Check Results In The output_sample Folder.')
