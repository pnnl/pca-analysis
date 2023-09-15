"""The main script for performing PCA-SIMS analysis."""

import os
import sys
import logging
import traceback
# Disable matplotlib font logging (it outputs unnecessary info about missing fonts)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from pca_sims.pca_sims import pca_sims


# ------------------------------------------------- VALUES THAT USERS MAY NEED TO CHANGE -------------------------------------------------
# The main PCA folder (for example, if storing your data on Windows at PNNL, this could be
# /mnt/c/Users/<INSERT USERNAME>/'OneDrive - PNNL'/Documents/pca-analysis/SIMS_PCA/SIMS_PCA).
pcaDir = '/home/welch688/pca-analysis/SIMS_PCA/SIMS_PCA/'

# Output folder
outDir = os.path.join(pcaDir, 'output_sample')

# TODO Implement GUI using CustomTkinter to get positive_or_negative_ion, f_rawsims_data, f_metadata, and f_report from user
# Indicates to rest of code whether we are handling positive or negative ions
positive_or_negative_ion = 'positive'

# SIMS data
f_rawsims_data = os.path.join(pcaDir, 'sims-data/OriginalData/Hifh P Pasture_Chris_Positive.TXT')

# Store the subset of groups from the data above which the user wants to analyze
f_group_numbers = os.path.join(pcaDir, 'sims-data/OriginalData/_groupnumbers.txt')

# SIMS metadata
f_metadata = os.path.join(pcaDir, 'sims-data/OriginalData/_metadata.txt')

# SIMS-PCA report
f_report = os.path.join(pcaDir, 'output_sample/report.docx')
# -----------------------------------------------------------------------------------------------------------------------------------------


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
pcasims = pca_sims(f_rawsims_data, f_metadata, f_doc_mass, pcaDir, outDir, positive_or_negative_ion, f_group_numbers)


# Take user input to decide whether we would like to do PCA as yes or no (usually done on the first pass) or update the
# document-based classifications with calibrated data (usually done on later passes).
# If the user does not enter a valid form of yes or no, then we allow for new attempts until a valid string is given.
while True:
    do_update = input('-------->If you would like to update the document values database from the report, type \'y\' and press Enter; ' +
                       'if you would like to generate the PCA report instead, enter \'n\' below.\n').strip()
    
    if (do_update != 'y') and (do_update != 'Y') and (do_update != 'n') and (do_update != 'N'):
        print('\n***Invalid option selected; enter either \'y\' for yes or \'n\' for no.\n')
        continue
    else:
        break


if (do_update == 'y') or (do_update == 'Y'):
    print('-------->Updating Peak Assignments from User Changes to Report...')

    # Perform database update
    pcasims.update_classifications(f_doc_mass, f_report)

    print('-------->Done.')
else:
    # Perform PCA
    pcasims.perform_pca()

    # Identify chemical components based on existing document mass; use user-specified string positive_or_negative_ion
    # to distinguish whether we should analyze using data from the .csv file containing + or - ions.
    # n is the maximum number of species classification possibilities to display in the table; change to desired value if needed.
    pcasims.identify_components_from_file(n=2)

    # Plot PCA result
    pcasims.plot_pca_result()

    # Generate the report
    pcasims.generate_report(f_report=f_report, ion_sign=positive_or_negative_ion)

    print('-------->Data Exporting...')
    print('\n\n\n-------->Congratulations!')
    print('-------->Please Check Results in the output_sample Folder.')