import customtkinter
import os
import sys
import logging
import traceback
# Disable matplotlib font logging (it outputs unnecessary info about missing fonts)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from pca_sims.pca_sims import pca_sims



# ------------------------------------------------------------------------ Set up useful variables  ------------------------------------------------------------------------
# The main PCA folder (for example, if storing your data on Windows at PNNL, this could be
# /mnt/c/Users/<INSERT USERNAME>/'OneDrive - PNNL'/Documents/pca-analysis/SIMS_PCA/SIMS_PCA).
pcaDir = '/home/welch688/pca-analysis/SIMS_PCA/SIMS_PCA/'

# Output folder
outDir = os.path.join(pcaDir, 'output_sample')

# TODO positive / negative ion editing
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
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------- Create the callback commands that will run the program ------------------------------------------------------------
# TODO The actual report and document masses don't seem to update while the program + GUI are running; still have to stop them each time. Why aren't updates happening automatically?
def update_callback():
    print('-------->Updating Peak Assignments from User Changes to Report...')

    # Perform database update
    pcasims.update_classifications(f_doc_mass, f_report)

    print('-------->Done.')

def generate_callback():
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
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------- Setup for basic GUI ---------------------------------------------------------------------------
app = customtkinter.CTk()
app.geometry('1080x720')
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('blue')
app.title('PCA Analysis Hub')
app.columnconfigure(0, weight=1)


# TODO Figure out best way to add a callback command to get the entry values below (use a button?)
# Add entries for the paths to essential files
pcaDir_label = customtkinter.CTkLabel(app, text='Enter PCA directory here: ', width=40, height=20)
pcaDir_label.grid(row=0, column=0, padx=50, pady=(50, 5), sticky="w")
pcaDir = customtkinter.CTkEntry(app, placeholder_text='/home/welch688/pca-analysis/SIMS_PCA/SIMS_PCA/', width=400, height=20)
pcaDir.grid(row=0, column=1, padx=50, pady=(50, 5), sticky="e")

raw_data_label = customtkinter.CTkLabel(app, text='Enter name of raw data file with file extension here: ', width=40, height=20)
raw_data_label.grid(row=1, column=0, padx=50, pady=5, sticky="w")
raw_data = customtkinter.CTkEntry(app, placeholder_text='Hifh P Pasture_Chris_Positive.TXT', width=400, height=20)
raw_data.grid(row=1, column=1, padx=50, pady=5, sticky="e")

report_label = customtkinter.CTkLabel(app, text='Enter desired name of report with file extension here: ', width=40, height=20)
report_label.grid(row=2, column=0, padx=50, pady=5, sticky="w")
report = customtkinter.CTkEntry(app, placeholder_text='report.docx', width=400, height=20)
report.grid(row=2, column=1, padx=50, pady=5, sticky="e")


# Add buttons to select whether the report is for positive or negative ion data
label_pos_or_neg = customtkinter.CTkLabel(app, text='Select whether current report contains positive or negative ions: ', width=150, height=80)
label_pos_or_neg.grid(row=3, column=0, padx=50, pady=10, sticky='w')
pos_or_neg = customtkinter.CTkSegmentedButton(app, width=100, height=80)
pos_or_neg.grid(row=3, column=1, padx=50, pady=10, sticky='e')
pos_or_neg.configure(values=['positive', 'negative'])
pos_or_neg.set('positive')


# TODO Sample group numbers button and editing
# Add checkboxes for selecting sample group numbers


# TODO Metadata button and editing
# Add button to pull up metadata file in order to edit it


# Finally, add a button to update the document values from the report and another to generate the report from the current database
button_update = customtkinter.CTkButton(app, text='Updated document values database from report', width=980, height=50, command=update_callback)
button_update.grid(row=4, padx=50, pady=(50, 5), sticky="ew", columnspan=2)
button_generate = customtkinter.CTkButton(app, text='Generate PCA report from current database', width=980, height=50, command=generate_callback)
button_generate.grid(row=5, padx=50, pady=(5, 50), sticky="ew", columnspan=2)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------- Run the application ---------------------------------------------------------------------------
app.mainloop()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------