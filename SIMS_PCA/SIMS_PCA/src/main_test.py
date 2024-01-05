import customtkinter
import os
import sys
import subprocess
import logging
# Disable matplotlib font logging (it outputs unnecessary info about missing fonts)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from pca_sims.pca_sims import pca_sims


# TODO Fix redundant initialization of paths and pos_or_neg between Ctk entries / buttons and original code
# Specifies the basic structure of the GUI application
class App(customtkinter.CTk):
    def __init__(self):
        # --------------------------------------------------------------------------- Setup for basic GUI ---------------------------------------------------------------------------
        super().__init__()
        self.geometry('960x720')
        customtkinter.set_appearance_mode('dark')
        customtkinter.set_default_color_theme('blue')
        self.title('PCA Analysis Hub')
        self.grid_columnconfigure(0, weight=1)


        # Add entries for the paths to essential files
        self.pca_dir_label = customtkinter.CTkLabel(self, text='Enter PCA directory here: ', width=40, height=20)
        self.pca_dir_label.grid(row=0, column=0, padx=10, pady=(50, 5), sticky="w")
        self.pca_dir_entry = customtkinter.CTkEntry(self, textvariable=customtkinter.StringVar(value='/home/welch688/pca-analysis/SIMS_PCA/SIMS_PCA/'), width=400, height=20)
        self.pca_dir_entry.grid(row=0, column=1, padx=10, pady=(50, 5))

        self.raw_data_label = customtkinter.CTkLabel(self, text='Enter name of raw data file with file extension here: ', width=40, height=20)
        self.raw_data_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.raw_data_entry = customtkinter.CTkEntry(self, textvariable=customtkinter.StringVar(value='High P Pasture_Chris_Positive.txt'), width=400, height=20)
        self.raw_data_entry.grid(row=1, column=1, padx=10, pady=5)

        self.report_label = customtkinter.CTkLabel(self, text='Enter desired name of report with file extension here: ', width=40, height=20)
        self.report_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.report_entry = customtkinter.CTkEntry(self, textvariable=customtkinter.StringVar(value='Report High P Pasture_Chris_Positive.docx'), width=400, height=20)
        self.report_entry.grid(row=2, column=1, padx=10, pady=5)


        # Add buttons to select whether the report is for positive or negative ion data
        self.label_pos_or_neg = customtkinter.CTkLabel(self, text='Select whether current data contains positive or negative ions: ', width=15, height=80)
        self.label_pos_or_neg.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        self.pos_or_neg = customtkinter.CTkSegmentedButton(self, width=400, height=80, command=self.pos_or_neg_button_callback)
        self.pos_or_neg.grid(row=3, column=1, padx=10, pady=10)
        self.pos_or_neg.configure(values=['positive', 'negative'])
        self.pos_or_neg.set('positive')


        # Add checkboxes for selecting sample group numbers
        self.button_sample_groups = customtkinter.CTkButton(self, text='Edit group numbers to analyze', width=400, height=40, command=self.sample_group_callback)
        self.button_sample_groups.grid(row=4, column=1, padx=50, pady=5)


        # Add button to pull up metadata file in order to edit it
        self.button_metadata = customtkinter.CTkButton(self, text='Edit metadata file', width=400, height=40, command=self.metadata_callback)
        self.button_metadata.grid(row=5, column=1, padx=50, pady=5)


        # Add entry to change number of PCA components desired for plots and tables in the report
        self.pcacomp_label = customtkinter.CTkLabel(self, text='Change number of PCA components used in report:', width=40, height=20)
        self.pcacomp_label.grid(row=6, column=0, padx=10, pady=(50,10), sticky='w')
        self.pcacomp_entry = customtkinter.CTkEntry(self, textvariable=customtkinter.StringVar(value='5'), width=400, height=20)
        self.pcacomp_entry.grid(row=6, column=1, padx=10, pady=(50,10))


        # Finally, add a button to update the document values from the report and another to generate the report from the current database
        self.button_update = customtkinter.CTkButton(self, text='Update document values database from report', width=980, height=50, command=self.update_callback)
        self.button_update.grid(row=7, padx=50, pady=(50, 5), sticky='ew', columnspan=2)
        self.button_generate = customtkinter.CTkButton(self, text='Generate PCA report from current database', width=980, height=50, command=self.generate_callback)
        self.button_generate.grid(row=8, padx=50, pady=(5, 50), sticky='ew', columnspan=2)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        # ---------------------------------------------------- Initialize various paths and parameters for pcasims to use later------------------------------------------------------
        # The main PCA folder (for example, if storing your data on Windows at PNNL, this could be
        # /mnt/c/Users/<INSERT USERNAME>/'OneDrive - PNNL'/Documents/pca-analysis/SIMS_PCA/SIMS_PCA).
        self.pca_dir = '/home/welch688/pca-analysis/SIMS_PCA/SIMS_PCA/'

        # TODO ^ (See above; redundant?)
        # SIMS data
        self.f_rawsims_data = os.path.join(self.pca_dir, 'sims-data/OriginalData/', 'High P Pasture_Chris_Positive.txt')

        # Store the subset of groups from the data above which the user wants to analyze
        self.f_group_numbers = os.path.join(self.pca_dir, 'sims-data/OriginalData/_groupnumbers.txt')

        # SIMS metadata
        self.f_metadata = os.path.join(self.pca_dir, 'sims-data/OriginalData/_metadata.txt')

        # SIMS-PCA report
        self.f_report = os.path.join(self.pca_dir, 'output_sample/', 'report.docx')

        # Output folder
        self.outDir = os.path.join(self.pca_dir, 'output_sample')

        # Document positive and negative mass
        self.f_doc_positive_mass = os.path.join(self.pca_dir, "sims-data", "positive_doc_mass_record.csv")
        self.f_doc_negative_mass = os.path.join(self.pca_dir, "sims-data", "negative_doc_mass_record.csv")

        # Indicates to rest of code whether we are handling positive or negative ions
        self.positive_or_negative_ion = 'positive'
        self.f_doc_mass = self.f_doc_positive_mass if self.positive_or_negative_ion == 'positive' else self.f_doc_negative_mass
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------- Create the callback commands that will run the program ------------------------------------------------------------
    # Callback on positive / negative button that identifies the type of ions in the data file
    def pos_or_neg_button_callback(self, value: str):
        self.positive_or_negative_ion = value
        print('Selected \'' + value + '\' doc mass.')

        if self.positive_or_negative_ion == 'positive':
            self.f_doc_mass = self.f_doc_positive_mass
        elif self.positive_or_negative_ion == 'negative':
            self.f_doc_mass = self.f_doc_negative_mass
        else:
            print('***Error! Invalid input for positive_or_negative_ion; choose \'positive\' or \'negative\'***')
            sys.exit()


    # TODO Change from directly editing file to scrollable checkbox dropdown contained in UI
    # Callback on button_sample_groups that opens the sample groups file for the user to edit
    def sample_group_callback(self):
        subprocess.call(['notepad.exe', self.f_group_numbers])


    # TODO Change from directly editing file to CTkEntries contained in UI
    # Callback on button_metadata that opens the metadata file for the user to edit
    def metadata_callback(self):
        subprocess.call(['notepad.exe', self.f_metadata])


    # Update the report from user changes
    def update_callback(self):
        # Initialize pcasims if it doesn't already exist and update it with any user changes the user has made to its desired parameters
        try:
            self.update_pca_instance()
        except:
            return

        print('-------->Updating Peak Assignments from User Changes to Report...')

        # Perform database update
        self.pcasims.update_classifications(self.f_report)

        print('-------->Done.')


    # Generate the report
    def generate_callback(self):
        # Initialize pcasims if it doesn't already exist and update it with any user changes the user has made to its desired parameters
        try:
            self.update_pca_instance()
        except:
            return

        # Perform PCA
        self.pcasims.perform_pca()

        # Identify chemical components based on existing document mass; use user-specified string positive_or_negative_ion
        # to distinguish whether we should analyze using data from the .csv file containing + or - ions.
        # n is the maximum number of species classification possibilities to display in the table; change to desired value if needed.
        self.pcasims.identify_components_from_file(n=4)

        # Plot PCA result
        self.pcasims.plot_pca_result(self.max_pcacomp)

        # Generate the report
        self.pcasims.generate_report(f_report=self.f_report, ion_sign=self.positive_or_negative_ion, max_pcacomp=self.max_pcacomp)

        print('-------->Data Exporting...')
        print('\n\n\n-------->Congratulations!')
        print('-------->Please Check Results in the output_sample Folder.')
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------- Set up backend input based on user changes  -----------------------------------------------------------------
    def update_pca_instance(self):
        # Update PCA directory
        if (('SIMS_PCA/SIMS_PCA/' in self.pca_dir_entry.get()) and self.pca_dir_entry.get()[-1] == '/'):
            self.pca_dir = self.pca_dir_entry.get()
            print('-------->Processed PCA directory successfully.')
        elif (not self.pca_dir_entry.get()):
            print('***Error! Empty entry. Please enter text and try again.***')
            raise ValueError
        else:
            print('***Error! Invalid input for pca_dir. Please make sure it ends with \'pca-analysis/SIMS_PCA/SIMS_PCA/\' and try again.***')
            raise ValueError
        
        # Update raw data file
        if (self.raw_data_entry.get()):
            self.f_rawsims_data = os.path.join(self.pca_dir, 'sims-data/OriginalData/', self.raw_data_entry.get())
            print('-------->Processed raw data file name successfully.')
        else:
            print('***Error! Empty entry. Please enter text and try again.***')
            raise ValueError

        # Update report
        if (self.report_entry.get()):
            self.f_report = os.path.join(self.pca_dir, 'output_sample/', self.report_entry.get())
            print('-------->Processed report file name successfully.')
        else:
            print('***Error! Empty entry. Please enter text and try again.***')
            raise ValueError
        
        # Check whether number of PCA components desired is actually an integer
        if (self.pcacomp_entry.get()):
            if (self.isint(self.pcacomp_entry.get())):
                self.max_pcacomp = int(self.pcacomp_entry.get())
            else:
                print(('***Error! Number of PCA components given is not an integer.***'))
                raise ValueError

            print('-------->Processed number of PCA components successfully.')
        else:
            print('***Error! Empty entry. Please enter text and try again.***')
            raise ValueError

        # Initialize the pca_sims instance
        self.pcasims = pca_sims(self.f_rawsims_data, self.f_metadata, self.f_doc_mass, self.pca_dir, self.outDir, self.positive_or_negative_ion, self.f_group_numbers)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Useful helper method for telling whether value x can be cast to int
    def isint(self, x):
        try:
            x_test = int(x)
        except (TypeError, ValueError):
            return False
        
        return True





# --------------------------------------------------------------------------- Run the application ---------------------------------------------------------------------------
app = App()
app.mainloop()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------