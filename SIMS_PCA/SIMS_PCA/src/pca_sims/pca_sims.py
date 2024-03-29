"""The class for performing PCA analysis on SIMS data."""

import re
import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import docx

from .species_classifier import species_classifier
from .plotting import plot_pca_result
from .report import pca_sims_report


positive_ion_category = ["Hydrocarbons", "Oxygen-containing organics", "Nitrogen-containing organics", "Benzene-containing organics", "PDMS"]
negative_ion_category = ["Hydrocarbons", "Nitrogen-containing organics", "SiOx", "SOx", "POx", "NOx", "Benzene-containing organics", "Organic acids", "Fatty acids"]


class pca_sims(object):

    def __init__(
        self,
        f_rawsims_data: str,
        f_metadata: str,
        f_doc_mass: str,
        pcaDir: str,
        outDir: str,
        positive_or_negative_ion: str,
        f_group_numbers: str
    ):
        print('\n-------->Reading Data...')
        # Read SIMS data
        try:
            rawdata=pd.read_csv(f_rawsims_data, sep='\t')
            rawdata.dropna(inplace=True)
            mass_raw = rawdata['Mass (u)'].values
            rawdata['Mass (u)']=rawdata['Mass (u)'].apply(np.round).astype(int)
            rawdata.set_index(rawdata['Mass (u)'], inplace=True)
            mass=rawdata.index
            rawdata.drop(columns=['Mass (u)'], inplace=True)
            # print("rawdata: ", rawdata)
            # print("mass: ", mass)
        except:
            print(traceback.print_exc())
            print('***Error! Cannot Find Correct Raw Data File!***')
            sys.exit()

        # TODO We don't throw an error on numbers entered into f_group_numbers .txt file that don't match up with the samples.
        #      We print all of them out, but incorrect numbers won't affect the filtering anyway. Is this assumption fine?
        # Read subset of group numbers on which we want to perform PCA from user-specified .csv file, then filter the raw data so that 
        # it only contains those columns
        try:
            group_nums = pd.read_csv(f_group_numbers)
            group_nums = group_nums['Group'].unique().tolist()
            group_nums.sort()
            print('\n\tSample group numbers: ', group_nums, '\n')

            columns_to_drop = []
            for label in rawdata.columns:
                label_has_no_sub_group_num = all([not str(num) in label for num in group_nums])
                if label_has_no_sub_group_num:
                    columns_to_drop.append(label)

            rawdata.drop(columns=columns_to_drop, inplace=True)
        except:
            print(traceback.print_exc())
            print('***Error! Group Numbers File Missing or Contains Incorrectly Formatted Values!***')
            sys.exit()
        
        # Read data description
        description = {}
        metadata_df = pd.read_csv(f_metadata, index_col=0, header=None)
        description['experiment'] = metadata_df.loc['Experiment',1]
        description['date'] = metadata_df.loc['Date',1]
        description['operator'] = metadata_df.loc['Operator',1]

        # Extract the sample names (e.g., Goethite-Tannic Acid 1400 ppm) from the metadata file. Note that we
        # must exclude the first 4 lines since they include other information.
        sample_description_set = []
        n_samples = metadata_df.shape[0] - 3
        for i in range(n_samples):
            sample_number      = metadata_df.index[i+3]
            sample_description = metadata_df.loc[sample_number,1]
            sample_description_set.append([int(sample_number), str(sample_description)])
        
        nmass, ncomp = rawdata.shape

        self.f_rawsims_data         = f_rawsims_data
        self.f_metadata             = f_metadata
        self.pcaDir                 = pcaDir
        self.outDir                 = outDir
        self.description            = description
        self.sample_description_set = sample_description_set
        self.rawdata                = rawdata
        self.mass                   = mass
        self.mass_raw               = mass_raw
        self.nmass                  = nmass
        self.ncomp                  = ncomp
        self.f_group_numbers        = f_group_numbers

        if positive_or_negative_ion == 'positive':
            self.positive_ion = True
        else:
            self.positive_ion = False

        # Generate the output folder if it does not exist
        if not os.path.exists(os.path.join(pcaDir, outDir)):
            os.makedirs(os.path.join(pcaDir, outDir))

        # Initialize the mass identification; make sure to sort by the raw masses so we don't mess up the classification
        # order when we write the report
        self.mass_id = pd.DataFrame(columns=['raw_mass', 'document_mass', 'true_assignment', 'possible_assignment'], index=mass)
        self.mass_id['raw_mass'] = mass_raw
        self.mass_id.sort_values(by=['raw_mass'], inplace=True)

        # TODO Currently, measured masses from previous document disappear upon switching from positive to negative document or vice versa. Is this fine?
        # Save the measured masses for ID later
        self.f_measured_masses = os.path.join(self.pcaDir, 'sims-data/measured_masses.csv')
        self.measured_masses = pd.read_csv(self.f_measured_masses)

        # Save the unit mass assignments from the chosen .csv document
        self.f_doc_mass = f_doc_mass
        self.doc_mass = pd.read_csv(f_doc_mass)
        self.doc_mass.set_index(self.doc_mass['Unit Mass'], inplace=True)

    
    def perform_pca(self):
        """Perform PCA on SIMS data."""
        rawdata = self.rawdata

        print('-------->PCA Processing...')
        try:
            # NORMALIZE DATA
            scaled_data=rawdata.T**0.5
            samplelist=scaled_data.index
            labels=['PC'+str(x) for x in range(1,self.ncomp+1)]

            # TODO Add error for case where no group numbers from file match up with data (see line 143).
            # PCA
            pca=PCA()
            pca.fit(scaled_data)
            pca_data=pca.transform(scaled_data)
            pca_df=pd.DataFrame(pca_data,index=samplelist,columns=labels)

        except:
            print(traceback.print_exc())
            print('***Error! Cannot Recognize Data!***')
            sys.exit()
        
        self.scaled_data = scaled_data
        self.samplelist  = samplelist
        self.pca         = pca
        self.pca_data    = pca_data
        self.pca_df      = pca_df

        self._get_loading_scores()
    

    # Use the species_classifier class to assign IDs and probabilities to the PCA data using mass_id
    # Params:
    #       mass_id - This DataFrame includes the raw masses to be tested
    #       doc_mass_list - The list of document masses
    #       species_list - The list of species masses, for which there is a 1:1 correspondence with doc_mass_list
    #       n - The maximum number of candidate species to be displayed in the report
    # Returns:
    #       classifier - An instance of the species_classifier class built on the doc masses; pass this the raw_mass
    #                    values and, for each of them, get the corresponding probability of it being each of the species in the doc_mass
    #       rel_prob_matrix - Stores the relative probabilities with number of rows = number of test masses (e.g., 800) 
    #                         and number of columns = number of reference masses (e.g., 48)
    #       top_n_species - The top 5 potential candidates for each list; we add them to the report later
    def classify_species(self, mass_id_raw: pd.Series, mass_id_doc: pd.Series, doc_mass_list: list, species_list: list, n: int):
        classifier = species_classifier(mass_id_raw, mass_id_doc, doc_mass_list, species_list)
        rel_prob_matrix = classifier.calculate_probabilities()
        top_n_species = classifier.identify_top_n_species(n)

        return classifier, rel_prob_matrix, top_n_species


    # Identify the PCA components using the file we are given.
    # Params:
    #   n - The number of species we desire to display in the report from the top n most probable ID candidates
    def identify_components_from_file(self, n:int):
        """Identify chemical components from the file passed to pca_sims."""
        print('-------->Finding assigned unit masses from file...')
        doc_mass = self.doc_mass

        # Store the possible assignments and document masses in a 1d list to get them in a format easily used by species_classifier later
        doc_mass_list = []
        species_list = []

        for unit_mass in self.mass_id.index:
            if unit_mass in doc_mass.index:
                assignment    = doc_mass.loc[unit_mass, 'Assignment'].split(',')
                assignment    = [assign.strip() for assign in assignment]
                document_mass = doc_mass.loc[unit_mass, 'Document Mass'].split(',') 
                document_mass = [float(mass) for mass in document_mass]
                if not isinstance(self.mass_id.loc[unit_mass, 'possible_assignment'], list):
                    self.mass_id.at[unit_mass, 'possible_assignment'] = assignment
                    self.mass_id.at[unit_mass, 'document_mass']       = document_mass
                else:
                    self.mass_id.at[unit_mass, 'possible_assignment'] += assignment
                    self.mass_id.at[unit_mass, 'document_mass']       += document_mass
                print('Identified unique mass {} from the documentation with Document Mass {} and assignment {}'.format(
                    unit_mass, document_mass, assignment))

                doc_mass_list.extend(document_mass)
                species_list.extend(assignment)

        # Do the same process as above to get the measured masses 'document_mass' column into the correct format (i.e., there are a 
        # bunch of floats and strings in this column; make sure to put all into lists of strings so species_classifier will process 
        # them correctly)
        for i in self.measured_masses.index:
            m_document_mass = str(self.measured_masses.loc[i, 'document_mass'])
            self.measured_masses.loc[i, 'document_mass'] = ''

            m_document_mass = m_document_mass.split(',')
            m_document_mass = [float(mass) for mass in m_document_mass]
            self.measured_masses.at[i, 'document_mass'] = m_document_mass

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print("measured_masses: ", self.measured_masses)

        # TODO Uncertainty calculated on all sample groups even if only a few subgroups are selected. Could this have a significant effect?
        # TODO Make second two outputs fields of the classifier instance instead of having them show up as separate variables here
        # Assign IDs and probabilities to the PCA data using the components found above
        self.classifier, self.rel_prob_matrix, self.top_n_species = self.classify_species(self.mass_id['raw_mass'], 
                                                                                                      self.mass_id['document_mass'], 
                                                                                                      doc_mass_list, 
                                                                                                      species_list, n)


    def plot_pca_result(
        self, 
        max_pcacomp:int=5
    ):
        """Plot PCA analysis result."""
        pca_maxpcacomp_df, fig_screeplot, fig_scores_set, fig_scores_confid_set, fig_scores_single_set, fig_loading_set = \
                plot_pca_result(self.pca, self.pca_data, self.samplelist, self.mass, 
                                self.sample_description_set, self.pcaDir, self.outDir, self.f_group_numbers, max_pcacomp)
        self.pca_maxpcacomp_df     = pca_maxpcacomp_df
        self.fig_screeplot         = fig_screeplot
        self.fig_scores_set        = fig_scores_set
        self.fig_scores_confid_set = fig_scores_confid_set
        self.fig_scores_single_set = fig_scores_single_set
        self.fig_loading_set       = fig_loading_set
    

    def generate_report(self, f_report:str='report.docx', ion_sign:str='positive', max_pcacomp:int=5):
        """Generate the report."""
        print('-------->Generating the report now...')

        # Initialize the report
        f_report = os.path.join(self.pcaDir, self.outDir, f_report)
        # Store flag for whether report file already exists. If it doesn't, then we don't want to do anything with measured masses since they haven't been entered yet.
        report_exists = os.path.exists(f_report)
        if report_exists:
             print('-------->Found an existing report. Running with measured masses...')
        else:
             print('-------->No existing report found. Running without measured masses...')
        self.report = pca_sims_report(f_report=f_report, ion_sign=ion_sign, description=self.description)

        # Include the overall pairwise PC plots in the report
        self.report.write_2dscore_plots(self.fig_scores_confid_set)

        if self.positive_ion:
            # PCA analysis of positive ToF-SIMS spectra
            for pcacomp in range(1,max_pcacomp+1):
                self.generate_analysis_pcacomp(report_exists, pcacomp)
        else:
            # PCA analysis of negative ToF-SIMS spectra
            for pcacomp in range(1,max_pcacomp+1):
                self.generate_analysis_pcacomp(report_exists, pcacomp)

        # Add the bar chart of percentage explained variance for each PC
        self.add_scree_plot()
        
        # Save the report
        self.report.save()
    

    def generate_analysis_pcacomp(self, report_exists:bool, pcacomp:int=1):
        """Generate the analysis for one pca component."""

        doc_mass = self.doc_mass

        # ---------------- Plot pages ------------------
        score_plot = self.fig_scores_single_set[pcacomp-1]

        # Write the loading plots
        loading_plot = self.fig_loading_set[pcacomp-1]

        # ----------------- Table pages -------------------
        # The top +/- 20 loading table
        fetchn_more=20
        loadingTable  = self.loadingTable.sort_values(by=[pcacomp])
        loadingTable_pcacomp = loadingTable[pcacomp]
        negative_topx = loadingTable.iloc[:fetchn_more][pcacomp].index.tolist()
        negative_topy = loadingTable.iloc[:fetchn_more][pcacomp].tolist()
        positive_topx = loadingTable.iloc[-fetchn_more:][pcacomp].index.tolist()[::-1]
        positive_topy = loadingTable.iloc[-fetchn_more:][pcacomp].tolist()[::-1]

        positive_loading_table=pd.DataFrame(
            data={"+ Loading No.":[x for x in range(1,fetchn_more+1)], "Unit Mass":positive_topx, "Document Mass":[""]*fetchn_more, "Initial Peak Assignment":[""]*fetchn_more, 
                  "Initial Probabilities":[""]*fetchn_more, "Measured Mass":[""]*fetchn_more, "Peak Assignment (Qualified)":[""]*fetchn_more, 
                  "Updated Peak Assignment (from Document Mass)":[""]*fetchn_more, "Updated Document Mass":[""]*fetchn_more})
        negative_loading_table=pd.DataFrame(
            data={"- Loading No.":[x for x in range(1,fetchn_more+1)], "Unit Mass":negative_topx, "Document Mass":[""]*fetchn_more, "Initial Peak Assignment":[""]*fetchn_more, 
                  "Initial Probabilities":[""]*fetchn_more, "Measured Mass":[" "]*fetchn_more, "Peak Assignment (Qualified)":[""]*fetchn_more, 
                  "Updated Peak Assignment (from Document Mass)":[""]*fetchn_more, "Updated Document Mass":[""]*fetchn_more})
        
        # Extract the species classifications (as strings) from each of the top n options, but only if the lists are nonempty to prevent errors
        # top_n_species_doc_masses = np.array([float(sub_list[0][0]) if sub_list[0] else np.nan for sub_list in self.top_n_species_doc])

        # Fill loading tables with Document Masses and their corresponding species assignments + probabilities
        for ind in positive_loading_table.index:
            # Get the top + loadings by unit mass, then find the species corresponding to that unit mass (subtract 1 from unit mass to account 
            # for 0-indexing)
            unit_mass = positive_loading_table.loc[ind, "Unit Mass"]

            positive_loading_table.at[ind, "Document Mass"] = self.top_n_species[unit_mass-1][0]
            positive_loading_table.at[ind, "Initial Peak Assignment"] = self.top_n_species[unit_mass-1][1]
            positive_loading_table.at[ind, "Initial Probabilities"] = self.top_n_species[unit_mass-1][2]

            # There are likely many blank cells in the Measured Mass column. We match the document mass from the current row to the document masses in the 
            # measured_masses DataFrame to see if any measured masses in our database correspond to the current entry, and if not, we leave these cells blank.
            # As a precondition, check first whether there is any assignment at all in the current row to analyze.
            if (positive_loading_table.at[ind, "Document Mass"] and positive_loading_table.at[ind, "Initial Peak Assignment"]):
                cur_doc_mass = positive_loading_table.at[ind, "Document Mass"][0]
                mm_dataframe_first_doc_masses = np.float_([mm[0] for mm in self.measured_masses['document_mass']])
                matching_array = np.isclose(mm_dataframe_first_doc_masses, cur_doc_mass, atol=1e-1)

                if (np.sum(matching_array) >= 1):
                    # Find the index of the assignment that matches the species
                    i = np.argmax(matching_array)
                    
                    # Make sure our measured masses are in the same order as their corresponding peak assignments.
                    assignments_unsorted = positive_loading_table.loc[ind,'Initial Peak Assignment']
                    assignments_sorted = doc_mass.at[unit_mass, 'Assignment'].split(', ')
                    # >>> Boolean flag - if report doesn't exist yet, skip any code related to adding measured masses or qualified peak assignments
                    if report_exists:
                        mm_unsorted = str(self.measured_masses.at[i, 'measured_mass']).split(', ')

                    # Since number of MMs can be shorter than the number of species, we have to be careful to only take a number of elements from unsorted_ordering
                    # equal to the number of MMs.
                    unsorted_ordering = [assignments_sorted.index(species) for species in assignments_unsorted]

                    # >>> Boolean flag - if report doesn't exist yet, skip any code related to adding measured masses or qualified peak assignments
                    if report_exists:
                        if (len(mm_unsorted) == len(unsorted_ordering)):     # If it matches assignment sorting length, use that ordering.
                            mm_sorted = [str(mm_unsorted[j]) for j in unsorted_ordering[:len(mm_unsorted)]]
                        elif (len(mm_unsorted) == 1):                        # We've got only one entry and don't need to do any sorting.
                            mm_sorted = mm_unsorted
                        else:                                                # Otherwise, MMs don't line up with DMs, so give the user an error message.
                            print('***Error! Different number of measured and updated document masses detected. Fix the following measured masses:', mm_unsorted, '***')
                            sys.exit()

                    # Add measured mass and qualified peak assignment to the loading table. Make sure to reformat "Measured Mass" values as a list to adhere to 
                    # conventions for other data in loading table. In addition, note that multiple "Measured Mass" values are separated with both a comma and a space.
                    # >>> Boolean flag - if report doesn't exist yet, skip any code related to adding measured masses or qualified peak assignments
                    if report_exists:
                        positive_loading_table.at[ind, "Measured Mass"] = mm_sorted
                        positive_loading_table.at[ind, "Peak Assignment (Qualified)"] = positive_loading_table.at[ind, "Initial Peak Assignment"]
        positive_loading_table.index = positive_loading_table["Unit Mass"]

        for ind in negative_loading_table.index:
            unit_mass = negative_loading_table.loc[ind, "Unit Mass"]

            negative_loading_table.at[ind, "Document Mass"] = self.top_n_species[unit_mass-1][0]
            negative_loading_table.at[ind, "Initial Peak Assignment"] = self.top_n_species[unit_mass-1][1]
            negative_loading_table.at[ind, "Initial Probabilities"] = self.top_n_species[unit_mass-1][2]

            # There are likely many blank cells in the Measured Mass column. We match the document mass from the current row to the document masses in the 
            # measured_masses DataFrame to see if any measured masses in our database correspond to the current entry, and if not, we leave these cells blank.
            # As a precondition, check first whether there is any assignment at all in the current row to analyze.
            if (negative_loading_table.at[ind, "Document Mass"] and negative_loading_table.at[ind, "Initial Peak Assignment"]):
                cur_doc_mass = negative_loading_table.at[ind, "Document Mass"][0]
                mm_dataframe_first_doc_masses = np.float_([mm[0] for mm in self.measured_masses['document_mass']])
                matching_array = np.isclose(mm_dataframe_first_doc_masses, cur_doc_mass, atol=1e-1)

                if (np.sum(matching_array) >= 1):
                    # Find the index of the assignment that matches the species
                    i = np.argmax(matching_array)

                    # Make sure our measured masses are in the same order as their corresponding peak assignments.
                    assignments_unsorted = negative_loading_table.loc[ind,'Initial Peak Assignment']
                    assignments_sorted = doc_mass.at[unit_mass, 'Assignment'].split(', ')
                    # >>> Boolean flag - if report doesn't exist yet, skip any code related to adding measured masses or qualified peak assignments
                    if report_exists:
                        mm_unsorted = str(self.measured_masses.at[i, 'measured_mass']).split(', ')

                    # Since number of MMs can be shorter than the number of species, we have to be careful to only take a number of elements from unsorted_ordering
                    # equal to the number of MMs.
                    unsorted_ordering = [assignments_sorted.index(species) for species in assignments_unsorted]

                    # >>> Boolean flag - if report doesn't exist yet, skip any code related to adding measured masses or qualified peak assignments
                    if report_exists:
                        if (len(mm_unsorted) == len(unsorted_ordering)):     # If it matches assignment sorting length, use that ordering.
                            mm_sorted = [mm_unsorted[j] for j in unsorted_ordering[:len(mm_unsorted)]]
                        elif (len(mm_unsorted) == 1):                        # We've got only one entry and don't need to do any sorting.
                            mm_sorted = mm_unsorted
                        else:                                                # Otherwise, MMs don't line up with DMs, so give the user an error message.
                            print('***Error! Different number of measured and updated document masses detected. Fix the following measured masses:', mm_unsorted, '***')
                            sys.exit()

                    # Add measured mass and qualified peak assignment to the loading table. Make sure to reformat "Measured Mass" values as a list to adhere to 
                    # conventions for other data in loading table. In addition, note that multiple "Measured Mass" values are separated with both a comma and a space.
                    # >>> Boolean flag - if report doesn't exist yet, skip any code related to adding measured masses or qualified peak assignments
                    if report_exists:
                        negative_loading_table.at[ind, "Measured Mass"] = mm_sorted
                        negative_loading_table.at[ind, "Peak Assignment (Qualified)"] = negative_loading_table.at[ind, "Initial Peak Assignment"]
        negative_loading_table.index = negative_loading_table["Unit Mass"]

        # print(loading_table)

        # ----------------- Detailed description --------------------
        # Get dominant ion categories
        if self.positive_ion:
            signals = self._get_dominant_positive_ions(positive_loading_table, negative_loading_table, loadingTable_pcacomp)
        else:
            signals = self._get_dominant_negative_ions(positive_loading_table, negative_loading_table, loadingTable_pcacomp)
        # print("pca component: {}".format(pcacomp))
        # print("positive_ion: {}".format(positive_ion))
        # print(positive_loading_table.index)
        # print(positive_ion)
        # print(signals)

        # ----------------- Write the report -----------------------
        # Plot page
        self.report.write_plot_page(pcacomp, self.positive_ion, self.fig_scores_single_set[pcacomp-1], self.fig_loading_set[pcacomp-1],
                                    positive_loading_table, negative_loading_table, signals)

        # Table page
        self.report.write_table_page(pcacomp, self.positive_ion, positive_loading_table, negative_loading_table)

        # Analysis page
        self.report.write_analysis_page(pcacomp, self.positive_ion, 
                                        positive_loading_table, negative_loading_table, signals)


    # Add the bar chart of percentage explained variance for each PC
    def add_scree_plot(self):
        self.report.add_scree_plot_page()


    # Create a loadings table from the PCA scores
    def _get_loading_scores(self):
        self.loading_scores = self.pca.components_
        self.loadingTable   = pd.DataFrame(self.loading_scores.T,index=self.mass,
                                           columns=list(range(1, self.ncomp+1)))


    # Save some well-known groups of ions (e.g., hydrocarbons, _-containing organics) so we can succinctly summarize the constituents of
    # the top loadings for the user later
    def _get_dominant_positive_ions(self, p_loading_table, n_loading_table, all_loading_table):
        """Write the dominant positive ions to the report."""
        signals = {}
        # "Hydrocarbons", 
        ion_list = [15, 27, 29, 41, 43, 55, 57] 
        if len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 2:
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass'])
        elif len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 2:
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass'])
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Hydrocarbons"] = {"active":active, "type":type, "top_ion_list":top_ion_list, "ion_list": selected_ion_list}
        
        # "Oxygen-containing organics", 
        ion_list = [31, 19] 
        if 31 in p_loading_table['Unit Mass']:
            active, type = True, '+pca'
            selected_ion_list = [31,19] if all_loading_table.loc[19] > 0 else [31]
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif 31 in n_loading_table['Unit Mass']:
            active, type = True, '-pca'
            selected_ion_list = [31,19] if all_loading_table.loc[19] < 0 else [31]
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Oxygen-containing organics"] = {"active":active, "type":type, "top_ion_list":top_ion_list, "ion_list": selected_ion_list}

        # "Nitrogen-containing organics", 
        ion_list = [30, 44, 70, 86, 18] 
        if len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 1:
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass'])
        elif len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 1:
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass'])
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Nitrogen-containing organics"] = {"active":active, "type":type, "top_ion_list":top_ion_list, "ion_list": selected_ion_list}

        # "Benzene-containing organics", 
        ion_list = [91, 77, 105, 115 ] 
        if 91 in p_loading_table['Unit Mass']:
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif 91 in n_loading_table['Unit Mass']:
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass'])
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Benzene-containing organics"] = {"active":active, "type":type, "top_ion_list":top_ion_list, "ion_list": selected_ion_list} 

        # "PDMS"
        ion_list = [73, 147] 
        if (73 in p_loading_table['Unit Mass']) and (all_loading_table.loc[147]>0):
            active, type = True, '+pca'
            selected_ion_list = [73, 147]
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif len(intersect(ion_list, n_loading_table['Unit Mass'])) == 2:
            active, type = True, '-pca'
            selected_ion_list = [73, 147]
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["PDMS"] = {"active":active, "type":type, "top_ion_list":top_ion_list, "ion_list": selected_ion_list}

        return signals


    def _get_dominant_negative_ions(self, p_loading_table, n_loading_table, all_loading_table):
        """Write the dominant negative ions to the report"""
        signals = {}
        # "Hydrocarbons", 
        ion_list = [12, 13, 24, 25] 
        if len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 2:
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass'])
        elif len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 2:
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass'])
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Hydrocarbons"] = {"active":active, "type":type, 
                                  "top_ion_list": top_ion_list,
                                  "ion_list": selected_ion_list}

        # "Nitrogen-containing organics", 
        ion_list = [26, 42] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 1) and \
           (np.sum(all_loading_table.loc[ion_list] > 0) == 2):
            active, type = True, '+pca'
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
            selected_ion_list = [26,42]
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 1) and \
             (np.sum(all_loading_table.loc[ion_list] < 0) == 2):
            active, type = True, '-pca'
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
            selected_ion_list = [26,42]
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Nitrogen-containing organics"] = {"active":active, "type":type, 
                                                  "top_ion_list": top_ion_list,
                                                  "ion_list": selected_ion_list}

        # "SiOx", 
        ion_list = [60, 61, 76, 77, 136, 137] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 2) and \
           (np.sum(all_loading_table.loc[ion_list] > 0) >= 3):
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 2) and \
             (np.sum(all_loading_table.loc[ion_list] < 0) >= 3):
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["SiOx"] = {"active":active, "type":type, 
                           "top_ion_list": top_ion_list,
                           "ion_list": selected_ion_list}

        # "SOx", 
        ion_list = [64, 80, 96] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 1) and \
           (np.sum(all_loading_table.loc[ion_list] > 0) >= 2):
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 1) and \
             (np.sum(all_loading_table.loc[ion_list] < 0) >= 2):
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["SOx"] = {"active":active, "type":type, 
                          "top_ion_list": top_ion_list,
                          "ion_list": selected_ion_list}
        
        # "POx", 
        ion_list = [63, 79] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 1) and \
           (np.sum(all_loading_table.loc[ion_list] > 0) >= 2):
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 1) and \
             (np.sum(all_loading_table.loc[ion_list] < 0) >= 2):
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["POx"] = {"active":active, "type":type,
                          "top_ion_list": top_ion_list,
                          "ion_list": selected_ion_list}

        # "NOx", 
        ion_list = [46, 62] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 1) and \
           (np.sum(all_loading_table.loc[ion_list] > 0) >= 2):
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 1) and \
             (np.sum(all_loading_table.loc[ion_list] < 0) >= 2):
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["NOx"] = {"active":active, "type":type,
                          "top_ion_list": top_ion_list,
                          "ion_list": selected_ion_list}

        # "Benzene-containing organics"
        ion_list = [49, 36, 73] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) >= 1) and \
           (np.sum(all_loading_table.loc[ion_list] > 0) >= 2):
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) >= 1) and \
             (np.sum(all_loading_table.loc[ion_list] < 0) >= 2):
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Benzene-containing organics"] = {"active":active, "type":type,
                                                 "top_ion_list": top_ion_list,
                                                 "ion_list": selected_ion_list}

        # "Organic acids"
        ion_list = [45] 
        if (len(intersect(ion_list, p_loading_table['Unit Mass'])) == 1):
            active, type = True, '+pca'
            selected_ion_list = [45]
            top_ion_list = [45]
        elif (len(intersect(ion_list, n_loading_table['Unit Mass'])) == 1):
            active, type = True, '-pca'
            selected_ion_list = [45]
            top_ion_list = [45]
        else:
            active, type, top_ion_list, selected_ion_list = False, None, [], []
        signals["Organic acids"] = {"active":active, "type":type,
                                    "top_ion_list": top_ion_list,
                                    "ion_list": selected_ion_list}

        # "Fatty acids"
        ion_list = [255, 281, 283] 
        if check_prominent(ion_list, all_loading_table, positive=True):
            active, type = True, '+pca'
            loadings_sign = all_loading_table.loc[ion_list] > 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, p_loading_table['Unit Mass']) 
        if check_prominent(ion_list, all_loading_table, positive=False):
            active, type = True, '-pca'
            loadings_sign = all_loading_table.loc[ion_list] < 0
            selected_ion_list = loadings_sign.index[loadings_sign].values
            top_ion_list = intersect(ion_list, n_loading_table['Unit Mass']) 
        else:
            active, type, selected_ion_list, top_ion_list = False, None, [], []
        signals["Fatty acids"] = {"active":active, "type":type,
                                  "top_ion_list": top_ion_list,
                                  "ion_list": selected_ion_list}

        return signals

    
    # TODO We are combing over all loading tables and getting user-entered values. Could the user enter 
    #      duplicate updates? If so, how does this code behave? Perhaps we need to have user enter updates in 
    #      a place without possible duplicates.
    # TODO Perhaps add functionality to tack on updated classifications to the end of old species classifications instead of overwriting them?
    # Update the assignment documents using user-entered masses and the peak assignments
    # Parameters:
    #   positive_or_negative_ion - Distinguishes whether to update positive_doc_mass_record.csv (if = 'positive') or negative_doc_mass_record.csv (if = 'negative')
    def update_classifications(self, f_report:str):
        measured_mass = pd.DataFrame(columns=['document_mass', 'measured_mass', 'deviation'])
        
        doc_mass = self.doc_mass
        
        # Try to generate the report from the path the user gave. If the report hasn't been generated yet, tell the user to do so first.
        try:
            report = docx.Document(f_report)
        except:
            print('***Error! No Report to Update! Ensure that PCA report has been generated by running code and selecting \'n\' option first.***')
            sys.exit()

        # Iterate over all tables in document
        for table in report.tables:
            # Iterate over all rows in table
            for row in table.rows:
                # Index 1 is Unit Mass, index 2 is Document Mass, index 5 is Measured Mass, index 6 is qualified peak assignment, index 7 is Updated Peak Assignment, 
                # and index 8 is Updated Document Mass
                cur_header_start = row.cells[0].text
                cur_doc_masses = format_user_input(row.cells[2].text)
                cur_measured_masses = format_user_input(row.cells[5].text)
                cur_updated_peak_assignment = format_user_input(row.cells[7].text)
                cur_updated_doc_masses = format_user_input(row.cells[8].text)

                # Ignore the header row
                if not ('No.' in cur_header_start):
                    # Check user updates for errors before we do anything else. Throw an error to the user if there is one.
                    self.check_update_for_errors(cur_updated_peak_assignment, cur_updated_doc_masses)


                    # Sort the user inputs from least to greatest so that they go into the document in a consistent order. Use the document
                    # mass sort order to sort the measured masses and the updated document mass sort order to sort the peak assignments.
                    doc_masses_unsorted = cur_doc_masses.split(', ')
                    doc_masses_sorted = cur_doc_masses.split(', ')
                    doc_masses_sorted.sort()
                    mm_unsorted = cur_measured_masses.split(', ')
                    updated_doc_masses_unsorted = cur_updated_doc_masses.split(', ')
                    updated_doc_masses_sorted = cur_updated_doc_masses.split(', ')
                    updated_doc_masses_sorted.sort()
                    updated_peak_assignments_unsorted = cur_updated_peak_assignment.split(', ')
                    
                    update_unsorted_ordering = [updated_doc_masses_unsorted.index(mass) for mass in updated_doc_masses_sorted]
                    unsorted_ordering = [doc_masses_unsorted.index(mass) for mass in doc_masses_sorted]

                    # Need to be careful about sorting measured masses since the number of entries might vary. Also account for using updated or old document masses.
                    if (len(mm_unsorted) == len(update_unsorted_ordering)):     # If it matches update length, use the update's ordering.   
                        mm_sorted = [mm_unsorted[i] for i in update_unsorted_ordering[:len(mm_unsorted)]]
                    elif (len(mm_unsorted) == 1):                               # If there's only one entry, we don't need to do any sorting.
                        mm_sorted = mm_unsorted
                    elif (len(mm_unsorted) == len(unsorted_ordering)):          # If there isn't a single element and it matches the original number of doc masses, use that ordering instead.
                        mm_sorted = [mm_unsorted[i] for i in unsorted_ordering[:len(mm_unsorted)]]
                    else:                                                       # Otherwise, MMs don't line up with DMs, so give the user an error message.
                        print('***Error! Different number of measured and updated document masses detected. Fix the following measured masses:', mm_unsorted, '***')
                        sys.exit()
                        
                    updated_peak_assignments_sorted = [updated_peak_assignments_unsorted[i] for i in update_unsorted_ordering[:len(updated_peak_assignments_unsorted)]]

                    cur_doc_masses = ', '.join(doc_masses_sorted)
                    cur_measured_masses = ', '.join(mm_sorted)
                    cur_updated_peak_assignment = ', '.join(updated_peak_assignments_sorted)
                    cur_updated_doc_masses = ', '.join(updated_doc_masses_sorted)


                    # Update the document masses Dataframe if any are found in the two rightmost columns; otherwise, skip this step.
                    if cur_updated_doc_masses or cur_updated_peak_assignment:
                        # In each row of the table, we should act on one of a few cases:
                        # Case 1) The user entered both an updated mass and an updated peak assignment that already exist in the document.
                        #         In this case, we find the corresponding 'Unit Mass' in the document and update its 'Assignment' and 'Document Mass'
                        # Case 2) The user entered both an updated mass and an updated peak assignment that don't exist.
                        #         In this case, just find the corresponding slot in the (ordered) 'Unit Mass' column of the doc. Then, insert a new entry.
                        # Case 3) None of the above cases was satisfied, meaning that the user made an error in their data entry which they must fix, so we throw an error.
                        cur_unit_mass = int(row.cells[1].text.strip())
                        if cur_updated_doc_masses and cur_updated_peak_assignment:
                            if cur_unit_mass in doc_mass.index:
                                doc_mass.at[cur_unit_mass,'Assignment'] = cur_updated_peak_assignment
                                doc_mass.at[cur_unit_mass,'Document Mass'] = cur_updated_doc_masses
                            else:
                                doc_mass.loc[cur_unit_mass] = [cur_unit_mass, cur_updated_peak_assignment, cur_updated_doc_masses]
                        else:
                            print('***Error! Invalid data update. Check each of your table entries to ensure that you entered BOTH an updated peak assignment and updated document mass.***')
                            sys.exit()
                

                    # Get the latest doc masses (i.e., from the update column if there is one; otherwise, just use the ones in the normal doc masses column).
                    latest_doc_masses = cur_updated_doc_masses if cur_updated_doc_masses else cur_doc_masses


                    # Update the measured masses if there is one in this row and we don't already have it in the measured_mass DataFrame. Do this after any updates to the peak
                    # assignments to make sure we get the most up-to-date assignment for the document_mass column in measured_mass.
                    if cur_measured_masses and (not latest_doc_masses in list(measured_mass['document_mass'].values)):
                        mm_size = len(measured_mass.index)

                        # Calculate the deviations between the measured masses and document masses and express them as floats to be stored for later.
                        # TODO Currently cutting off any elements beyond first n (i.e., if one array is longer than the other); may need to fix in future.
                        try:
                            mm_array = np.array(np.float_(cur_measured_masses.split(',')))
                            dm_array = np.array(np.float_(latest_doc_masses.split(',')))
                        except:
                            print("***Error! Encountered row containing a Measured Mass entry (see ", cur_measured_masses, ") but neither a Document Mass nor an", 
                                  "Updated Document Mass entry. Please add one of these entries before trying again.***")
                            sys.exit()
                        # If these arrays have differing lengths, only take the first n elements, where n is the shorter array's length.
                        n = min(len(mm_array), len(dm_array))
                        difference_array = abs(mm_array[:n] - dm_array[:n])

                        measured_mass.loc[mm_size, 'measured_mass'] = cur_measured_masses
                        measured_mass.loc[mm_size, 'document_mass'] = latest_doc_masses
                        difference_list = list(np.round(abs(difference_array), 6))
                        measured_mass.loc[mm_size, 'deviation'] = ','.join(map(str, difference_list))
        
        # TODO When taking split(',')[0], what if there's an entry with multiple possible assignments and the first one isn't close enough to trigger the tolerance?
        #      Increased tolerance to account for this, but be aware of possibly throwing out a valid entry in future.
        # TODO 1) Measured Mass entries that have been deleted in report aren't removed in measured_masses.csv, and 
        #      2) only changing the first instance of a Measured Mass entry will update the .csv file. Are these acceptable?
        # Make sure to filter out any old entries in measured_mass by only keeping those with a document_mass found in the doc_mass DataFrame
        try:
            valid_masses = [float(val.split(',')[0]) for val in doc_mass['Document Mass'].unique()]
            measured_mass = measured_mass[
                measured_mass['document_mass'].apply( 
                    lambda x: any(np.isclose(float(x.split(',')[0]), valid_value, atol=2e-1) for valid_value in valid_masses)
                    )
                ]
            
            # If our dataframe is empty, we must reset it to retain the column headers. Otherwise, measured_masses.csv would be left without the initial row
            # containing the column headers, which would cause errors later on.
            if measured_mass.empty:
                measured_mass = pd.DataFrame(columns=['document_mass', 'measured_mass', 'deviation'])
        except:
            print("***Error! Encountered row containing a Measured Mass entry (see ", cur_measured_masses, ") but neither a Document Mass nor an", 
                                  "Updated Document Mass entry. Please add one of these entries before trying again.***")
            sys.exit()

        # Ensure unit masses are integers and that our Dataframe is sorted before writing it to the given file
        doc_mass['Unit Mass'] = doc_mass['Unit Mass'].astype(int)
        doc_mass.sort_index(inplace=True)
        # print(doc_mass)

        # Write updated document masses to file
        doc_mass.to_csv(self.f_doc_mass, index=False)
        measured_mass.to_csv(self.f_measured_masses, index=False)


    # Makes sure user input from the Updated Peak Assignment (from Document Mass) and Updated Document Mass columns is correctly formatted.
    # Allow for empty entries, which we consider valid.
    # Params:
    #   cur_updated_peak_assignment - str of comma-separated values; each one *should* be a valid species ID
    #   cur_updated_doc_mass - str of comma-separated values; each one *should* be able to be converted to a valid float
    def check_update_for_errors(self, cur_updated_peak_assignment, cur_updated_doc_mass):
        # Empty entries are fine, so we can exit early if this is the case.
        if (len(cur_updated_peak_assignment) == 0) and (len(cur_updated_doc_mass) == 0):
            return
        
        if len(cur_updated_peak_assignment.split(",")) == 1:
            valid_updated_peak_assignment = (is_number(cur_updated_peak_assignment) == False) or not cur_updated_peak_assignment
        else:
            valid_updated_peak_assignment = all((is_number(element) == False) for element in cur_updated_peak_assignment.split(","))

        if len(cur_updated_doc_mass.split(",")) == 1:
            valid_updated_doc_mass = (is_number(cur_updated_doc_mass) == True) or not cur_updated_doc_mass
        else:
            valid_updated_doc_mass = all((is_number(element) == True) for element in cur_updated_doc_mass.split(","))

        if not valid_updated_peak_assignment or not valid_updated_doc_mass:
            print('***Error! Make sure the data you entered in \"Updated Peak Assignment (from Document Mass)\" is a valid species or newline/comma-separated list ' +
                    'of species and that the data you entered in \"Updated Document Mass\" is a real number or newline/comma-separated list of real numbers.***')
            sys.exit()

        if cur_updated_peak_assignment and cur_updated_peak_assignment and ( len(cur_updated_peak_assignment.split(",")) != len(cur_updated_doc_mass.split(",")) ):
            print('***Error! Make sure the data you entered in \"Updated Peak Assignment (from Document Mass)\" and \"Updated Document Mass\" have the same number of entries.***')
            sys.exit()

        for species in cur_updated_peak_assignment.split(","):
            if species[-1].strip() != "+" and species[-1].strip() != "-":
                print('***Error! Ion missing charge! Please ensure each species you entered has at least one + or - sign at the end.***\n')
                print("Species causing error: ", species)
                sys.exit()


# ------------------------------------------------------------------------------ Some useful helper methods ------------------------------------------------------------------------------

# Removes all extra whitespace from updated masses + peak assignments entered by user and replaces newlines with commas to ensure input is processed correctly later
def format_user_input(user_str: str):
    user_str = user_str.strip()
    user_str = re.sub("\n+", ",", user_str)
    user_str = re.sub("\s+", "", user_str)
    user_str = re.sub(",", ", ", user_str)

    return user_str


def intersect(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# Check whether ion_list is prominent in the loading_table_sub
def check_prominent(ion_list, all_loading_table, positive=True):
    if positive:
        loading_table = all_loading_table[all_loading_table>0] 
    else:
        loading_table = all_loading_table[all_loading_table<0] 

    for ion in ion_list:
        if ion not in loading_table.index:
            continue

        iloc = loading_table.index.get_loc(ion)
        iloc_set = [iloc-i for i in range(1,6) if iloc-i >= 0] + [iloc] + [iloc+i for i in range(1,6) if iloc+i <= loading_table.index.size-1]
        loading_table_sub = loading_table.iloc[iloc_set]

        mean = loading_table_sub.mean()
        ion_max = loading_table_sub.idxmax()
        if (ion==ion_max) and (loading_table_sub.loc[ion] > 2*mean):
            return True
    
    return False

# Returns True if s can be converted to a float and False otherwise.
def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False