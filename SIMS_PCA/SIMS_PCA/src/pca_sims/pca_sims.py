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


positive_ion_category = ["Hydrocarbon", "Oxygen-containing organics", "Nitrogen-containing organics", "Benzene-containing organics", "PDMS"]
negative_ion_category = ["Hydrocarbon", "Nitrogen-containing organics", "SiOx", "SOx", "POx", "NOx", "Benzene-containing organics", "Organic acids", "Fatty acids"]


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
            rawdata=pd.read_csv(f_rawsims_data,sep='\t')
            rawdata.dropna(inplace=True)
            mass_raw = rawdata['Mass (u)'].values
            rawdata['Mass (u)']=rawdata['Mass (u)'].apply(np.round).astype(int)
            rawdata.set_index(rawdata['Mass (u)'],inplace=True)
            mass=rawdata.index
            rawdata.drop(columns=['Mass (u)'],inplace=True)
            # print("rawdata: ", rawdata)
            # print("mass: ", mass)
        except:
            print(traceback.print_exc())
            print('***Error! Cannot Find Correct File!***')
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

        # Save the measured masses for ID later
        self.measured_masses = pd.read_csv('SIMS_PCA/SIMS_PCA/sims-data/measured_masses.csv')

        # Get the unit mass assignments from the chosen .csv document
        self.doc_mass = pd.read_csv(f_doc_mass, index_col=0)

    
    def perform_pca(self):
        """Perform PCA on SIMS data."""
        rawdata = self.rawdata

        print('-------->PCA Processing...')
        try:
            # NORMALIZE DATA
            scaled_data=rawdata.T**0.5
            samplelist=scaled_data.index
            labels=['PC'+str(x) for x in range(1,self.ncomp+1)]

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
    

    # TODO Expose number of top n species in main.py?
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


    def identify_components_from_file(self):
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

        # Do the same process as above to get the measured masses 'document_mass' column into the correct format (i.e., there are a bunch of floats and strings in this column; make sure to put all into lists
        # of strings so species_classifier will process them correctly)
        for i in self.measured_masses.index:
            m_document_mass = str(self.measured_masses.loc[i, 'document_mass'])
            self.measured_masses.loc[i, 'document_mass'] = ''

            m_document_mass = m_document_mass.split(',')
            m_document_mass = [float(mass) for mass in m_document_mass]
            self.measured_masses.at[i, 'document_mass'] = m_document_mass

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print("measured_masses: ", self.measured_masses)

        # TODO Make second two outputs fields of the classifier instance instead of having them show up as separate variables here
        # Assign IDs and probabilities to the PCA data using the components found above
        self.classifier_doc, self.rel_prob_matrix_doc, self.top_n_species_doc = self.classify_species(self.mass_id['raw_mass'], 
                                                                                                      self.mass_id['document_mass'], 
                                                                                                      doc_mass_list, 
                                                                                                      species_list, 5)
        # TODO Remove the uncertainty / top n calculations now that measured_mass isn't used for separate probability calculations?
        # TODO If keeping, deal with problem where fewer than 2 entries in measured_masses causes div by 0 error
        # Do the same for the measured masses; we keep track of them so the user can see the peak assignments and probabilities corresponding to these values on 
        # the next run of PCA analysis
        self.classifier_measured, self.rel_prob_matrix_measured, self.top_n_species_measured = self.classify_species(self.measured_masses['measured_mass'],
                                                                                                                    self.measured_masses['document_mass'], 
                                                                                                                    doc_mass_list, 
                                                                                                                    species_list, 5)


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
        self.report = pca_sims_report(f_report=f_report, ion_sign=ion_sign, description=self.description)

        # Include the overall pairwise PC plots in the report
        self.report.write_2dscore_plots(self.fig_scores_confid_set)

        if self.positive_ion:
            # PCA analysis of positive ToF-SIMS spectra
            for pcacomp in range(1,max_pcacomp+1):
                self.generate_analysis_pcacomp(pcacomp)
        else:
            # PCA analysis of negative ToF-SIMS spectra
            for pcacomp in range(1,max_pcacomp+1):
                self.generate_analysis_pcacomp(pcacomp)
        
        # Save the report
        self.report.save()
    

    def generate_analysis_pcacomp(self, pcacomp:int=1):
        """Generate the analysis for one pca component."""

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

        # TODO Apply Group Numbers.txt filtering to the loading tables (not just the PCA plots)
        positive_loading_table=pd.DataFrame(
            data={"+ Loading No.":[x for x in range(1,fetchn_more+1)], "Unit Mass":positive_topx, "Document Mass":[" "]*fetchn_more, "Initial Peak Assignment":[" "]*fetchn_more, 
                  "Initial Probabilities":[" "]*fetchn_more, "Measured Mass":[" "]*fetchn_more, "Deviation from Document Mass":[" "]*fetchn_more, 
                  "Updated Peak Assignment (from Document Mass)":[" "]*fetchn_more, "Updated Document Mass":[" "]*fetchn_more})
        negative_loading_table=pd.DataFrame(
            data={"- Loading No.":[x for x in range(1,fetchn_more+1)], "Unit Mass":negative_topx, "Document Mass":[" "]*fetchn_more, "Initial Peak Assignment":[" "]*fetchn_more, 
                  "Initial Probabilities":[" "]*fetchn_more, "Measured Mass":[" "]*fetchn_more, "Deviation from Document Mass":[" "]*fetchn_more, 
                  "Updated Peak Assignment (from Document Mass)":[" "]*fetchn_more, "Updated Document Mass":[" "]*fetchn_more})
        
        # TODO Fix masses that are off by too much going to another slot (maybe output error message to user that mass entered is too far off from doc value?)
        # Extract the species classifications (as strings) from each of the top n options, but only if the lists are nonempty to prevent errors
        top_n_species_measured_ids = np.array([sub_list[1][0] if sub_list[1] else '' for sub_list in self.top_n_species_measured])
        n = len(top_n_species_measured_ids)

        # Fill loading tables with Document Masses and their corresponding species assignments + probabilities
        for ind in positive_loading_table.index:
            # Get the top + loadings by unit mass, then find the species corresponding to that unit mass (subtract 1 from unit mass to account 
            # for 0-indexing)
            unit_mass = positive_loading_table.loc[ind, "Unit Mass"]

            positive_loading_table.at[ind, "Document Mass"] = self.top_n_species_doc[unit_mass-1][0]
            positive_loading_table.at[ind, "Initial Peak Assignment"] = self.top_n_species_doc[unit_mass-1][1]
            positive_loading_table.at[ind, "Initial Probabilities"] = self.top_n_species_doc[unit_mass-1][2]

            # TODO For robustness, in future, don't just check first elements of top_n_species_measured_ids and cur_species_repeated - check all of them.
            # There are likely many blank cells in the Measured Mass column. We match the peak assignment to the species in top_n_species_measured to ensure we skip these blank cells
            # As a precondition, check first whether there is any assignment at all in the current row to analyze, then whether any of the top n species match
            # the current peak assignment
            if (positive_loading_table.at[ind, "Document Mass"] and positive_loading_table.at[ind, "Initial Peak Assignment"]):
                cur_species_repeated = np.repeat(positive_loading_table.at[ind, "Initial Peak Assignment"][0], n)
                matching_array = cur_species_repeated == top_n_species_measured_ids
                if (np.sum(matching_array) >= 1):
                    # Find the index of the assignment that matches the species
                    i = np.argmax(matching_array)
                    positive_loading_table.at[ind, "Measured Mass"] = self.measured_masses.at[i, 'measured_mass']
                    positive_loading_table.at[ind, "Deviation from Document Mass"] = self.measured_masses.at[i, 'deviation']
        positive_loading_table.index = positive_loading_table["Unit Mass"]

        for ind in negative_loading_table.index:
            unit_mass = negative_loading_table.loc[ind, "Unit Mass"]

            negative_loading_table.at[ind, "Document Mass"] = self.top_n_species_doc[unit_mass-1][0]
            negative_loading_table.at[ind, "Initial Peak Assignment"] = self.top_n_species_doc[unit_mass-1][1]
            negative_loading_table.at[ind, "Initial Probabilities"] = self.top_n_species_doc[unit_mass-1][2]

            # TODO For robustness, in future, don't just check first elements of top_n_species_measured_ids and cur_species_repeated - check all of them.
            # There are likely many blank cells in the Measured Mass column. We match the peak assignment to the species in top_n_species_measured to ensure we skip these blank cells
            # As a precondition, check first whether there is any assignment at all in the current row to analyze, then whether any of the top n species match
            # the current peak assignment
            if (negative_loading_table.at[ind, "Document Mass"] and negative_loading_table.at[ind, "Initial Peak Assignment"]):
                cur_species_repeated = np.repeat(negative_loading_table.at[ind, "Initial Peak Assignment"][0], n)
                matching_array = cur_species_repeated == top_n_species_measured_ids
                if (np.sum(matching_array) >= 1):
                    # Find the index of the assignment that matches the species
                    i = np.argmax(matching_array)
                    negative_loading_table.at[ind, "Measured Mass"] = self.measured_masses.at[i, 'measured_mass']
                    negative_loading_table.at[ind, "Deviation from Document Mass"] = self.measured_masses.at[i, 'deviation']
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


    def _get_loading_scores(self):
        self.loading_scores = self.pca.components_
        self.loadingTable   = pd.DataFrame(self.loading_scores.T,index=self.mass,
                                           columns=list(range(1, self.ncomp+1)))


    def _get_dominant_positive_ions(self, p_loading_table, n_loading_table, all_loading_table):
        """Write the dominant positive ions to the report"""
        signals = {}
        # "Hydrocarbon", 
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
        signals["Hydrocarbon"] = {"active":active, "type":type, "top_ion_list":top_ion_list, "ion_list": selected_ion_list}
        
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
        # "Hydrocarbon", 
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
        signals["Hydrocarbon"] = {"active":active, "type":type, 
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
    # TODO Implement selectable data files from subset of 000-099
    # Update the assignment documents using user-entered masses and the peak assignments
    # Parameters:
    #   positive_or_negative_ion - Distinguishes whether to update positive_doc_mass_record.csv (if = 'positive') or negative_doc_mass_record.csv (if = 'negative')
    def update_classifications(self, f_doc_mass:str, f_report:str):
        doc_mass = pd.read_csv(f_doc_mass)
        doc_mass.set_index(doc_mass['Unit Mass'], inplace=True)
        measured_mass = pd.DataFrame({'document_mass':[], 'measured_mass':[]})
        report = docx.Document(f_report)

        # Iterate over all tables in document
        for table in report.tables:
            # Iterate over all rows in table
            for row in table.rows:
                # TODO Highlight last two columns differently
                # Index 1 is Unit Mass, index 2 is Document Mass, index 5 is Measured Mass, index 6 is deviation from measured mass, index 7 is Updated Peak Assignment, 
                # and index 8 is Updated Document Mass
                cur_header_start = row.cells[0].text
                cur_doc_mass = format_user_input(row.cells[2].text).split(",")[0]
                cur_measured_mass = format_user_input(row.cells[5].text)
                cur_updated_peak_assignment = format_user_input(row.cells[7].text)
                cur_updated_doc_mass = format_user_input(row.cells[8].text)

                # Update the measured masses if there is one in this row
                if not ('No.' in cur_header_start) and cur_measured_mass and (not cur_measured_mass in measured_mass['measured_mass'].values):
                    mm_size = len(measured_mass.index)
                    measured_mass.loc[mm_size, 'document_mass'] = cur_doc_mass
                    measured_mass.loc[mm_size, 'measured_mass'] = cur_measured_mass
                    measured_mass.loc[mm_size, 'deviation'] = str(np.round(abs(float(cur_measured_mass) - float(cur_doc_mass)), 6))

                # Ignore the header at the top of the column and rows without any updates
                if not ('No.' in cur_header_start) and (cur_updated_doc_mass or cur_updated_peak_assignment):
                    cur_unit_mass = int(row.cells[1].text.strip())
                    # print(cur_unit_mass, cur_doc_mass, cur_updated_doc_mass, cur_updated_peak_assignment)

                    # In each row of the table, we should act on one of a few cases:
                    # Case 1) The user only entered an updated mass for a classification that already exists in the document.
                    #         In this case, we just find the corresponding 'Unit Mass' in the document and update its 'Document Mass'
                    # Case 2) The user entered both an updated mass and an updated peak assignment that already exist in the document.
                    #         In this case, we find the corresponding 'Unit Mass' in the document and update its 'Assignment' and 'Document Mass'
                    # Case 3) The user entered both an updated mass and an updated peak assignment that don't exist.
                    #         In this case, just find the corresponding slot in the (ordered) 'Unit Mass' column of the doc. Then, insert a new entry.
                    # Case 4) None of the above cases was satisfied, meaning that the user made an error in their data entry which they must fix, so we throw an error.
                    if (cur_updated_doc_mass and not cur_updated_peak_assignment and (cur_unit_mass in doc_mass.index)):
                        doc_mass.at[cur_unit_mass,'Document Mass'] = cur_updated_doc_mass
                    elif (cur_updated_doc_mass and cur_updated_peak_assignment):
                        if (cur_unit_mass in doc_mass.index):
                            doc_mass.at[cur_unit_mass,'Assignment'] = cur_updated_peak_assignment
                            doc_mass.at[cur_unit_mass,'Document Mass'] = cur_updated_doc_mass
                        else:
                            doc_mass.loc[cur_unit_mass] = [cur_unit_mass, cur_updated_peak_assignment, cur_updated_doc_mass]
                    else:
                        print('***Error! Invalid data update. Check each of your mass entries to ensure that either 1) you entered only an updated document mass for an entry that already ' +
                              'exists, 2) you entered both an updated document mass and updated peak assignment for an entry that already exists, or 3) you entered both an updated document mass ' +
                              'and peak assignment for an entry that does not yet exist.***')
                        sys.exit()

        # Ensure unit masses are integers and that our Dataframe is sorted before writing it to the given file
        doc_mass['Unit Mass'] = doc_mass['Unit Mass'].astype(int)
        doc_mass.sort_index(inplace=True)
        # print(doc_mass)

        # Write updated document masses to file
        doc_mass.to_csv(f_doc_mass, index=False)
        measured_mass.to_csv('SIMS_PCA/SIMS_PCA/sims-data/measured_masses.csv', index=False)


# ------------------------------------------------------------------------------ Some useful helper methods ------------------------------------------------------------------------------

# Removes all extra whitespace from updated masses + peak assignments entered by user and replaces newlines with commas to ensure input is processed correctly later
def format_user_input(user_str:str):
    user_str = user_str.strip()
    user_str = re.sub("\n+", ",", user_str)
    user_str = re.sub("\s+", "", user_str)
    user_str = re.sub(",", ", ", user_str)

    return user_str


def intersect(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


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

        # Check whether this ion is prominent in the loading_table_sub
        mean = loading_table_sub.mean()
        ion_max = loading_table_sub.idxmax()
        if (ion==ion_max) and (loading_table_sub.loc[ion] > 2*mean):
            return True
    
    return False