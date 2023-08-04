"""The class for performing PCA analysis on SIMS data."""

from ast import AsyncFunctionDef
import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .species_classifier import species_classifier
from .plotting import plot_pca_result
from .report import pca_sims_report
import time


positive_ion_category = ["Hydrocarbon", "Oxygen-containing organics", "Nitrogen-containing organics", "Benzene-containing organics", "PDMS"]
negative_ion_category = ["Hydrocarbon", "Nitrogen-containing organics", "SiOx", "SOx", "POx", "NOx", "Benzene-containing organics", "Organic acids", "Fatty acids"]


class pca_sims(object):

    def __init__(
        self,
        f_rawsims_data: str,
        f_metadata: str,
        pcaDir: str,
        outDir: str,
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
            # TODO Remove
            print("rawdata: ", rawdata)
            print("mass: ", mass)
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
        description['ion'] = metadata_df.loc['Ion',1]

        # Extract the sample names (e.g., Goethite-Tannic Acid 1400 ppm) from the metadata file. Note that we
        # must exclude the first 4 lines since they include other information.
        sample_description_set = []
        n_samples = metadata_df.shape[0] - 4
        for i in range(n_samples):
            sample_number      = metadata_df.index[i+4]
            sample_description = metadata_df.loc[sample_number,1]
            sample_description_set.append([int(sample_number), str(sample_description)])
        
        nmass, ncomp = rawdata.shape

        self.f_rawsims_data         = f_rawsims_data
        self.f_metadata             = f_metadata
        # self.f_group_name   = f_group_name
        self.pcaDir                 = pcaDir
        self.outDir                 = outDir
        self.description            = description
        self.sample_description_set = sample_description_set
        self.rawdata                = rawdata
        self.mass                   = mass
        self.mass_raw               = mass_raw
        self.nmass                  = nmass
        self.ncomp                  = ncomp

        # Generate the output folder if it does not exist
        if not os.path.exists(os.path.join(pcaDir, outDir)):
            os.makedirs(os.path.join(pcaDir, outDir))

        # Initialize the mass identification
        self.positive_mass_id = pd.DataFrame(columns=['raw_mass', 'document_mass', 'true_assignment', 'possible_assignment'], index=mass)
        self.negative_mass_id = pd.DataFrame(columns=['raw_mass', 'document_mass', 'true_assignment', 'possible_assignment'], index=mass)
        self.positive_mass_id['raw_mass'] = mass_raw
        self.negative_mass_id['raw_mass'] = mass_raw

        # TODO Change formatting so we can move over from obsolete train_data to positive/negative_doc_mass_record.csv
        # TODO Add flexibility for either pos / neg ions (may need to move this code down to where we can use mass_id; plus, we we would
        #      then be able to use this to calculate the uncertainty in the classifier from the actual data)
        # Initialize the classifier instance; we will pass this the raw_masses and, for each of them, get the corresponding probabilities of it being each of the species in the doc_mass
        self.classifier = species_classifier('SIMS_PCA/SIMS_PCA/src/train_data.csv', self.positive_mass_id)
        # Get the relative probabilities with number of rows = number of test masses (e.g., 800) and number of columns = number of reference masses (e.g., 48)
        self.rel_prob_matrix = self.classifier.calculate_probabilities()
        # Save the top 3 potential candidates for each list.  We'll add them to the report later.
        self.top_n_species = self.classifier.identify_top_n_species(3)

    
    # def perform_pca(self, max_pcacomp:int=5):
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
    
    def identify_components_from_file(self, f:str, positive_ion=True):
        """Identify chemical components from an existing file."""
        print('-------->Finding assigned unit mass from file: {}'.format(f))
        # Get the assigned unit mass
        doc_mass = pd.read_csv(f, index_col=0)

        if positive_ion:
            mass_id = self.positive_mass_id
            self.positive_ion = True
        else:
            mass_id = self.negative_mass_id
            self.positive_ion = False

        for unit_mass in mass_id.index:
            if unit_mass in doc_mass.index:
                assignment    = doc_mass.loc[unit_mass, 'Assignment'].split(',')
                assignment    = [assign.strip() for assign in assignment]
                document_mass = doc_mass.loc[unit_mass, 'Document Mass'].split(',') 
                document_mass = [float(mass) for mass in document_mass]
                # if np.isnan(self.mass_id.loc[unit_mass, 'possible_assignment']):
                if not isinstance(mass_id.loc[unit_mass, 'possible_assignment'], list):
                    mass_id.at[unit_mass, 'possible_assignment'] = assignment
                    mass_id.at[unit_mass, 'document_mass']       = document_mass
                else:
                    mass_id.at[unit_mass, 'possible_assignment'] += assignment
                    mass_id.at[unit_mass, 'document_mass']       += document_mass 
                print('Identified unique mass {} from the documentation with Document Mass {} and assignment {}'.format(
                    unit_mass, document_mass, assignment))
        
        # TODO Remove
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print("mass_id: ", mass_id)

        if positive_ion:
            self.positive_mass_id = mass_id
        else:
            self.negative_mass_id = mass_id
        # TODO Remove
        print("positive_mass_id: ", self.positive_mass_id)
        print("negative_mass_id: ", self.negative_mass_id)

    # TODO This method is pretty much empty and should be removed.
    def perform_rule_based_analysis(self):
        """Perform rule-based analysis to identify chemical components."""
        print('-------->Rule based analysis...')

    def plot_pca_result(
        self, 
        max_pcacomp:int=5
    ):
        """Plot PCA analysis result."""
        pca_maxpcacomp_df, fig_screeplot, fig_scores_set, fig_scores_confid_set, fig_scores_single_set, fig_loading_set = \
                plot_pca_result(self.pca, self.pca_data, self.samplelist, self.mass, 
                                # self.pcaDir, self.outDir, self.f_group_name, max_pcacomp)
                                self.sample_description_set, self.pcaDir, self.outDir, max_pcacomp)
        self.pca_maxpcacomp_df     = pca_maxpcacomp_df
        self.fig_screeplot         = fig_screeplot
        self.fig_scores_set        = fig_scores_set
        self.fig_scores_confid_set = fig_scores_confid_set
        self.fig_scores_single_set = fig_scores_single_set
        self.fig_loading_set       = fig_loading_set
    
    def generate_report(self, f_report='report.docx', max_pcacomp:int=5):
        """Generate the report"""
        print('-------->Generating the report now...')

        # Initialize the report
        f_report = os.path.join(self.pcaDir, self.outDir, f_report)
        self.report = pca_sims_report(f_report=f_report, description=self.description)

        # Include the overall pairwise PC plots in the report
        self.report.write_2dscore_plots(self.fig_scores_confid_set)

        if self.positive_ion:
            # PCA analysis of positive ToF-SIMS spectra
            for pcacomp in range(1,max_pcacomp+1):
                self.generate_analysis_pcacomp(pcacomp,positive_ion=True)

        else:
            # PCA analysis of negative ToF-SIMS spectra
            for pcacomp in range(1,max_pcacomp+1):
                self.generate_analysis_pcacomp(pcacomp,positive_ion=False)
        
        # Save the report
        self.report.save()
    
    def generate_analysis_pcacomp(self, pcacomp:int=1, positive_ion:bool=True):
        """Generate the analysis for one pca component"""
        # Get the mass_id
        if positive_ion:
            mass_id = self.positive_mass_id
        else:
            mass_id = self.negative_mass_id

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
            data={"+ loading":[" "]*fetchn_more, "No. #":[x for x in range(1,fetchn_more+1)],
                  "Unit Mass":positive_topx, "Document Mass":[" "]*fetchn_more, "Initial Peak Assignment":[" "]*fetchn_more, "Initial Probabilities":[" "]*fetchn_more,
                  "Measured Mass":[" "]*fetchn_more, "Updated Peak Assignment":[" "]*fetchn_more, "Updated Probabilities":[" "]*fetchn_more,})
        negative_loading_table=pd.DataFrame(
            data={"- loading":[" "]*fetchn_more, "No. #":[x for x in range(1,fetchn_more+1)],
                  "Unit Mass":negative_topx, "Document Mass":[" "]*fetchn_more, "Initial Peak Assignment":[" "]*fetchn_more, "Initial Probabilities":[" "]*fetchn_more,
                  "Measured Mass":[" "]*fetchn_more, "Updated Peak Assignment":[" "]*fetchn_more, "Updated Probabilities":[" "]*fetchn_more,})
        
        # TODO This is where we need to change the tables that get put into the document; need to edit 'Document Mass' and 'Initial Peak Assignment' columns to look like:
        #           6.0146, 2.0152, 1.0073, 6Li+, H2+, H+, 4.20e-07, 0.239, 0.761
        # Fill loading tables with Document Masses and their corresponding species assignments + probabilities
        for ind in positive_loading_table.index:
            # Get the top + loadings by unit mass, then find the species corresponding to that unit mass (subtract 1 from unit mass to account 
            # for 0-indexing)
            unit_mass = positive_loading_table.loc[ind, "Unit Mass"]
            # positive_loading_table.loc[ind, "Accurate Mass"] = mass_id.loc[unit_mass, 'raw_mass']
            positive_loading_table.at[ind, "Document Mass"] = self.top_n_species[unit_mass-1][0]
            positive_loading_table.at[ind, "Initial Peak Assignment"] = self.top_n_species[unit_mass-1][1]
            positive_loading_table.at[ind, "Initial Probabilities"] = self.top_n_species[unit_mass-1][2]
        positive_loading_table.index = positive_loading_table["Unit Mass"]

        for ind in negative_loading_table.index:
            unit_mass = negative_loading_table.loc[ind, "Unit Mass"]
            # negative_loading_table.at[ind, "Accurate Mass"] = mass_id.loc[unit_mass, 'raw_mass']
            negative_loading_table.at[ind, "Document Mass"] = self.top_n_species[unit_mass-1][0]
            negative_loading_table.at[ind, "Initial Peak Assignment"] = self.top_n_species[unit_mass-1][1]
            negative_loading_table.at[ind, "Initial Probabilities"] = self.top_n_species[unit_mass-1][2]
        negative_loading_table.index = negative_loading_table["Unit Mass"]
        
        med=pd.DataFrame(data={"+ loading":["- loading"],"No. #":["No. #"],"Unit Mass":["Unit Mass"],"Document Mass":["Document Mass"],
                               "Initial Peak Assignment":["Initial Peak Assignment"], "Initial Probabilities":["Initial Probabilities"], 
                               "Measured Mass":["Measured Mass"], "Updated Peak assignment":["Updated Peak assignment"], "Updated Probabilities":["Updated Probabilities"],})

        loading_table = pd.concat([positive_loading_table, med, negative_loading_table])

        # print(loading_table)

        # ----------------- Detailed description --------------------
        # Get dominant ion categories
        if positive_ion:
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
        self.report.write_plot_page(pcacomp, positive_ion, self.fig_scores_single_set[pcacomp-1], self.fig_loading_set[pcacomp-1],
                                    positive_loading_table, negative_loading_table, signals)

        # Table page
        self.report.write_table_page(pcacomp, positive_ion, positive_loading_table, negative_loading_table)

        # Analysis page
        self.report.write_analysis_page(pcacomp, positive_ion, 
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

    
    # TODO Add code from test.py here once it has been tested out for a few lines using the model.
    def update_classifications(self):
        """Update the masses and the peak assigntments using data the user has entered into the"""
        measured_masses = []


# Some useful helper methods included as part of this module.
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