import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import docx

doc = pd.read_csv('SIMS_PCA/SIMS_PCA/sims-data/negative_doc_mass_record.csv')
report = docx.Document('SIMS_PCA/SIMS_PCA/output_sample/report.docx')

# Set the indices of the document DataFrame to the Unit Masses for ease of access
doc.set_index(doc['Unit Mass'], inplace=True)

# TODO Maybe store all data in one DataFrame?
# TODO We are combing over all loading tables and getting user-entered values. Could the user enter 
# duplicate updates? If so, how does this code behave? Perhaps we need to have user enter updates in 
# a place without possible duplicates.

# Iterate over all tables in document
for table in report.tables:
    # Iterate over all rows in table
    for row in table.rows:
        # Ignore the header at the top of the column and rows without any updates
        # Index 2 is Unit Mass, index 6 is Measured Mass, and index 7 is Updated Peak Assignment
        if not ('loading' in row.cells[0].text) and (row.cells[6].text.strip() or row.cells[7].text.strip()):
            cur_unit_mass = row.cells[2].text.strip()
            cur_measured_mass = row.cells[6].text.strip()
            cur_updated_peak_assignment = row.cells[7].text.strip()
            
            # print(cur_unit_mass, cur_measured_mass, cur_updated_peak_assignment)

            # In each row of the table, we should act on one of a few cases:
            # Case 1) The user only entered a new mass. 
            #         In this case, we just find the corresponding 'Unit Mass' in the document and update its 'Document Mass'
            # Case 2) The user entered both a new mass and a new classification that already exist in the document.
            #         In this case, we find the corresponding 'Unit Mass' in the document and update its 'Assignment' and 'Document Mass'
            # Case 3) The user entered both a new mass and a new classification that don't exist.
            #         In this case, just find the corresponding slot in the (ordered?) 'Unit Mass' column of the doc. Then, insert a new entry.
            # Shouldn't ever have to write to 'Measured Mass' or 'Updated Peak Assignment' columns
            if (cur_measured_mass and not cur_updated_peak_assignment):
                doc.at[cur_unit_mass,'Document Mass'] = cur_measured_mass   # TODO Why is this just adding a new row?
            elif (cur_measured_mass and cur_updated_peak_assignment):
                if (True):
                    pass
                elif (False):
                    pass

print(doc)