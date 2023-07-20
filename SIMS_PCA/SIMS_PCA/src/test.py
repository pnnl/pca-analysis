import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import docx

doc = docx.Document('/mnt/c/Users/welc688/OneDrive - PNNL/Documents/pca/SIMS_PCA/SIMS_PCA/output_sample/report.docx')


# Store second round of mass data gathered from mass spectrometry data.
measured_masses = []

for table in doc.tables:
    # Get any masses the user may have entered.
    measured_masses_column = table.columns[5]
    for cell in measured_masses_column.cells:
        # Ignore the header at the top of the column and blank cells.
        if cell.text != 'Measured Mass' and cell.text.strip():
            measured_masses.append(cell.text)

print(measured_masses)


# Now iterate over our masses and make update the peak assignments accordingly
updated_peak_assignments = []

for mass in measured_masses:
    # This is where we need to use our model once it is developed (output )
    updated_peak_assignments.append()

    # Rounding and unit masses
    # Read SIMS data
    # try:
    #     rawdata=pd.read_csv(f_rawsims_data,sep='\t')
    #     rawdata.dropna(inplace=True)
    #     mass_raw = rawdata['Mass (u)'].values
    #     rawdata['Mass (u)']=rawdata['Mass (u)'].apply(np.round).astype(int)
    #     rawdata.set_index(rawdata['Mass (u)'],inplace=True)
    #     mass=rawdata.index
    #     rawdata.drop(columns=['Mass (u)'],inplace=True)

    #