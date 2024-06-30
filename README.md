---
output:
  html_document: default
  pdf_document: default
  word_document: default
---
# SIMS-PCA

Created by Peishi Jiang, modified by Qian Zhao and Cole Welch

SIMS-PCA is a Python-based library that (1) performs PCA analysis on SIMS experiment data and (2) generates a Word-based report.

### Procedures

**Step 1**: Prepare the python software 

1.0 If on Windows, download the latest Ubuntu version in addition to the Windows Subsystem for Linux (WSL) on your personal or PNNL computer. If you are a Mac user, you can skip to the next step.

1.1. Download and install Python software with version 3.8.2 or newer at `https://www.python.org/downloads/` or by using your preferred package manager between pip and Anaconda.

1.2. Open Terminal to launch Python.

|         On Mac, press `cmd+Space`, then type `Terminal` to open Terminal.
|         On PC, press the `Ctrl+Alt+W` to open Terminal.
        
1.3. Check installed packages.

|        1.3.1 (Optional) Exit python in the Terminal by typing `exit()`. Now, your path should under `computername:~username$`
        
|        1.3.2 type `pip list` to check the list of packages installed. You need to have packages listed in the table near the end of this file.
        
1.4. Install packages (If not already installed).

|        1.4.1 Enter `pip install packagename` into Terminal. If on PNNL network, you NEED to use the proxy option when installing packages via pip or you will get error messages. The command `pip install --proxy=http://proxy01.pnl.gov:3128 packagename` should do the trick.
        
|        1.4.2 If you cannot install with `pip install` command, install package manually. 
        
|                   For example: Google `python-docx` to download this package at `https://pypi.org/project/python-docx/`
              
|                   Manually install this package by typing `python3 -m pip install ./downloads/python-docx-0.8.11.tar.gz` (Mac version) in the 
|                   Terminal or typing `py -m pip install ./downloads/python-docx-0.8.11.tar.gz` (PC version) in the Terminal. 
|                   (Use the correct path of the file)
        
**Step 2**: Prepare SIMS original files 

2.1 Prepare the SIMS raw data files under `SIMS_PCA/SIMS_PCA/sims-data/OriginalData` according to the more detailed instructions found in PCA_Analysis_Manual_rev4.

2.2 Prepare the catalog file `catalog.csv` under `SIMS_PCA/SIMS_PCA/sims-data/Catalog` using either `main.py` or `main_gui.py` according to the more detailed instructions in PCA_Analysis_Manual_rev4.

**Step 3**: Generate SIMS report by executing the code 

3.1 Open `main.py` / `main_gui.py` under `SIMS_PCA/SIMS_PCA/src/` and make sure that the correct file / folder paths in `main.py`, including the following variables: `pcaDir`, `outDir`, `f_rawsims_data`, `f_metadata`, and `positive_or_negative_ion`. The user interface file `main_gui.py` will walk you through these steps.

3.2 If running the command-line file `main.py` without a GUI, change to the correct directory in the Terminal.

|       For example, type `cd Downloads/pca/SIMS_PCA/SIMS_PCA/src` in the Terminal under `computername:~username$`and hit `Enter`.

3.3 Run the `main.py` script by typing `python main.py` in the Terminal and hit `Enter` or run `main_gui.py` instead.

3.3 Get the Word report in the defined `outDir`, which is `output_sample/<insert_report_name>.docx` by default.


### Prerequisites

The code repository has been developed using Python 3.8.10 and the following packages:

|     Packages    | tested with version |
| :-------------: | :-----------------: |
|       pip       |       23.2.1        |
|   scikit-learn  |       1.3.0         |
|   python-docx   |       0.8.11        |
|      pandas     |       2.0.3         |
|      numpy      |       1.24.4        |
|   matplotlib    |       3.7.2         |
|  customtkinter  |       5.2.0         |
|      scipy      |       1.10.1        |



### Repository structure

```
|-- output_sample
|   |-- *.png
|   `-- report.docx
|-- sims-data
|   |-- OriginalData
|   |-- Catalog
|   |-- measured_masses.csv
|   |-- negative_doc_mass_record.csv
|   `-- positive_doc_mass_record.csv
|-- src
|   |-- main.py
|   |-- main_gui.py
|   |-- test_FCN.ipynb
|   |-- test_Gaussian.ipynb
|   |-- file_editor.ipynb
|   |-- train_data.csv
|   |-- pca_sims
|       |-- __init__.py
|   		|-- pca_sims.py
|       |-- plotting.py
|       |-- report.py
|       |-- species_classifier.py
|       `-- __pycache__
```

- `output`: provides the outputs of the code, including multiple PCA plots and a Word report (the name of the repository can be changed; see below)
- `sims-data`: provides the original sims data, document masses, and measured masses to run the code
  - `Catalog`: contains the catalog to keep track of sample group numbers and metadata
  - `OriginalData`: stores the sample sims raw data
- `src`: provides the source code
  - `pca_sims`: stores the source code for PCA analysis, plotting, and report generation.
  - `main.py`: a python module that runs all the programs above in one file
  - `main_gui.py`: a python module that runs all the programs above but encapsulates them within a graphical user interface for ease of use




**Developers**: Peishi Jiang, Cole Welch, Zihua Zhu, Yadong Zhou

**Contact**: cole.welch@pnnl.gov; peishi.jiang@pnnl.gov; zihua.zhu@pnnl.gov