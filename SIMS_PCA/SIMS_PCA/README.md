---
output:
  html_document: default
  pdf_document: default
  word_document: default
---
# SIMS-PCA

Created by Peishi Jiang, modified by Qian Zhao and Cole Welch

SIMS-PCA is a Python-based library that (1) performs PCA analysis on SIMS experiment data; and (2) generate a word-based report.

### Procedures

**Step 1**: Prepare the python software 

1.1. Download and install Python software with version 3.8.2 or newer at `https://www.python.org/downloads/`. 

1.2. Open Terminal to launch Python.

|         In Mac, press `cmd+Space`, then type `Terminal` to open Terminal.
|         In PC, press the `Ctrl+Alt+W` to open Terminal
        
1.3. Check installed packages.

|        1.3.1 (Optional) Exit python in the Terminal by typing `exit()`. Now, your path should under `computername:~username$`
        
|        1.3.2 type `pip list` to check the list of packages installed. You need to have packages listed below in the table.
        
1.4. Install packages (Optional).

|        1.4.1 Type `pip install packagename`
        
|        1.4.2 If you cannot install with `pip install` command, install package manually. 
        
|                   For example: Google `python-docx` to download this package at `https://pypi.org/project/python-docx/`
              
|                   Manually install this package by typing `python3 -m pip install ./downloads/python-docx-0.8.11.tar.gz` (Mac version) in the 
|                   Terminal or typing `py -m pip install ./downloads/python-docx-0.8.11.tar.gz` (PC version) in the Terminal. 
|                   (Use the correct path of the file)
        
**Step 2**: Prepare SIMS original files 

2.1 Prepare the SIMS experiment file `sample_sim.txt` and update old files under `SIMS_PCA/sims-data/OriginalData/`.

2.2 Prepare Metadata file `sample_sims_metadata.txt` with Experiment, Date, Operator, Ion, and Group_Names and update old files under `SIMS_PCA/sims-data/OriginalData/`.

**Step 3**: Generate SIMS report by executing the code 

3.1 Open `main.py` under `SIMS_PCA/src/` and make sure defining the correct file/folder paths in `main.py`, including the following variables: `pcaDir`, `outDir`, `f_rawsims_data`, `f_metadata`, and `positive_or_negative_ion`.

3.2 Change the correct path of `main.py` file in the Terminal. 

|       For example, type `cd Downloads/SIMS_PCA/src/` in the Terminal under `computername:~username$`and hit `Enter`.

3.3 Run `main.py `script by type `python main.py` in the Terminal and hit `Enter`.

3.3 Get the word report in the defined `outDir`, which is `output_sample/report.docx` by default.


### Prerequisites

The code repository has been developed using Python 3.8.2 and the following packages:

|  Packages   | tested with version |
| :---------: | :-----------------: |
|    numpy    |       1.18.4        |
|    scipy    |        1.4.1        |
| matplotlib  |        3.2.1        |
|   sklearn   |       0.23.1        |
|   pandas    |        1.0.3        |
| python-docx |       0.8.11        |



### Repository structure

```
|-- sims-data
|   |-- Original
|   |-- negative_doc_mass_record.csv
|   `-- positive_doc_mass_record.csv
|-- output
|   |-- *.png
|   `-- report.docx
|-- src
|   |-- main.py
|   |-- pca_sims
|       |-- __init__.py
|   		|-- pca_sims.py
|       |-- plotting.py
|       |-- report.py
|       |-- __pycache__
```

- `sims-data`: provides the original sims data, document mass, and group names to run the code
  - `OriginalData`: stores the sample sims experiment data and the metadata of the experiment description.
- `output`: provides the outputs of the code, including multiple PCA plots and a word report (the name of the repository can be changed; see below)
- `src`: provides the source codes
  - `pca_sims`: stores the source code for PCA analysis, plotting, and report generation.
  - `main.py`: a python function that run the codes on one experiment.




**Developers**: Peishi Jiang, Cole Welch, Zihua Zhu, Yadong Zhou

**Contact**: cole.welch@pnnl.gov; peishi.jiang@pnnl.gov; zihua.zhu@pnnl.gov
