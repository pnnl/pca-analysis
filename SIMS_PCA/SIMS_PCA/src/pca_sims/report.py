"""This class is used for generating the report."""

import re
from typing import Dict, Any, Optional

from pty import STDERR_FILENO
from docx import Document
from docx.shared import Inches
from docx.oxml import OxmlElement, ns
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, Cm

import pandas as pd

class pca_sims_report(object):

    def __init__(self, f_report:str, ion_sign, description:Optional[Dict]=None):
        self.document = Document()
        self.f_report = f_report
        self.description = description

        # Set the page margins
        sections = self.document.sections
        for section in sections:
            section.top_margin = Inches(0.625)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(0.625)
            section.right_margin = Inches(0.625)

        # Title page
        self._create_title_page(ion_sign)
    
    
    def _create_title_page(self, ion_sign:str):
        document = self.document
        document.add_heading("\t\tPCA-SIMS Spectra Analysis Report", 0)

        # Add author and time
        description = self.description
        if description is not None:
            run=document.add_paragraph().add_run()
            run.add_break(); run.add_break();run.add_break()

            # p=document.add_paragraph("{}".format(description['experiment']))
            p=document.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("{}".format(description['experiment']) + " (" + ion_sign + " ions)")
            run.bold = True
            run.font.size = Pt(18)
            run.add_break(); run.add_break();run.add_break()
            run.add_break(); run.add_break();run.add_break()

            # p=document.add_paragraph('ToF-SIMS testing date: {}'.format(description['date']))
            p=document.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run('ToF-SIMS testing date: {}'.format(description['date']))
            run.font.size = Pt(14)

            # p=document.add_paragraph('ToF-SIMS operator: {}'.format(description['operator']))
            p=document.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run('ToF-SIMS operator: {}'.format(description['operator']))
            run.font.size = Pt(14)
    

    def write_2dscore_plots(self, score2d_plots):
        document = self.document

        # Start a new page
        document.add_page_break()

        # Write the title
        document.add_heading('2D PCA scores plots')

        # Write the score plots
        for plots in score2d_plots:
            for plot in plots:
                document.add_picture(plot, width=Inches(6))  # (6" Score plots + 0.625" margin + 0.625" margin = 7.5" total)


    def write_plot_page(self, pcacomp:int, positive_ion:bool, score_plot:str, loading_plot:str,
                        positive_loading_table:pd.DataFrame, negative_loading_table:pd.DataFrame, 
                        ion_signals: Dict):
        sign_ion = 'Positive' if positive_ion else 'Negative'

        document = self.document
        # Start a new page
        document.add_page_break()

        # Write the title
        document.add_heading('{} ion spectra, PCA analysis results -- PC{}'.format(sign_ion, pcacomp), 1)

        # Write the score plots
        document.add_picture(score_plot, width=Inches(5))  # Score plot
        document.add_picture(loading_plot, width=Inches(5))  # Loading plot

        # Write the top 20 positive loading assignments
        document.add_paragraph("High score samples contain more:")
        p = document.add_paragraph("", style="List Bullet")
        for index in positive_loading_table.index:
            unit_mass, assign = positive_loading_table.loc[index, "Unit Mass"], positive_loading_table.loc[index, "Initial Peak Assignment"]
            # TODO Remove trailing comma + space (, ) after last set of assignments here for polish
            # TODO Create a helper method to encapsulate this boilerplate (it's repeated 6 times)
            if str(assign) == 'nan':
                p.add_run("m/z {} (-), ".format(unit_mass))
            else:
                p.add_run("m/z {} (".format(unit_mass))
                document_add_assignment(p, assign)
                p.add_run("), ")
        positive_dominant_ions = [ion_type for ion_type in ion_signals if ion_signals[ion_type]['active'] and (ion_signals[ion_type]['type']=='+pca')] 
        document.add_paragraph(", ".join(positive_dominant_ions), style="List Bullet") # ion categories

        # Write the top 20 negative loading assignments
        document.add_paragraph("Low score samples contain more:")
        p = document.add_paragraph("", style="List Bullet")
        for index in negative_loading_table.index:
            unit_mass, assign = negative_loading_table.loc[index, "Unit Mass"], negative_loading_table.loc[index, "Initial Peak Assignment"]
            if str(assign) == 'nan':
                p.add_run("m/z {} (-), ".format(unit_mass))
            else:
                p.add_run("m/z {} (".format(unit_mass))
                document_add_assignment(p, assign)
                p.add_run("), ")
        negative_dominant_ions = [ion_type for ion_type in ion_signals if ion_signals[ion_type]['active'] and (ion_signals[ion_type]['type']=='-pca')] 
        document.add_paragraph(", ".join(negative_dominant_ions), style="List Bullet") # ion categories


    # Writes a PCA loading table to the document. Columns are:
    # +/- loading | No. # | Unit Mass | Document Mass | Initial Peak Assignment | Measured Mass | Updated Peak Assignment
    def write_table_page(self, pcacomp:int, positive_ion:bool,
                         p_loading_table:pd.DataFrame, n_loading_table:pd.DataFrame, 
     ):
        document = self.document
        sign_ion = 'Positive' if positive_ion else 'Negative'

        # Start a new page
        document.add_page_break()

        # Write the title
        document.add_heading('{} ion spectra, top positive loadings -- PC{}'.format(sign_ion, pcacomp), 1)

        # Write positive loading values ...
        document_add_table(document, p_loading_table)
        
        # Start a new page
        document.add_page_break()

        # Write the title
        document.add_heading('{} ion spectra, top negative loadings -- PC{}'.format(sign_ion, pcacomp), 1)

        # Write negative loading values ...
        document_add_table(document, n_loading_table)


    def write_analysis_page(self, pcacomp:int, positive_ion:bool, 
                            positive_loading_table:pd.DataFrame, negative_loading_table:pd.DataFrame, 
                            ion_signals: Dict):
        document = self.document
        sign_ion = 'Positive' if positive_ion else 'Negative'

        # Start a new page
        document.add_page_break()

        # Write the title
        document.add_heading('{} ion spectra, molecular information from PC{} loadings plot'.format(sign_ion, pcacomp), 1)

        # Write the overview description
        p = document.add_paragraph("""The major positive PC{0} loadings are """.format(pcacomp), style="List Bullet")
        for index in positive_loading_table.index:
            unit_mass, assign = positive_loading_table.loc[index, "Unit Mass"], positive_loading_table.loc[index, "Initial Peak Assignment"]
            if str(assign) == 'nan':
                p.add_run("m/z {} (-), ".format(unit_mass))
            else:
                p.add_run("m/z {} (".format(unit_mass))
                document_add_assignment(p, assign)
                p.add_run("), ")
        p.add_run(", indicating they are more observed in high PC{0} score samples.".format(pcacomp))

        p = document.add_paragraph("""The major negative PC{0} loadings are """.format(pcacomp), style="List Bullet")
        for index in negative_loading_table.index:
            unit_mass, assign = negative_loading_table.loc[index, "Unit Mass"], negative_loading_table.loc[index, "Initial Peak Assignment"]
            if str(assign) == 'nan':
                p.add_run("m/z {} (-), ".format(unit_mass))
            else:
                p.add_run("m/z {} (".format(unit_mass))
                document_add_assignment(p, assign)
                p.add_run("), ")
        p.add_run(", indicating they are more observed in high PC{0} score samples.".format(pcacomp))

        # Write the dominant ion categories
        positive_dominant_ions = [ion_type for ion_type in ion_signals if ion_signals[ion_type]['active'] and (ion_signals[ion_type]['type']=='+pca')] 
        negative_dominant_ions = [ion_type for ion_type in ion_signals if ion_signals[ion_type]['active'] and (ion_signals[ion_type]['type']=='-pca')] 
        for ion_type in positive_dominant_ions:
            dominant_mass = ion_signals[ion_type]['top_ion_list']
            
            p = document.add_paragraph("{} signals, such as ".format(ion_type), style="List Bullet")
            for unit_mass in dominant_mass:
                assign = positive_loading_table.loc[unit_mass, "Initial Peak Assignment"]
                if str(assign) == 'nan':
                    p.add_run("m/z {} (-), ".format(unit_mass))
                else:
                    p.add_run("m/z {} (".format(unit_mass))
                    document_add_assignment(p, assign)
                    p.add_run("), ")

            p.add_run(", are mostly found in positive loadings, indicating that high PC{} score samples contain more {}.".format(pcacomp, ion_type))
        
        for ion_type in negative_dominant_ions:
            dominant_mass = ion_signals[ion_type]['top_ion_list']
            
            p = document.add_paragraph("{} signals, such as ".format(ion_type), style="List Bullet")
            # print(ion_type, ion_signals[ion_type])
            for unit_mass in dominant_mass:
                # print(unit_mass)
                # print(negative_loading_table.index)
                assign = negative_loading_table.loc[unit_mass, "Initial Peak Assignment"]
                if str(assign) == 'nan':
                    p.add_run("m/z {} (-), ".format(unit_mass))
                else:
                    p.add_run("m/z {} (".format(unit_mass))
                    document_add_assignment(p, assign)
                    p.add_run("), ")

            p.add_run(", are mostly found in negative loadings, indicating that high PC{} score samples contain more {}.".format(pcacomp, ion_type))
    
    def save(self):
        # TODO: Generate table of contents

        # Add page numbers
        self.document.sections[0].footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        document_add_page_number(self.document.sections[0].footer.paragraphs[0].add_run())

        # Save the report
        self.document.save(self.f_report)


# ------------------------------------------------------------------------------ Some useful helper methods ------------------------------------------------------------------------------

# Add a table to the end and create a reference variable along with an extra header row
def document_add_table(document:Document, df:pd.DataFrame):
    # Create number of rows + columns corresponding to the dataframe's size
    t = document.add_table(df.shape[0]+1, df.shape[1])
    t.style = 'Table Grid'

    # Add the headers at the top of each column
    for j in range(df.shape[-1]):
        t.cell(0,j).text = df.columns[j]

    # Add the rest of the dataframe. Start at index 1 (see "i+1") to avoid conflicts with the header row, and handle 4 separate cases:
    # 1) Value is NaN - Add blank text to the cell
    # 2) Value is a list of non-floats - parse each string in the list (see document_add_assignment) and add it to a paragraph that is then added to the cell
    # 3) Value is a list of floats - convert each list entry to a string and add it to the cell
    # 4) Value is any non-list entity - convert it to a string and add it to the cell
    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            cur_entry = df.values[i,j]
            # We expect each group to be either a nested list of lists or a single-depth list, which we handle with the if case here; if list is empty, also default to the 1st case
            if not cur_entry or not isinstance(cur_entry, list) or not isinstance(cur_entry[0], list):
                group_size = 1
            else:
                group_size = len(cur_entry)

            # Iterate over multiple sublists of the current dataframe entry, but only if it has a nested list structure
            for k in range(group_size):
                if not cur_entry or not isinstance(cur_entry, list) or not isinstance(cur_entry[0], list):
                    cur_group = cur_entry
                else:
                    cur_group = cur_entry[k]

                if str(cur_group) == 'nan':
                    t.cell(i+1,j).text = ''
                elif isinstance(cur_group, list) and (not cur_entry or not is_float(cur_group[0])):
                    p_species = t.cell(i+1,j).add_paragraph()
                    document_add_assignment(p_species, cur_group, in_table=True)
                elif isinstance(cur_group, list) and (not cur_entry or is_float(cur_group[0])):
                    p_probs = t.cell(i+1,j).add_paragraph()
                    p_probs.add_run('\n'.join(map(str, cur_group)))
                else:
                    t.cell(i+1,j).text = str(cur_group)


# TODO Superscripting is done wrong on - sign after a number and for ? marks (for example: see SNO2-?)
# Parse strings representing the species assignments that we intend to add to a cell.
#   Parameters:
#   p - (docx paragraph) The paragraph to which we add text
#   assignment - The list of items to add to the paragraph
#   in_table - True means we use new line separators to add text to a table, False means we use comma separators to add text elsewhere
def document_add_assignment(p, assignment:list, in_table:bool=False) -> None:
    n_assign = len(assignment)
    # For each chemical species assignment (expect assign to be a string)
    for j,assign in enumerate(assignment):
        # Pull off sign out front, then split the remaining string on any digits in the chemical formula (e.g., 'C3H6' -> ['C', '3', 'H', '6'])
        assign_, signs = re.split('([+-]+)', assign)[0:2]
        assign_ = re.split('(\d+)', assign_)
        assign_ = [s for s in assign_ if s != '']
        # Add number and letter
        for i,s in enumerate(assign_):
            if i==0 and is_float(s):
                text = p.add_run(s)
                text.font.superscript = True
            elif i!=0 and is_float(s):
                text = p.add_run(s)
                text.font.subscript = True
            else:
                text = p.add_run(s)
        # Add any sign(s) at the end
        text = p.add_run(signs)
        text.font.superscript = True
        # If not the last element, add a separator appropriate to the context (table or not in table)
        if j != n_assign-1 and in_table:
            p.add_run('\n')
        elif j != n_assign-1:
            p.add_run(', ')
        

def create_element(name):
    return OxmlElement(name)


def create_attribute(element, name, value):
    element.set(ns.qn(name), value)


def document_add_page_number(run):
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')

    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "PAGE"

    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'end')

    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)


def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False