import customtkinter

# Set up basic GUI traits
app = customtkinter.CTk()
app.geometry('1080x720')
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('blue')
app.title('PCA Analysis Hub')
app.columnconfigure(0, weight=1)


# Add entries for the paths to essential files
pcaDir = customtkinter.CTkEntry(app, placeholder_text='Enter PCA directory here', width=400, height=20)
pcaDir.grid(row=0, column=0, padx='50', pady='5', sticky="ew")
raw_data = customtkinter.CTkEntry(app, placeholder_text='Enter raw_data file path here', width=400, height=20)
raw_data.grid(row=1, column=0, padx='50', pady='5', sticky="ew")
report = customtkinter.CTkEntry(app, placeholder_text='Enter report file path here', width=400, height=20)
report.grid(row=2, column=0, padx='50', pady='5', sticky="ew")

# Add buttons to select whether the report is for positive or negative ion data
# TODO Add a label to the left to tell the user what this button does
pos_or_neg = customtkinter.CTkSegmentedButton(app, width=400, height=80)
pos_or_neg.grid(row=3, column=0, padx='10', pady='10')
pos_or_neg.configure(values=['positive', 'negative'])
pos_or_neg.set('positive')

# Add checkboxes for selecting sample group numbers
# TODO

# Add button to pull up metadata file in order to edit it
# TODO

# Finally, add a button to update the document values from the report and another to generate the report from the current database
button_update = customtkinter.CTkButton(app, text='Updated document values database from report')
button_update.grid(row=4, column=0, padx='50', pady='5', sticky="ew")
button_generate = customtkinter.CTkButton(app, text='Generate PCA report from current database')
button_generate.grid(row=5, column=0, padx='50', pady='5', sticky="ew")

# Run the application
app.mainloop()