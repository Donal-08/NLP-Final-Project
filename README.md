# Pedagogical Ability Assessment of AI Tutors - Group 22

## Group & Project Information

*Group Number:* 22

*Project Title:* Pedagogical Ability Assessment of AI Tutors

*Team Members:*
*   Donal Loitam (AI21BTECH11009)
*   Iragavarapu Sai Pradeep (AI21BTECH11013)
*   Suraj Kumar (AI21BTECH11029)


## Codebase Structure

The codebase is organized within this folder (22-code) as follows:

├── 22-code.ipynb  # Main Jupyter Notebook with implementation

├── experiment_results.csv   

├── assignment_3_ai_tutors_dataset.json 
   


In case you don't want to run the time-consuming 36 experiments present in one of the cells inside, the file "experiment_results.csv" contains the results. You can run all the cells below the experimentation cell, which uses loads the results from the file and does the analysis. 


*   *22-code.ipynb*: The core Jupyter Notebook containing the data loading, preprocessing, model training, evaluation, and analysis steps.
*   *22-README.md*: Provides instructions and information about the codebase.

## Setup Instructions

Follow these steps to set up the environment and run the code:

1.  *Extract the Code:*
    Unzip the 22-code.zip file.

2.  *Navigate to Directory:*
    Open a terminal or command prompt and change to the extracted directory:
    bash
    cd path/to/extracted/22-code
    

3.  *Install Dependencies:*
    You can install the required packages using just by running the **first code cell which contains all the required installations**
    

4.  *Place Dataset:*
    Ensure the MathDial/Bridge dataset is named "assignment_3_ai_tutors_dataset.json" and is in the same file structure as above.
    Modify the file paths within the 22-code.ipynb notebook if your data files have specific names or structures not accounted for.

## Running the Code

The primary way to execute the project code is through the Jupyter Notebook.

1.  *Launch Jupyter:*
    Ensure you are in the project's root directory (path/to/extracted/22-code) in your terminal. Launch JupyterLab (recommended) or Jupyter Notebook:
    bash
    jupyter lab
    
    or
    bash
    jupyter notebook
    

2.  *Open the Notebook:*
    Your web browser should open the Jupyter interface. Navigate to and open 22-code.ipynb.

3.  *Run the Cells:*
    Execute the cells in the notebook sequentially from top to bottom. The notebook contains detailed explanations for each step.
