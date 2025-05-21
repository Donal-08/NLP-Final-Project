# Pedagogical Ability Assessment of AI Tutors - Group 22
This project aims to evaluate the pedagogical effectiveness of AI tutors by building classification models to analyze tutor responses based on four specific criteria. The project uses the MathDial/Bridge dataset.
## Group & Project Information

*Group Number:* 22
*Team Members:*
*   Donal Loitam (AI21BTECH11009)
*   Iragavarapu Sai Pradeep (AI21BTECH11013)
*   Suraj Kumar (AI21BTECH11029)



## Project Overview

The core task is to classify tutor responses to student mistakes across four pedagogical dimensions:
1.  **Mistake Identification:** Does the tutor recognize the student's mistake? (Labels: Yes / To some extent / No)
2.  **Mistake Location:** Does the tutor pinpoint where the mistake is? (Labels: Yes / To some extent / No)
3.  **Pedagogical Guidance:** Does the tutor offer helpful and correct explanations or hints? (Labels: Yes / To some extent / No)
4.  **Actionability:** Does the tutor clearly state what the student should do next? (Labels: Yes / To some extent / No)

The models are evaluated using Accuracy and Macro F1 scores, calculated for both exact label matching and a lenient version (grouping "Yes" and "To some extent").

## Dataset

* **Source:** `assignment_3_ai_tutors_dataset.json` (MathDial/Bridge dataset)
* **Format:** JSON, containing student-tutor dialogues with annotations for the four tasks.
* **Key Features Used:** `conversation_history`, `response` text, and the annotation labels.

## Methodology

The project explores various configurations of preprocessing, model architectures, task strategies, and loss functions. The primary implementation is within the `NLP_assignment3.ipynb` Jupyter Notebook.

### 1. Data Loading and EDA
* The `load_data` function parses the JSON dataset into a Pandas DataFrame.
* Exploratory Data Analysis (EDA) was performed to understand label distributions and text lengths, informing subsequent choices (e.g., `MAX_LENGTH=256`, handling class imbalance).

### 2. Preprocessing
* **Text Cleaning (`clean_text`):** Includes options for lowercasing, punctuation removal, and stopword removal (using NLTK).
* **Input Formulation:** Experiments were run with:
    * Tutor `response` only (`include_history=False`).
    * `conversation_history` + `[SEP]` + `response` (`include_history=True`).
* **Tokenization (`TutorResponseDataset`):** Utilizes `AutoTokenizer` from Hugging Face for models like `distilbert-base-uncased`, `bert-base-uncased`, and `roberta-base`. Text is padded/truncated to `MAX_LENGTH=256`.
* **Data Splitting (`create_dataloaders`):** Data is split into training (80%), validation (10%), and test (10%) sets, with stratification based on the 'Mistake_Identification' label.

### 3. Model Architectures
* **Base Models:** Pre-trained Transformers from Hugging Face:
    * `distilbert-base-uncased`
    * `bert-base-uncased`
    * `roberta-base`
* **Task Strategies:**
    * **Single-Task (`create_single_task_model`):** Four independent `AutoModelForSequenceClassification` models, one for each task.
    * **Multi-Task (`MultiTaskModel` class):** A single model with a shared base and four separate classification heads.

### 4. Training and Fine-Tuning (`train_and_evaluate`)
* **Optimizer:** AdamW with a learning rate of `2e-5`.
* **Loss Functions:**
    * Standard Cross-Entropy Loss (`nn.CrossEntropyLoss`).
    * Weighted Cross-Entropy Loss (weights calculated based on class distribution).
    * Custom `FocalLoss` (gamma=2.0) to address class imbalance.
* **Training:** Models trained for `EPOCHS=5` with early stopping (`patience=3`) based on average validation lenient F1 score.

## How to Run

1.  **Dependencies:**
    * Ensure Python 3.x is installed.
    * Install required libraries:
        ```bash
        pip install torch transformers datasets scikit-learn pandas matplotlib seaborn nltk
        ```
    * Download NLTK stopwords if not already present (the notebook handles this).

2.  **Dataset:**
    * Place the `assignment_3_ai_tutors_dataset.json` file in the same directory as the notebook, or update the `DATA_PATH` variable in the notebook.

3.  **Execution:**
    * Open and run the `NLP_assignment3.ipynb` Jupyter Notebook.
    * The notebook is structured to:
        * Set up configurations and constants.
        * Load and preprocess data.
        * Perform EDA.
        * Define model architectures, dataset/dataloader classes, and evaluation/training functions.
        * Run the experiment loop, iterating through different configurations.
        * (The notebook currently has placeholders for results analysis and final test set evaluation which would need to be completed based on the experiment outputs).

## Key Findings (Based on Validation Results)

* **Model Choice:** `roberta-base` and `distilbert-base-uncased` showed slightly better and more consistent performance than `bert-base-uncased` in terms of Average Lenient Accuracy.
* **Task Strategy:** Single-task models consistently outperformed the multi-task approach. This might be due to negative interference between tasks in the multi-task setup.
* **Input Formulation:** Using only the tutor's `response` (`include_history=False`) yielded better results than including `conversation_history`. This was likely due to severe truncation of long histories with `MAX_LENGTH=256`, leading to loss of crucial context or introduction of noise.
* **Loss Function:** `FocalLoss` provided the best average performance, effectively handling class imbalance. Weighted Cross Entropy performed the worst, possibly due to over-correction or sensitivity to weight calculation.

*(Refer to the notebook and presentation slides for detailed tables and plots of results.)*

## Future Work

* Conduct more extensive hyperparameter tuning.
* Perform a deeper error analysis of misclassified examples.
* Explore more advanced models or ensemble methods.
* Investigate alternative methods for incorporating conversation history (e.g., hierarchical encoding, increasing `MAX_LENGTH` if compute allows).
* Run a complete set of multi-task learning experiments if initial runs were limited.

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
