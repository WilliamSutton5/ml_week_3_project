# Data Cleaning Lab I

This repository contains my submission for the Data Cleaning Lab I assignment.

## Project Structure

- `step1_3_analysis.py`  
  Contains written responses for Steps 1–3, including:
  - Dataset review and modeling questions  
  - Business metrics and data preparation decisions  
  - Data instincts and potential concerns  

- `step4_pipelines.py`  
  Contains reusable data preparation pipelines for both datasets, as required in Step 4.  
  Running this file will load the data, clean it, encode and scale features, and produce
  training, tuning, and testing datasets.

- `cc_institution_details.csv`  
  College completion / institutional characteristics dataset.

- `job_placement.csv`  
  Campus recruitment / job placement dataset.

- `requirements.txt`  
  Python dependencies required to run the code.

## How to Run

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
