PROJECT DETAILS FOR HEART DISEASE DATASET

1. Overview
    Project Goals
    - Identify key indicators of heart disease and create a solution to help predict and prevent heart disease developing
    - Construct a machine learning model that accurately predicts those at risk and improve overall quality of life

2. Project Description
    - Heart disease is the leading cause of death in the United States. The term 'heart disease' refers to several types of heart conditions. In the Unites States, the most common type of heart disease is coronary arter disease(CAD), which can lead to heart attack. You can greately reduce your risk for heart disease through lifestyle changes and, in some cases, medicine. This project uses the data of 400k adults and their responses to the 2020 annual CDC survey. 

3. Initial Hypothesis/Questions

    - Does risk of heart disease increase as age increases?
    - Does difficulty walking up stairs a key indicator for at risk of heart disease? 
    - Does being diabetic increase your risk for heart disease? 
    - Is smoking a key indicator of those at risk of heart disease?
    
4. Data Dictionary

    |Column | Description | Dtype|
    |--------- | --------- | ----------- |
    heart_disease | responders report having heart disease or myocardial infarction | object |
    BMI | Body Mass Index | float64 |
    smoking | responders that have smoked at least 100 cigarettes in their lives | object |
    alcohol_drinking | Men having more than 14 drinks per week and women having more than 7 drinks per week | int object |
    stroke | Responders that have had a stroke | object |
    physical_health | Days during the last 30 days where physical health was not good | float64 |
    mental_health | Days during the last 30 days where mental health was not good | float64 |
    diff_walking | Having serious difficulty walking or climbing stairs | object |
    sex | Female or Male | object |
    age_category | 13 age groups from 18 to 80 or older| object |
    race | 7 race groups | object |
    diabetic | Status of diabetes | object |
    physical_activity | Responders that reported doing physical activity or exercises during last 30 days | object |
    gen_health | Self evaluation of own health by 5 categories from 'Poor' to 'Excellent' | object |
    sleep_time | On average, the numbers of hours of sleep responders get in a 24 hour period | float64 |
    asthma | Status of asthma | object |
    kidney_disease | Responders that have had kidney disease except kidney stones, bladder infection, or incontinence | object |
    skin_cancer | Responders that have had skin cancer | object |
    

6. Project Plan

    Recreate the plan by following these steps

    Planning
    - Define goals
    - Determine audience and delivery format
    - Ask questions/formulate hypothesis
    - what is my MVP?

    Acquisition
    - Create a function that acquires the heart disease csv
    - Create and save in wrangle.py so function can be imported
    - Test functions

    Preparation
    - Create a function that preps the acquired data
    - This function will:
        - remove duplicates columns 
        - rename columns
        - encoding categorical variables for machine learning
    - Create a function that splits the data into 3 sets. Train, Validate, Test
        - Split 20% (test data), 24% (validate data), and 56%(test data)
    - Create a function that will normalize the numerical data
        - uses MinMaxScaler
    - Create functions and save in wrangle.py to be easily imported
    - Test functions
    
    Exploration
    - Use the initial questions to guide the exploration process
    - Create visualizations to help identify key indicators
    - Use statistical testing to confirm or deny hypothesis
    - Save work with notations in heart_explore.ipynb
    - Document answers to questions as takewaways

    Model
    - Train model
    - Make predictions
    - Evaluate model
    - Compute accuracy

    Delivery
    - Report is saved in Jupyter Notebook
    - Audience is anyone looking at my github and my peers reviewing my work

7. Recreation of Project:
    - You will need the .csv file for access to the dataset
    - Create a gitignore to prevent pushing any unwanted files
    - Download the wrangle.py file to working directory
    - Create a final notebook to your working directory
    - Review this README.md
    - Libraries used are pandas, numpy, matplotlib, Scipy, sklearn, seaborn
    - Run heart_final.ipynb

8. Key Findings and takeaways
    - As age increases so does your risk for heart disease
    - Difficulty walking or climbing stairs is an indicator of at risk for heart disease
    - Being diabetic increases your risk for heart disease
    - Smoking cigarettes increases your risk for heart disease
    - Reported Physical Health can be a risk factor