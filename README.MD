Project Details for Predicting Heart Disease

PROJECT DETAILS FOR TELCO DATASET

1. Overview
    Project Goals
    - Identify key drivers of customer churn and create a solution to help improve customer retention
    - Construct a machine learning model that accurately predicts at risk customers so Telco can better focus its retention efforts 

2. Project Description
    - Telco Communications prides itself on customer satisfaction and quality of service. Maintaing a good reputation comes by focusing on current customers and keeping up with emerging patterns in the data. By accurately predicting the at risk customers we can help improve customer retention and increase company revenue. Telco can then focus on improving customer experience and quality of product.

3. Initial Hypothesis/Questions

    - Does contract type have association with churn?
    - What is the average length of tenure before customers decide to churn? 
    - Does service type play a key factor in the decision to churn? 
    - How much do customers pay per month?
    
4. Data Dictionary

    |Column | Description | Dtype|
    |--------- | --------- | ----------- |
    churn | the target variable: has churned or not | uint8 |
    customer_id | unique id number to identify customer | int64 |
    is_senior_citizen | whether customer is senior citizen | int64 |
    tenure | months with the company | int 64 |
    multiple_lines | whether customer has more than one line | int 64 |
    monthly_charges | customer monthly rate | float64 |
    total_charges | how much customer has paid | float 64 |
    contract_type | one year, two year, month-to-month | object |
    internet_service_type | DSL, Fiber optic, None | object |
    payment_type | how the customer pays balance | object |
    is_male | whether customer is male or not | uint8 |
    has_partner | whether customer has partner or not | uint8 |
    has_dependents | whether customer has dependents or not | uint8 |
    online_security | whether customer has online security or not | uint8 |
    online_backup | whether customer has online back up or not |uint8 |
    phone_service | whether customer has phone service or not | uint8 |
    device_protection | whether customer has device protection or not | uint8 |
    tech_support | whether customer has tech support or not | uint8 |
    streaming_tv | whether customer streams tv or not | uint8  |
    streaming_movies | whether customer streams movies or not | uint |
    paperless_billing | whether customer has paperless billing or not | uint8 |
    one_line | whether customer has one line or not | uint8 |
    no_phone_service | whether customer has no phone service or not | uint8 |
    has_multiple_lines | whether customer has multiple lines or not | uint8 |
    month_to_month_contract | whether customer has month to month or not | uint8 |
    one_year_contract | whether customer has one year contract or not | uint8 |
    two_year_contact | whether customer has two year contract or not | uint8 |
    dsl_internet | whether customer has dsl internet or not | uint8 |
    fiber_optic_internet | whether custome rhas fiber optic or not | uint8 |
    no_internet_service | whether a customer has no internet or not | uint8 |
    bank_transfer_autopay | whether a customer pays via bank transfer | uint8 |
    credit_card_autopay | whether a customer pays via credit card | uint8 |
    electronic_check_nonauto | whether a customer pays via e-check | uint8 |
    mailed_check_nonauto | whether a customer pays via mailed check | uint8 |
    is_autopay | whether a customer has autopay or not | uint8 |

6. Project Plan

    Recreate the plan by following these steps

    Planning
    - Define goals
    - Determine audience and delivery format
    - Ask questions/formulate hypothesis
    - what is my MVP?

    Acquisition
    - Create a function that establishes connection to telco_churn_db
    - Create a function that holds your SQL query and reads results
    - Creating functing for caching data and stores as .csv for ease
    - Create and save in acquire.py so functions can be imported
    - Test functions

    Preparation
    - Create a function that preps the acquired data
    - This function will:
        - remove duplicates columns 
        - convert data types
        - encoding categorical variables for machine learning
        - empty values from new customers are dropped
        - changes 'no internet service' values with 'no' for easier encoding
        - renames columns
    - Create a function that splits the data into 3 sets. Train, Validate, Test
        - Split 20% (test data), 24% (validate data), and 56%(test data)
    - Create functions and save in prepare.py to be easily imported
    - Test functions
    
    Exploration
    - Use the initial questions to guide the exploration process
    - Create visualizations to help identify drivers
    - Use statistical testing to confirm or deny hypothesis
    - Save work with notations in telco_work.ipynb
    - Document answers to questions as takewaways

    Model
    - Train model
    - Make predictions
    - Evaluate model
    - Compute accuracy

    Delivery
    - Report is saved in Jupyter Notebook
    - Presented via Zoom
    - The audience is direct manager and their manager

7. Recreation of Project:
    - You will need an env file with database credentials saved to you working directory
        - database credentials (username, password, hostname) 
    - Create a gitignore with env file inside to prevent sharing of credentials
    - Download the acquire.py and prepare.py files to working directory
    - Create a final notebook to your working directory
    - Review this README.md
    - Libraries used are pandas, matplotlib, Scipy, sklearn, seaborn, and numpy
    - Run telco_final.ipynb
    - Create a .csv file with the predictions from your ML model of at risk customers

8. Key Findings and takeaways
    - Customers who paying more than the average monthly rate are at higher risk of churn 
    - Fiber optic internet service is associated with higher risk of churn
    - Non autopay plans are at higher risk for churn
    - Month to month contracts churn more than other long-term contracts
    - The logistical regression model performed with 80% accuracy compared to the baseline accuracy of 73%
    - 