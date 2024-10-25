# CreditScoringAPI

This project was developepd in an academic context based on the scenario presented at https://www.kaggle.com/c/home-credit-default-risk/.

hereby is presented the dashboard for interacting with the [API](https://github.com/nicolascuervo/CreditScoringAPI) that serves ONE model for ONE client loan request

> It is running localy at this stage

# How to use

Include the following files  folder project-root/data
```
project-root/
│
└── data/
    ├── application_test.csv
    ├── application_train.csv
    └── input_information.csv
```
application_test/train.csv are available at https://www.kaggle.com/c/home-credit-default-risk/data.
input_information.csv is generated in parallel [CreditScoring project](https://github.com/nicolascuervo/CreditScoring)
  
# run

On terminal input:
```
streamlit run path/to/dashboard.py
```
> the dashboard is ment to work with a limited MIN_SK_ID_CURR = 100001 and MAX_SK_ID_CURR = 456255. If the data ever expands this values shoul be changed