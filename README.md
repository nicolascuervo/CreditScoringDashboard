# CreditScoringAPI

This project was developepd in an academic context based on the scenario presented at https://www.kaggle.com/c/home-credit-default-risk/.

hereby is presented the dashboard for interacting with the [API](https://github.com/nicolascuervo/CreditScoringAPI) that serves ONE model for ONE client loan request

A fonctionning version can be found at : https://credit-score-dashboard-886b059eff82.herokuapp.com/

# How to use
## Locally
the following environment variables are to be defined in a file '.env' at root:

```
MIN_SK_ID_CURR=100001
MAX_SK_ID_CURR=456255
FAST_API=http://127.0.0.1:8000
```
On terminal go to root directory and input:

```
streamlit run streamlit/dashboard.py
```
## Cloud


### Heroku Deployment

Update requirements.txt with
 ```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```
and push the requirements.txt file

Also push with file `Procfile` contaning this line for the API to deploy

```
web: streamlit run streamlit/dashboard.py --server.port $PORT --server.address 0.0.0.0
```

the following environment variables are to be defined:

```
MIN_SK_ID_CURR=100001
MAX_SK_ID_CURR=456255
FAST_API=https://api-app_name-00000.herokuapp.com
```

When the API is running swagger for the API can be consulted on
```url
url_api/docs
```
