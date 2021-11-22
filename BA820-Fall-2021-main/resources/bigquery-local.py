# Run through of the resources and code to query Big Query locally
# need to setup environment var per resource 1 below
# you may also need to run a few installs as noted below

# resources
# 1. https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries#client-libraries-install-python
# 2. https://cloud.google.com/bigquery/docs/bigquery-storage-python-pandas
# Supplemental: https://cloud.google.com/docs/authentication/getting-started
# TUTORIAL: https://googleapis.dev/python/bigquery/1.16.0/usage/pandas.html

# NOTES:  you should ensure that you have the environment variable set per 1

# POTENTIAL ISSUES: need to pip install pyarrow and google-cloud-bigquery-storage
# pip install --upgrade google-cloud-bigquery
# pip install pyarrow
# pip install pandas-gbq

# imports 
import pandas as pd

import google.auth
from google.cloud import bigquery



# this is my billing project
# NOTE:  You shoudl replace below with your own billing project
PROJECT_ID = 'questrom'


# make the query and get the data
SQL = "select * from `questrom.datasets.diamonds` limit 5"


# we can also do this via pandas 
df = pd.read_gbq(SQL, PROJECT_ID)

# IMPORTANT:
# NOTE:  you need to ensure you have the GOOGLE_APPLICATION_CREDENTIALS environment variable 
# #      properly set to the path per #1 above