# install -- if needed in colab or in your environment
# pip install mlxtend

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    # interactive plotting for notebook environments

from mlxtend.frequent_patterns import apriori, association_rules

# my BILLING project in Google Cloud - replace "questrom" with your project
PROJECT = "YOUR_PROJECT_HERE"

# NOTE: the dataset for module 1 is also in the Github repo, but this is only temporary
# NOTE: you will need to configure GCP accounts and be able to query Big Query on your own!

# DATA Dictionary for the groceries dataset
# tid = transaction id (id in a database that groups all items on a receipt together)
# item = the individual item