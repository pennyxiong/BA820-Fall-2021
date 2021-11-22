# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    # interactive plotting for notebook environments

from mlxtend.frequent_patterns import apriori, association_rules

# my BILLING project in Google Cloud - replace "questrom" with your project
PROJECT = "questrom"

# SQL
SQL = """
select * from `questrom.datasets.groceries`
"""

groceries = pd.read_gbq(SQL, PROJECT)


# what we have
type(groceries)
groceries.shape
groceries.sample(3)


# QUICK EXERCISE
# identify the number of unique transactions
# identify the number of unique items (item)
# determine the number of duplicates
# 5-minutes 

# unique transactions
groceries.tid.nunique()
groceries['tid'].nunique()

groceries.item.nunique()

dupes = groceries.duplicated()
dupes.sum()

# reshape our dataset
groceries['purchased'] = True
groceries.head(3)

tx = groceries.pivot(index='tid', columns='item', values='purchased')
tx.shape

# alot of missing data
tx.isna().sum()

# replace the NaN
tx.fillna(False, inplace=True)
tx.isna().sum().sum()

tx.shape

## calcuate the number of items per transaction
# distribution
item_freq = tx.sum(axis=1)
item_freq.value_counts()
item_freq.value_counts().plot(kind="bar")
plt.show()

## your turn
# EXERCISE
# IDENTIFY THE TOP 10 products
# how many transaction were contained 
prod_freq = tx.sum(axis=0)
prod_freq.sort_values(ascending=False)[:10]
prod_freq.sort_values(ascending=False)[-10:]


## fit the model - Part 1
itemsets = apriori(tx, min_support=.003, use_colnames=True)
type(itemsets)
itemsets.head(3)

# part 2 -- filter the rules
rules = association_rules(itemsets, metric='confidence', min_threshold=.2)
len(rules)

rules.head(1).T



# we need to figure out how to sell more bottled water
bottled_water = rules.loc[rules.consequents == {'bottled water'}, :]
len(bottled_water)

bottled_water.sample(4).T


# visualize the relationships

sns.scatterplot(data=bottled_water, x="support", y="confidence", size="lift", hue="lift")
plt.show()
