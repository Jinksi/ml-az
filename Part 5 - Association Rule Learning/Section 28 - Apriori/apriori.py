# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
rows_len = len(dataset)
columns_len = len(dataset.iloc[0, :].values)

# Split the dataset into a list of lists
transactions = []
for i in range(0, rows_len):
    transaction = dataset.values[i]
    # Apriori method is expection each transaction in a string
    transaction_string = str(transaction)
    transactions.append(transaction_string)
transactions

# Training the Apriori model
from apyori import apriori
# support is the minimum amount of times an item is purchased per day
    # 3 times a day * 7 days a week / 7500 total transactions = 0.0028
# confidence is the percentage of transactions that share product1 + product2
    # 20%
# lift is the likelihood that new transactions will contain both products
    # 300%
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3)

# Visualise the results
# Haven't been able to visualise... moving on
# results = list(rules)
