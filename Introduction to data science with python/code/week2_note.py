#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:46:13 2019

@author: jinlingxing
"""

####week 2

##############################################################################
############################    notes      ###################################
##############################################################################

import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

# Your code here
a= df['Item Purchased']

df.loc['Store 1', 'Cost']

#transpose column name into indices
df.T
print(df.T)

#.loc does row selection
#.loc also supports slicing


#didn't change the data frame, but give you a copy of the data frame with the removing rows
df.drop('Store 1')

copy_df = df
copy_df = copy_df.drop('Store 1')

del copy_df['Name']
print(copy_df)

df['location'] = None
print(df)

#For the purchase records from the pet store, 
#how would you update the DataFrame, applying a discount of 20% across all the values in the 'Cost' column?
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

df['Cost'] = df['Cost'] *0.8

Costs = df['Cost']
Costs+=10
print(df)

#cat olympics.csv 

# Write a query to return all of the names of people who bought products worth more than $3.00.
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

df['Cost']>3
df_cost_greater_3 = df[df['Cost']>3]
df_cost_greater_3['Name']

df['Name'][df['Cost']>3]


# Reindex the purchase records DataFrame to be indexed hierarchically, first by store, then by person. 
# Name these indexes 'Location' and 'Name'. Then add a new entry to it with the value of:
# Name: 'Kevyn', Item Purchased: 'Kitty Food', Cost: 3.00 Location: 'Store 2'.
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

#df =df.set_index([' ', ' '])

df = df.set_index([df.index,'Name'])
df.index.names = ['Location','Name']

df = df.append(pd.Series(data= {'Item Purchased': 'Kitty Food', 'Cost':3.00}, name = ('Store 2', 'Kevyn')))


