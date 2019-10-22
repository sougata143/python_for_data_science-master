# Pandas contain high level data structures and manipulation tools to make data analysis fast and easy in Python.
# working with series, data frames, dropping entries from an axis, working with missing values, etc

#source https://bigdata-madesimple.com/step-by-step-approach-to-perform-data-analysis-using-python/

import pandas as pd 
import numpy as np
from pandas import Series, DataFrame # Series and Data Frame are two data structures available in python

#Series
print("Working with Series in Pandas")
mjp= Series([5,4,3,2,1]) #a simple series
print("A simple series: {}".format(mjp))
print("Values of series mjp: {}".format(mjp.values))
print("returns index of the series: {}".format(mjp.index))
jeeva = Series([5,4,3,2,1,-7,-29], index =['a','b','c','d','e','f','h']) # The index is specified
print("printing series jeeva: {}".format(jeeva))
print("printing values of jeeva: {}".format(jeeva.values))
print("printing index of jeeva: {}".format(jeeva.index))
jeeva['a'] = 26 #changing value of a particular index
print("jeeva after change: {}".format(jeeva))
print("returning only positive values: {}".format(jeeva[jeeva>0]))
jeeva['a'] = 5
print("mean value of jeeva: {}".format(np.mean(jeeva)))
print("checking an index is present in the series or not: {}".format('b' in jeeva))
print("checking an index is present in the series or not: {}".format('z' in jeeva))
player_salary ={
    'Rooney': 50000, 
    'Messi': 75000, 
    'Ronaldo': 85000, 
    'Fabregas':40000, 
    'Van persie': 67000
}
new_player = Series(player_salary)
print("before making it series: {}".format(player_salary))
print("after making it series: {}".format(new_player))
players =['Klose', 'Messi', 'Ronaldo', 'Van persie', 'Ballack'] 
# changed the index of the Series. Since, no value was not found for Klose and Ballack, it appears as NAN
player_1 =Series(player_salary, index= players) 
print("printing series with index: {}".format(player_1))
print("checks if there is null values or not: {}".format(pd.isnull(player_1)))
print("checks if there is not null values or not: {}".format(pd.notnull(player_1)))
player_1.name ='Bundesliga players' # name for the Series
player_1.index.name='Player names' #name of the index
print(player_1)
player_1.index =['Neymar', 'Hulk', 'Pirlo', 'Buffon', 'Anderson'] # is used to alter the index of Series
print("after changing index: {}".format(player_1))

#DataFrame
print("working with DataFrames in pandas")
states ={'State' :['Gujarat', 'Tamil Nadu', ' Andhra', 'Karnataka', 'Kerala'],
                  'Population': [36, 44, 67,89,34],
                  'Language' :['Gujarati', 'Tamil', 'Telugu', 'Kannada', 'Malayalam']}
india = DataFrame(states) # creating a data frame
print("DataFrame: ")
print(india)
india1 = DataFrame(states, columns=['Language','State', 'Population']) # change the sequence of column index
print("after changing sequence of column index: ")
print(india1)
#if you pass a column that isnt in states, it will appear with Na values
new_farme = DataFrame(states, 
                        columns=['State', 'Language', 'Population', 'Per Capita Income'], 
                        index =['a','b','c','d','e'])
print("if you pass a column that isnt in states, it will appear with Na values")
print(new_farme)
print("retrieveing data like dictionary")
print(new_farme['State']) # retrieveing data like dictionary
# rows can be retrieved using .ic function
print("rows can be retrieved using .ic function")
print(new_farme.ix[3])
#like series
print("like series")
print(new_farme.Population)
# the empty per capita income column can be assigned a value
print("the empty per capita income column can be assigned a value")
new_farme['Per Capita Income'] = np.arange(5)
print(new_farme)
#when assigning list or arrays to a column, the values lenght should match the length of the DataFrame
print("#when assigning list or arrays to a column, the values lenght should match the length of the DataFrame")
series = Series([44,33,22], index =['b','c','d'])
new_farme['Per Capita Income'] = series
print(new_farme)
new_farme['Development'] = new_farme.State == 'Gujarat'# assigning a new column
print("after adding column development")
print(new_farme)
del new_farme['Development']
print("after deleting the development column")
print(new_farme)
new_data ={'Modi': {2010: 72, 2012: 78, 2014 : 98},'Rahul': {2010: 55, 2012: 34, 2014: 22}}
elections = DataFrame(new_data) 
print("the outer dict keys are columns and inner dict keys are rows")
print(elections)
print("Transpose of elections")
print(elections.T)
ex= {'Gujarat':elections['Modi'][:-1], 'India': elections['Rahul'][:2]}
px =DataFrame(ex)
print("**********")
print(px)
print("Assiging names")
px.index.name = 'year'
px.columns.name = 'politicians'
print(px)

#Reindex
print("Reindex")
var = Series(['Python', 'Java', 'c', 'c++', 'Php'], index =[5,4,3,2,1])
print (var)
var1 = var.reindex([1,2,3,4,5])# reindex creates a new object 
print("after reindexing")
print (var1) 

#Dropping entries from axis
print("Dropping entries from aixs")
er = Series(np.arange(5), index =['a','b','c','d','e'])
print(er)
erd = er.drop(['a','b'])
print("after dropping a and b")
print(erd)
states ={'State' :['Gujarat', 'Tamil Nadu', ' Andhra', 'Karnataka', 'Kerala'],
                  'Population': [36, 44, 67,89,34],
                  'Language' :['Gujarati', 'Tamil', 'Telugu', 'Kannada', 'Malayalam']}
india = DataFrame(states, columns =['State', 'Population', 'Language'])
print("original")
print(india)
print("after drop")
print(india.drop([0,1]))

#Selection, Indexing and Filtering
print("Selection, Indexing and Filtering")
var = Series(['Python', 'Java', 'c', 'c++', 'Php'], index =[5,4,3,2,1])
print(var)
print(var[5])
print(var[2:4])
print("Printing first 3 rows")
print(india[:3])