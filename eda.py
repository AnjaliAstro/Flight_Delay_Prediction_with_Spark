import pandas as pd
import numpy as np 
import matplotlib as plt
import matplotlib.pyplot as pyplot
import seaborn as sns
import ast, json
import glob
import csv 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
pd.set_option('display.max_columns', 19)

# Printing column names with Panda

df = pd.read_csv('C:\\Users\\Anjali\\Downloads\\try2\\Anjali.csv')            #('C:\\Users\\Anjali\\Desktop\\pycsv\\644175800_T_ONTIME_REPORTING.csv')
# df = pd.read_csv('C:\\Users\\Anjali\\Downloads\\H-data\\1987.csv')


delete_col = ['YEAR', 'MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UniqueCarrier','TAIL_NUM','CANCELLED','CANCELLATION_CODE','DIVERTED',
'NAS_DELAY','CARRIER_DELAY', 'WEATHER_DELAY', 'LATE_AIRCRAFT_DELAY']
# delete_col = ['Year', 'Month','DayofMonth','DayOfWeek','UniqueCarrier','FlightNum','TailNum','Cancelled','CancellationCode','Diverted',
# 'NASDelay']
# delete_col = ['YEAR', 'MONTH','DAY','DAY_OF_WEEK','AIRLINE', 'FLIGHT_NUMBER' ,'TAIL_NUMBER','CANCELLED','CANCELLATION_REASON','DIVERTED',
# 'AIRLINE_DELAY','AIR_SYSTEM_DELAY', 'SECURITY_DELAY']
df = df.drop(delete_col,axis = 1)
#print(df.describe())
col_names = list(df.columns)
print("\n columns :- ",col_names)
print('\n')

correlation_matrix = df.corr()
#print(correlation_matrix)
sns.heatmap(correlation_matrix,annot = True)
pyplot.show()





# Printing the column names using csv
# with open('C:\\Users\\Anjali\\Desktop\\pycsv\\Anjali.csv') as csv_file: 
  
#     csv_reader = csv.reader(csv_file, delimiter = ',') 
   
#     list_of_column_names = [] 
  
#     for row in csv_reader: 
#     	list_of_column_names.append(row) 
#     	break 
# print("List of column names : ", 
#       list_of_column_names[0]) 