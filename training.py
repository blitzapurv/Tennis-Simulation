import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("data/pbp_matches_atp_qual_current.csv")
df_info = df[['tny_name','server1','server2','pbp_id','pbp','winner']]
df_info['winner_new'] = np.where(df_info['winner'] == 1, df['server1'],df['server2'] )
df_info.rename(columns={'tny_name': 'Tournament_Name','pbp_id': 'Match_id'}, inplace=True)
df_info['pbp'] = df_info['pbp'].str.split(';')
df_info = df_info.explode('pbp').reset_index(drop= True)

def clean(row):
    strng = ''
    for chrs in row:
        if chrs in {'S', 'R', 'D', 'A'}:
            strng += chrs
    return strng

df_info['Game_number'] = df_info.groupby(['server1']).cumcount() + 1
df_info['pbp'] = df_info['pbp'].apply(clean)

i = 1
j = 0

while i < len(df_info['Match_id']):
    
    if df_info.loc[i,'Match_id'] != df_info.loc[i-1,'Match_id']:
        i= i+1
        j = 0
    
    if j % 2 == 0:
        df_info.loc[i, 'server1'], df_info.loc[i, 'server2'] = df_info.loc[i, 'server2'], df_info.loc[i, 'server1']
    
    i += 1
    j += 1


df_info.rename(columns={'server1': 'server','server2': 'Receiver'}, inplace=True)

df_info['pbp'] = df_info['pbp'].str.replace('D','R')
df_info['pbp'] = df_info['pbp'].str.replace('A','S')

df_info['pbp'] = df_info['pbp'].apply(lambda x: ','.join(list(x)))


df_info['pbp'] = df_info['pbp'].str.split(',')
df_info = df_info.explode('pbp').reset_index(drop= True)

df_info['Serve_number'] = df_info.groupby(['Match_id']).cumcount() + 1

df_info['server_won'] = np.where(df_info['pbp']== 'S',1,0)

df_info['server_won_cumulative']  = df_info.groupby(['Tournament_Name','server','Match_id'])['server_won'].cumsum()
df_info['server_serves_cumulative']  = df_info.groupby(['Tournament_Name','server','Match_id'])['Serve_number'].cumcount()

df_info['server_won_cumulative'] = np.where(df_info['server_won'] == 1,df_info['server_won_cumulative']-1,df_info['server_won_cumulative'])



encoder = LabelEncoder()
df_info['server'] = encoder.fit_transform(df_info['server'])
df_info['Receiver'] = encoder.fit_transform(df_info['Receiver'])

with open('encoders/player_encoder.pkl', 'wb') as files:
    pickle.dump(encoder, files)

df_info['server'] = df_info['server'].astype('category')
df_info['Receiver'] = df_info['Receiver'].astype('category')

df_info['server_win_ratio'] = df_info['server_won_cumulative']/df_info['server_serves_cumulative']
df_info['server_win_ratio'] = np.where(df_info['server_win_ratio'].isna(),0,df_info['server_win_ratio'])

df_info.to_csv("data/data.csv")

# separate the data into features and target
feature_cols = ['server', 'Receiver','Game_number','Serve_number','server_won_cumulative']
       
X = df_info[feature_cols]
y = df_info.server_won


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth = 10)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the classes of the testing data
y_pred = clf.predict(X_test)


with open('models/classifier.pkl', 'wb') as files:
    pickle.dump(clf, files)
        
        

