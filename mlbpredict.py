import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# need to add cross validation scores
import numpy as np 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# import graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pickle


df = pd.read_csv('2011mlb.csv')
df = df.dropna()
df = df.drop_duplicates()



df3 = pd.read_csv('2014mlb.csv')
df3 = df3.dropna()
df3 = df3.drop_duplicates()



df6 = pd.read_csv('2015mlb.csv')
df6 = df6.dropna()
df6 = df6.drop_duplicates()

df4 = pd.read_csv('2018mlb.csv')
df4 = df4.dropna()
df4 = df4.drop_duplicates()

df5 = pd.read_csv('2016mlb.csv')
df5 = df5.dropna()
df5 = df5.drop_duplicates()

df7 = pd.read_csv('2013mlb.csv')
df7 = df7.dropna()
df7 = df7.drop_duplicates()

df = pd.concat([df, df3, df6, df7, df5, df4])

list1 = ['Past_10_v', 'Past_10_h', 'rating1_pre', 'rating2_pre']
#Past_10_v	Past_10_h rating1_pre	rating2_pre
# , 'Past_10_h', 'Past_10_v' , 'rating2_pre'

train_years = ['2011', '2013', '2014', '2015', '2016', '2017']
test_years = ['2018']
sport = 'mlb'
train_fns = ['./data/' + year + sport + '.csv' for year in train_years]
test_fns = ['./data/' + year + sport + '.csv' for year in test_years]
def clean(fn):
    df = pd.read_csv(fn)
    df = df.dropna()
    df = df.drop_duplicates()
    return df  
def dfs(fns):
    dfs = []
    for fn in fns:
        tmp_df = clean(fn)
        dfs.append(tmp_df)
    df = pd.concat(dfs)
    return df 


train_df = dfs(train_fns)
test_df = dfs(test_fns)
def yo(odd):
    # to find the adjusted odds multiplier 
    # returns float
    if odd == 0:
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)

feature_cols = list1
x = train_df[feature_cols]
y = train_df['h_win']

clfgtb = GradientBoostingClassifier(random_state = 0, n_estimators = 43, max_depth = 1, learning_rate = 1).fit(x, y)


df2 = pd.read_csv('2017mlb.csv')
df2 = df2.dropna()
df2 = df2.drop_duplicates()

new_feature_cols = list1
x_new = df2[new_feature_cols]
y_new = df2['h_win']

guy = df2['h_ML']
print(guy)


print(str(clfgtb.score(x_new, y_new)) + ' gdpercent on first playoffs')

probsgd = clfgtb.predict_proba(x_new)
probsgd = probsgd.tolist()
h = len(probsgd)
print(df2.columns)

h_lines = list(guy)


a_lines = df2['Open']
a_lines = list(a_lines)
winners = df2['h_win']
winners = list(winners)
abets = []
hbets = []
allbets = []
for i in range(h):

	home_winprob = probsgd[i][1]
	away_winprob = probsgd[i][0]
	winner = winners[i]
	h_line = h_lines[i]
	a_line = a_lines[i]
	evhome = home_winprob * yo(h_line) - away_winprob 
	evaway = away_winprob * yo(a_line) - home_winprob

	if winner == 1:
		roi_home = yo(h_line)
		roi_away = -1

	if winner == 0:
		roi_home = -1
		roi_away = yo(a_line)

	if  .1>evaway > 0:
		bet_amt = 1

	if .3>evaway > .1:
		bet_amt = 1.5

	if .5 >evaway > .3:
		bet_amt = 2

	if 1 > evaway > .5:
		bet_amt = 2.5

	if evaway > 1:
		bet_amt = 3

	if  .1> evhome > 0:
		bet_amt = 1

	if .3>evhome > .1:
		bet_amt = 1.5

	if .5 > evhome > .3:
		bet_amt = 2

	if 1 > evhome > .5:
		bet_amt = 2.5

	if evhome > 1:
		bet_amt = 3


	if evaway > 0:
		a_bets = [away_winprob, a_line, evaway, bet_amt*roi_away, winner]
		abets.append(a_bets)
		allbets.append(a_bets)

	if evhome > 0:
		h_bets = [home_winprob, h_line, evhome, bet_amt*roi_home, winner]
		hbets.append(h_bets)
		allbets.append(h_bets)

all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
total_roi = all_df['roi'].sum() 
print(total_roi)
print(all_df)
	
