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



list1 = ['Past_10_h', 'Past_10_v']
#Past_10_v	Past_10_h rating1_pre	rating2_pre
# , 'Past_10_h', 'Past_10_v' , 'rating2_pre'

train_years = ['2014']
test_years = ['2013']
sport = 'mlb'
train_fns = ['./data/' + sport + '_' + year + 's.csv' for year in train_years]
test_fns = ['./data/' + sport + '_' + year + 's.csv' for year in test_years]

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


# df2 = pd.read_csv('2017mlb.csv')
# df2 = df2.dropna()
# df2 = df2.drop_duplicates()

new_feature_cols = list1
x_new = test_df[new_feature_cols]
y_new = test_df['h_win']

guy = test_df['h_ML']
print(guy)


print(str(clfgtb.score(x_new, y_new)) + ' gdpercent on first playoffs')

probsgd = clfgtb.predict_proba(x_new)
probsgd = probsgd.tolist()
h = len(probsgd)
#print(df2.columns)

h_lines = list(guy)


a_lines = test_df['Open']
a_lines = list(a_lines)
winners = test_df['h_win']
winners = list(winners)
abets = []
hbets = []
allbets = []
n = 0
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


	if evaway > .1:
		a_bets = [away_winprob, a_line, evaway, roi_away, winner]
		abets.append(a_bets)
		allbets.append(a_bets)
		n += roi_away

	if evhome > .1:
		h_bets = [home_winprob, h_line, evhome, roi_home, winner]
		hbets.append(h_bets)
		allbets.append(h_bets)
		n += roi_home



all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
home_df = pd.DataFrame(hbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
away_df = pd.DataFrame(abets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])



total_roi = all_df['roi'].sum() 
rois = all_df['roi']
n = 0
for roi in rois:
	n += roi
	print(n)
	print(roi)

print(total_roi)
#print(all_df)
print(home_df)
print(away_df)
print(n)
print(all_df['roi'].mean())

g1 = [[.4, .7]]
pred1 = clfgtb.predict(g1)
prob1 = clfgtb.predict_proba(g1)
print(prob1)
print(pred1)

