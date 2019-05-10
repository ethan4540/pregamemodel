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
from sklearn.metrics import brier_score_loss
import warnings
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')
import pickle


df = pd.read_csv('./data/2016reg.csv')
df = df.dropna()
df = df.drop_duplicates()
print(type(df))
df2 = pd.read_csv('./data/2017reg.csv')
df2 = df2.dropna()
df2 = df.drop_duplicates()

df3 = pd.read_csv('./data/2018reg.csv')
df3 = df3.dropna()
df3 = df3.drop_duplicates()



df6 = pd.read_csv('./data/2019reg.csv')
df6 = df6.dropna()
df6 = df6.drop_duplicates()

df7 = pd.read_csv('./data/2017playoffs.csv')
df8 = pd.read_csv('./data/2016playoffs.csv')

df = pd.concat([df, df2, df3, df6, df7, df8])
columns = df.columns
list1 = columns[29:47]
list2 = columns[75:93]
list3 = list1.append(list2)

print(list3)

list4 = ['elo1_pre', 'elo2_pre', 'h_Home_Scorerol', 'h_Away_Scorerol', 'Away_Scorerol', 'Home_Scorerol']

# df1 = pd.read_csv('reg2017fullML&elo.csv')
# df2 = pd.read_csv('reg2018fullML&elo.csv')
# df = pd.concat([df, df1, df2])
def yo(odd):
    # to find the adjusted odds multiplier 
    # returns float
    if odd == 0:
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)


print(df.shape)
feature_cols = list4
x = df[feature_cols]
y = df['winner']

# x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.05, random_state=2)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
#use GridSearch Cv to hypertune the model 
param_grid = {'n_estimators': np.arange(1, 50)}
param_grid1 = {'learning_rate': np.arange(1, 10)/10}
param_grid2 = {'max_depth': np.arange(1, 3)}
param_grid3 = {'random_state': np.arange(0, 10)}
param_dist = {'n_estimators': np.arange(1, 50),
              'learning_rate': np.arange(1, 10)/10,
              'max_depth': np.arange(1, 3),
              'random_state': np.arange(0, 10)}
param_neighbors = {'n_neighbors': np.arange(3, 40)}


param_distnn = {'hidden_layer_sizes': np.arange(1, 50),
              
              'alpha': np.arange(1, 10)/10,
}
param_grids = [param_grid, param_grid1, param_grid2, param_grid3]



svc = LinearSVC(random_state=2).fit(x, y)

#n_estimators = 50 learning_rate=1.0, max_depth=1, random_state=0
clfgtb = GradientBoostingClassifier(random_state = 0, n_estimators = 43, max_depth = 1, learning_rate = 1)
#clfgtb = RandomizedSearchCV(clfgtb, param_dist,  cv=100)




cv_score = cross_val_score(clfgtb, x, y, cv = 10)
clfgtb = clfgtb.fit(x, y)
print(cv_score )


knn = KNeighborsClassifier(n_neighbors = 35).fit(x, y)


clf = RandomForestClassifier().fit(x, y)

#nn = GridSearchCV(nn, param_grids, cv = 15)

df2 = pd.read_csv('./data/2018playoffs.csv')

df2 = df2.dropna()
df2 = df2.drop_duplicates()






# df9 = pd.read_csv('po2017fullML&elo.csv')
# df10 = pd.read_csv('po2018fullML&elo.csv')
# df2 = pd.concat([df8, df9, df10])

new_feature_cols = list4
x_new = df2[new_feature_cols]
y_new = df2['winner']
# x.head()
print(x_new)
print(y_new)

#print(str(knn.score(x_new, y_new)) + 'knn percent on first playoffs')
print(str(clfgtb.score(x_new, y_new)) + ' gdpercent on first playoffs')
print(str(clf.score(x_new, y_new)) + 'clf percent on first playoffs')
print(str(svc.score(x_new, y_new)) + 'svm percent on first playoffs')

prob_pos_gd = clfgtb.predict_proba(x_new)[:, 1]
clf_score = brier_score_loss(y_new, prob_pos_gd)
print(clf_score)

clf_sigmoid = CalibratedClassifierCV(clfgtb, cv=2, method='sigmoid')
clf_sigmoid.fit(x_new, y_new)
# clfgtb = clf_sigmoid
prob_pos_sigmoid = clfgtb.predict_proba(x_new)[:, 1]
clf_sigmoid_score = brier_score_loss(y_new, prob_pos_sigmoid)
print(clf_sigmoid_score)
# probssvc = svc.predict_proba(x_new)
# probssvc = probssvc.tolist()

probsrf = clf.predict_proba(x_new)
probsrf = probsrf.tolist()
probsgd = clfgtb.predict_proba(x_new)
probsgd = probsgd.tolist()
h = len(probsgd)
probsknn = knn.predict_proba(x_new)
probsknn = probsknn.tolist()


h_lines = df2['h_ML']
winners = df2['winner']
winners = list(winners)

h_lines = list(h_lines)
#print(h_lines[0])

a_lines = df2['ML']
a_lines = list(a_lines)
dates = df2['Date']
dates = dates.tolist()
# print(len(a_lines))
# print(type(a_lines))
# print(a_lines[0])
net = 0
home_gains = 0
away_gains = 0
home_losses = 0
away_losses = 0
n = 0

gd_total = 0
knn_total = 0
svc_total = 0
total = 0
ntotal = 0
print(len(probsgd))
print(len(a_lines))
print(len(h_lines))

info_list = []
abets = []
hbets = []
allbets = []
n= 0
for i in range(h):

	home_winprob = probsgd[i][1]
	away_winprob = probsgd[i][0]
	winner = winners[i]
	h_line = h_lines[i]
	a_line = a_lines[i]
	evhome = home_winprob * yo(h_line) - away_winprob 
	evaway = away_winprob * yo(a_line) - home_winprob

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


	if winner == 'H':
		roi_home = yo(h_line)
		roi_away = -1

	if winner == 'A':
		roi_home = -1
		roi_away = yo(a_line)

	info = [home_winprob, h_line, evhome, roi_home, away_winprob, a_line, evaway, roi_away, winner]

	if evaway > 0:
		a_bets = [away_winprob, a_line, evaway, bet_amt * roi_away, winner]
		abets.append(a_bets)
		allbets.append(a_bets)
		n+= roi_away
		#print(n)

	if evhome > 0:
		h_bets = [home_winprob, h_line, evhome, bet_amt * roi_home, winner]
		hbets.append(h_bets)
		allbets.append(h_bets)
		n+=roi_home
		#print(n)
	info_list.append(info)
	
	
all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
home_df = pd.DataFrame(hbets, columns = ['home_winprob', 'h_line', 'evhome', 'roi_home', 'winner'])
away_df = pd.DataFrame(abets, columns = ['away_winprob', 'a_line', 'evaway', 'roi_away', 'winner'])

home_roi = home_df['roi_home'].sum() 
#print(home_roi)
home_ev = home_df['evhome'].sum()
# print('this is it boi')
# print(home_ev)
# print(home_df.head())
# print(away_df.head())

away_roi = away_df['roi_away'].sum() 
# print(away_roi)
away_ev = away_df['evaway'].sum()
# print('this is it boi')
# print(away_ev)

total_roi = all_df['roi'].sum() 
print(total_roi)
total_ev = all_df['ev'].sum()
print('this is it boi')
print(total_ev)
#print(all_df.shape)
#print(all_df)
#print(home_df)
#print(away_df)
average = all_df['roi'].mean() 
#print(all_df['roi'])
print(average)
#g1 = [[1595, 1662, 120.37, 108.62, 110.9, 106.3]]
#g2 = [[1651, 1673, 121.9, 108.6, 112.6, 106.225]]
g1 = [[1600, 1657, 104, 105.7, 122.2, 113.6]]
g2 = [[1670, 1653, 115.7, 109.5, 123.1, 113]]
g3 = [[1525, 1500, 120, 105, 120, 115]]
line1h = -170
line1a = 140


pred1 = clfgtb.predict(g1)
prob1 = clfgtb.predict_proba(g1)
print(prob1)

evhome = prob1[0][1] * yo(line1h) - prob1[0][0]
evaway = prob1[0][0] * yo(line1a) - prob1[0][1]
if  .1> evhome > 0:
	bet_amt = 1
	print('bet 1 on hometeam in game1')

if .3>evhome > .1:
	bet_amt = 1.5
	print('bet 1.5 on hometeam in game1')
if .5 > evhome > .3:
	bet_amt = 2
	print('bet 2 on hometeam in game1')

if 1 > evhome > .5:
	bet_amt = 2.5
	print('bet 2.5 on hometeam in game1')

if evhome > 1:
	bet_amt = 3
	print('bet 3 on hometeam in game1')

if  .1> evaway > 0:
	bet_amt = 1
	print('bet 1 on awayteam in game1')

if .3>evaway > .1:
	bet_amt = 1.5
	print('bet 1.5 on awayteam in game1')
if .5 > evaway > .3:
	bet_amt = 2
	print('bet 2 on awayteam in game1')

if 1 > evaway > .5:
	bet_amt = 2.5
	print('bet 2.5 on awayteam in game1')

if evaway > 1:
	bet_amt = 3
	print('bet 3 on awayteam in game1')

else:
	print('skip this')



pred1 = clfgtb.predict(g1)
prob1 = clfgtb.predict_proba(g1)
print(prob1)
print(pred1)
print('above is blazersgd')

pred2 = clfgtb.predict(g2)
prob2 = clfgtb.predict_proba(g2)
print(prob2)
print(pred2)
print('above is rocketsgd')


pred3 = clfgtb.predict(g3)
prob3 = clfgtb.predict_proba(g3)
print(prob3)
print(pred3)
print('above is rocketsgd')



# for i in range(h):
# 	if 1 > probsgd[i][1] * yo(h_lines[i]) - probsgd[i][0] > 0:
# 		#print('bought')
# 		n += 1
# 		if winners[i] == 'H':
# 			net += yo(h_lines[i])
# 			#print(road_teams[i])
# 			# print(check[i])
# 			#print(dates[i])
# 			#print(probsgd[i][1])
# 			#print('above is probs for home')
# 			#print(h_lines[i])
# 			#print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			gd_total += yo(h_lines[i])
# 			print(net)
# 		if winners[i] == 'A':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses +=1
			
# 			net -=1
# 			gd_total += -1
# 			print(net)
		
# 	# if 1 > probssvc[i][1] * yo(h_lines[i]) - probssvc[i][0] > 0:
# 	# 	#print('bought')
# 	# 	n += 1
# 	# 	if winners[i] == 'H':
# 	# 		net += yo(h_lines[i])
# 	# 		#print(road_teams[i])
# 	# 		# print(check[i])
# 	# 		#print(dates[i])
# 	# 		#print(probsgd[i][1])
# 	# 		#print('above is probs for home')
# 	# 		#print(h_lines[i])
# 	# 		#print('above is home line which we took')
# 	# 		# print('WW')
# 	# 		home_gains += yo(h_lines[i])
# 	# 		svc_total += yo(h_lines[i])
# 	# 		print(net)
# 	# 	if winners[i] == 'A':
# 	# 		# print(gdprobs[i][1])
# 	# 		# print('above is probs for home')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is home line which we took')
# 	# 		# print('LL')
			
			
# 	# 		net -=1
# 	# 		svc_total += -1
# 	# 		print(net)
		




# 	# if 1 > probsknn[i][1] * yo(h_lines[i]) - probsknn[i][0] > 0:
# 	# 	#print('bought')
# 	# 	n += 1
# 	# 	if winners[i] == 'H':
# 	# 		net += yo(h_lines[i])
# 	# 		knn_total += yo(h_lines[i])
# 	# 		#print(road_teams[i])
# 	# 		# print(check[i])
# 	# 		#print(dates[i])
# 	# 		#print(probsgd[i][1])
# 	# 		#print('above is probs for home')
# 	# 		#print(h_lines[i])
# 	# 		#print('above is home line which we took')
# 	# 		# print('WW')
# 	# 		home_gains += yo(h_lines[i])
			
# 	# 		print(net)
# 	# 	if winners[i] == 'A':
# 	# 		# print(gdprobs[i][1])
# 	# 		# print('above is probs for home')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is home line which we took')
# 	# 		# print('LL')
# 	# 		home_losses += 1
# 	# 		knn_total -=1
			
# 	# 		net -=1
# 	# 		print(net)

# 	# if 1 > probsrf[i][1] * yo(h_lines[i]) - probsrf[i][0] > 0:
# 	# 	#print('bought')
# 	# 	n += 1
# 	# 	if winners[i] == 'H':
# 	# 		net += yo(h_lines[i])
# 	# 		#print(road_teams[i])
# 	# 		# print(check[i])
# 	# 		#print(dates[i])
# 	# 		#print(probsgd[i][1])
# 	# 		#print('above is probs for home')
# 	# 		#print(h_lines[i])
# 	# 		#print('above is home line which we took')
# 	# 		# print('WW')

# 	# 		home_gains += yo(h_lines[i])
			
# 	# 		print(net)
# 	# 	if winners[i] == 'A':
# 	# 		# print(gdprobs[i][1])
# 	# 		# print('above is probs for home')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is home line which we took')
# 	# 		# print('LL')
# 	# 		home_losses += 1
			
# 	# 		net -=1
# 	# 		print(net)



# 	if  .3 >probsgd[i][0] * yo(a_lines[i]) - probsgd[i][1] > .1:
# 		n+= 1
# 		#print('bought')
# 		if winners[i] == 'A':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			gd_total += yo(a_lines[i])
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')

# 			print(yo)
# 			print(net)


# 		if winners[i] == 'H':
# 			away_losses += 1
			
# 			net -=1
# 			gd_total += -1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			print(net)

# 	# if  .3 >probsknn[i][0] * yo(a_lines[i]) - probsknn[i][1] > .1:
# 	# 	n+= 1
# 	# 	#print('bought')
# 	# 	if winners[i] == 'A':
# 	# 		net += yo(a_lines[i])
# 	# 		away_gains += yo(a_lines[i])
			
# 	# 		# print(probs[i][0])
# 	# 		# print('above is probs for away')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is away line which we took')
# 	# 		# print('WW')
# 	# 		knn_total += yo(a_lines[i])
# 	# 		print(yo)
# 	# 		print(net)


# 	# 	if winners[i] == 'H':
# 	# 		away_losses += 1
			
# 	# 		net -=1
# 	# 		knn_total -= 1
# 	# 		# print(probs[i][0])
# 	# 		# print('above is probs for away')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is away line which we took')
# 	# 		# print('LL')
# 	# 		print(net)

# 	# if  .3 >probsrf[i][0] * yo(a_lines[i]) - probsrf[i][1] > .1:
# 	# 	n+= 1
# 	# 	#print('bought')
# 	# 	if winners[i] == 'A':
# 	# 		net += yo(a_lines[i])
# 	# 		away_gains += yo(a_lines[i])
			
# 	# 		# print(probs[i][0])
# 	# 		# print('above is probs for away')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is away line which we took')
# 	# 		# print('WW')
# 	# 		print(yo)
# 	# 		print(net)


# 	# 	if winners[i] == 'H':
# 	# 		away_losses += 1
			
# 	# 		net -=1
# 	# 		# print(probs[i][0])
# 	# 		# print('above is probs for away')
# 	# 		# print(h_lines[i])
# 	# 		# print('above is away line which we took')
# 	# 		# print('LL')
# 	# 		print(net)
	
	
# print(net)
# print('above is net')
# print(home_gains - home_losses)
# print('above is home gains')
# print(away_gains - away_losses)
# print('above is away gains')
# print(gd_total)
# print('above is gd gains')
# print(knn_total)
# print('above is knn gains')
# print(svc_total)
# print('above is svc gains')
# print(n)

# total += net
# ntotal += n


# g1 = [[1538, 1591, 107.9, 106.9, 112, 113]]
# g2 = [[1577, 1622, 108.6, 106.9, 108, 112]]
# g3 = [[1525, 1637, 120, 105, 107, 115]]


# pred1 = clfgtb.predict(g1)
# prob1 = clfgtb.predict_proba(g1)
# print(prob1)
# print(pred1)
# print('above is cleticsgd')

# pred2 = clfgtb.predict(g2)
# prob2 = clfgtb.predict_proba(g2)
# print(prob2)
# print(pred2)
# print('above is netsgd')

# pred3 = clfgtb.predict(g3)
# prob3 = clfgtb.predict_proba(g3)
# print(prob3)
# print(pred3)
# print('above is rocketsgd')






# pred1 = knn.predict(g1)
# prob1 = knn.predict_proba(g1)
# print(prob1)
# print(pred1)
# print('above is netsknn')

# pred2 = knn.predict(g2)
# prob2 = knn.predict_proba(g2)
# print(prob2)
# print(pred2)
# print('above is warsknn')

# pred3 = knn.predict(g3)
# prob3 = knn.predict_proba(g3)
# print(prob3)
# print(pred3)
# print('above is nugsknn')

# pred1 = clf.predict(g1)
# prob1 = clf.predict_proba(g1)
# print(prob1)
# print(pred1)
# print('above is nets rf')

# pred2 = clf.predict(g2)
# prob2 = clf.predict_proba(g2)
# print(prob2)
# print(pred2)
# print('above is wars rf')

# pred3 = clf.predict(g3)
# prob3 = clf.predict_proba(g3)
# print(prob3)
# print(pred3)
# print('above is nugs rf')






# df3 = pd.read_csv('reg2017fullML&elo.csv')
# df4 = pd.concat([df, df2, df3])
# print(df4.shape)
# feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x = df4[feature_cols]
# y = df4['winner']


# clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)
# knn = KNeighborsClassifier(n_neighbors=100).fit(x, y)
# clf = RandomForestClassifier().fit(x, y)

# df2 = pd.read_csv('po2017fullML&elo.csv')
# # df9 = pd.read_csv('po2017fullML&elo.csv')
# # df10 = pd.read_csv('po2018fullML&elo.csv')
# # df2 = pd.concat([df8, df9, df10])


# new_feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x_new = df2[new_feature_cols]
# y_new = df2['winner']
# # x.head()

# probsclf = clf.predict_proba(x_new)
# probsclf = probsclf.tolist()
# probsknn = knn.predict_proba(x_new)
# probsknn = probsknn.tolist()
# print(str(knn.score(x_new, y_new)) + 'knn percent on second playoffs')
# print(str(clf.score(x_new, y_new)) + 'clf percent on second playoffs')

# print(str(clfgtb.score(x_new, y_new)) + 'gd percent on second playoffs')

# probsgd = clfgtb.predict_proba(x_new)
# probsgd = probsgd.tolist()
# h = len(probsgd)

# h_lines = df2['h_ML']
# winners = df2['winner']
# winners = list(winners)

# h_lines = list(h_lines)
# #print(h_lines[0])

# a_lines = df2['ML']
# a_lines = list(a_lines)
# dates = df2['Date']
# dates = dates.tolist()
# # print(len(a_lines))
# # print(type(a_lines))
# # print(a_lines[0])
# net = 0
# home_gains = 0
# away_gains = 0
# home_losses = 0
# away_losses = 0
# n = 0

# gd_gains = 0
# gd_losses = 0

# print(len(probsgd))
# print(len(a_lines))
# print(len(h_lines))

# for i in range(h):
# 	if 1 > probsgd[i][1] * yo(h_lines[i]) - probsgd[i][0] > 0:
# 		#print('bought')
# 		n += 1
# 		if winners[i] == 'H':
# 			net += yo(h_lines[i])
# 			#print(road_teams[i])
# 			# print(check[i])
# 			#print(dates[i])
# 			#print(probsgd[i][1])
# 			#print('above is probs for home')
# 			#print(h_lines[i])
# 			#print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			gd_gains += yo(h_lines[i])
# 			print(net)
# 		if winners[i] == 'A':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			print(net)
		



# 	if  .3 >probsgd[i][0] * yo(a_lines[i]) - probsgd[i][1] > .1:
# 		n+= 1
# 		#print('bought')
# 		if winners[i] == 'A':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			gd_gains += yo(a_lines[i])
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')
# 			print(net)


# 		if winners[i] == 'H':
# 			away_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			print(net)
	
	
# print(net)
# print('above is net')
# print(home_gains - home_losses)
# print('above is home gains')
# print(away_gains - away_losses)
# print('above is away gains')
# print(gd_gains - gd_losses)
# print('above is gd gains')
# print(n)
# total += net
# ntotal += n

# df = pd.read_csv('reg2016fullML&elo.csv')
# df1 = pd.read_csv('po2016fullML&elo.csv')
# df2 = pd.read_csv('reg2017fullML&elo.csv')
# df3 = pd.read_csv('po2017fullML&elo.csv')
# df4 = pd.read_csv('reg2018fullML&elo.csv')

# df = pd.concat([df, df1, df2, df3, df4])
# feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x = df[feature_cols]
# y = df['winner']

# clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)
# knn = KNeighborsClassifier(n_neighbors=100).fit(x, y)
# clf = RandomForestClassifier().fit(x, y)

# df2 = pd.read_csv('po2018fullML&elo.csv')
# # df9 = pd.read_csv('po2017fullML&elo.csv')
# # df10 = pd.read_csv('po2018fullML&elo.csv')
# # df2 = pd.concat([df8, df9, df10])


# new_feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x_new = df2[new_feature_cols]
# y_new = df2['winner']
# # x.head()

# print(str(knn.score(x_new, y_new)) + 'knn percent on third playoffs')
# print(str(clf.score(x_new, y_new)) + 'rf percent on third playoffs')

# print(str(clfgtb.score(x_new, y_new)) + 'gd percent on third playoffs')
# probsgd = clfgtb.predict_proba(x_new)
# probsgd = probsgd.tolist()
# h = len(probsgd)
# probsknn = knn.predict_proba(x_new)
# probsknn = probsknn.tolist()

# h_lines = df2['h_ML']
# winners = df2['winner']
# winners = list(winners)

# h_lines = list(h_lines)
# #print(h_lines[0])

# a_lines = df2['ML']
# a_lines = list(a_lines)
# dates = df2['Date']
# dates = dates.tolist()
# # print(len(a_lines))
# # print(type(a_lines))
# # print(a_lines[0])
# net = 0
# home_gains = 0
# away_gains = 0
# home_losses = 0
# away_losses = 0
# n = 0

# gd_gains = 0
# gd_losses = 0

# print(len(probsknn))
# print(len(probsgd))
# print(len(a_lines))
# print(len(h_lines))

# for i in range(h):
# 	if 1 > probsgd[i][1] * yo(h_lines[i]) - probsgd[i][0] > 0:
# 		#print('bought')
# 		n += 1
# 		if winners[i] == 'H':
# 			net += yo(h_lines[i])
# 			#print(road_teams[i])
# 			# print(check[i])
# 			#print(dates[i])
# 			#print(probsgd[i][1])
# 			#print('above is probs for home')
# 			#print(h_lines[i])
# 			#print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			gd_gains += yo(h_lines[i])
# 			print(net)
# 			#print(net)
# 		if winners[i] == 'A':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			print(net)
		



# 	if  .3 >probsgd[i][0] * yo(a_lines[i]) - probsgd[i][1] > .1:
# 		n+= 1
# 		#print('bought')
# 		if winners[i] == 'A':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			gd_gains += yo(a_lines[i])
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')
# 			print(net)


# 		if winners[i] == 'H':
# 			away_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			print(net)
	
	
# print(net)
# print('above is net')
# print(home_gains - home_losses)
# print('above is home gains')
# print(away_gains - away_losses)
# print('above is away gains')
# print(gd_gains - gd_losses)
# print('above is gd gains')
# print(n)
# total += net
# ntotal += n

# df = pd.read_csv('reg2016fullML&elo.csv')
# df1 = pd.read_csv('po2016fullML&elo.csv')
# df2 = pd.read_csv('reg2017fullML&elo.csv')
# df3 = pd.read_csv('po2017fullML&elo.csv')
# df4 = pd.read_csv('reg2018fullML&elo.csv')
# df5 = pd.read_csv('po2018fullML&elo.csv')


# df = pd.concat([df, df1, df2, df3, df4, df5])
# feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x = df[feature_cols]
# y = df['winner']


# clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)
# knn = KNeighborsClassifier(n_neighbors=100).fit(x, y)
# clf = RandomForestClassifier().fit(x, y)

# nn = MLPClassifier(hidden_layer_sizes=15, alpha=1).fit(x, y)
# df2 = pd.read_csv('reg2019fullML&elo.csv')
# # df9 = pd.read_csv('po2017fullML&elo.csv')
# # df10 = pd.read_csv('po2018fullML&elo.csv')
# # df2 = pd.concat([df8, df9, df10])


# new_feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x_new = df2[new_feature_cols]
# y_new = df2['winner']
# # x.head()


# print(str(knn.score(x_new, y_new)) + 'knn percent on third playoffs')
# print(str(clf.score(x_new, y_new)) + 'rf percent on third playoffs')
# print(str(nn.score(x_new, y_new)) + 'nn percent on third playoffs')

# print(str(clfgtb.score(x_new, y_new)) + 'gd percent on third playoffs')
# probsgd = clfgtb.predict_proba(x_new)
# probsgd = probsgd.tolist()
# h = len(probsgd)
# probsknn = knn.predict_proba(x_new)
# probsknn = probsknn.tolist()

# probsnn = nn.predict_proba(x_new)
# probsnn = probsnn.tolist()

# h_lines = df2['h_ML']
# winners = df2['winner']
# winners = list(winners)

# h_lines = list(h_lines)
# #print(h_lines[0])

# a_lines = df2['ML']
# a_lines = list(a_lines)
# dates = df2['Date']
# dates = dates.tolist()
# # print(len(a_lines))
# # print(type(a_lines))
# # print(a_lines[0])
# net = 0
# home_gains = 0
# away_gains = 0
# home_losses = 0
# away_losses = 0
# n = 0

# gd_gains = 0
# gd_losses = 0

# print(len(probsknn))
# print(len(probsgd))
# print(len(a_lines))
# print(len(h_lines))

# for i in range(h):
# 	if 1 > probsgd[i][1] * yo(h_lines[i]) - probsgd[i][0] > 0:
# 		#print('bought')
# 		n += 1
# 		if winners[i] == 'H':
# 			net += yo(h_lines[i])
# 			#print(road_teams[i])
# 			# print(check[i])
# 			#print(dates[i])
# 			#print(probsgd[i][1])
# 			#print('above is probs for home')
# 			#print(h_lines[i])
# 			#print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			gd_gains += yo(h_lines[i])
# 			print(net)
# 			#print(net)
# 		if winners[i] == 'A':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			print(net)
		



# 	if  .3 >probsgd[i][0] * yo(a_lines[i]) - probsgd[i][1] > .05:
# 		n+= 1
# 		#print('bought')
# 		if winners[i] == 'A':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			gd_gains += yo(a_lines[i])
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')
# 			print(net)


# 		if winners[i] == 'H':
# 			away_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			print(net)
	
	
# print(net)
# print('above is net')
# print(home_gains - home_losses)
# print('above is home gains')
# print(away_gains - away_losses)
# print('above is away gains')
# print(gd_gains - gd_losses)
# print('above is gd gains')
# print(n)
# total += net
# ntotal += n
# print(total)
# print(ntotal)

# df = pd.concat([df, df2])
# feature_cols = ['elo1_pre', 'elo2_pre', 'total']
# x = df[feature_cols]
# y = df['winner']

# clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)
# knn = KNeighborsClassifier(n_neighbors=100).fit(x, y)
# clf = RandomForestClassifier().fit(x, y)

# g1 = [[1492, 1571, 227.5]]
# g2 = [[1539, 1637, 236.5]]
# g3 = [[1569, 1585, 210.5]]


# pred1 = clfgtb.predict(g1)
# prob1 = clfgtb.predict_proba(g1)
# print(prob1)
# print(pred1)
# print('above is nets')

# pred2 = clfgtb.predict(g2)
# prob2 = clfgtb.predict_proba(g2)
# print(prob2)
# print(pred2)
# print('above is wars')

# pred3 = clfgtb.predict(g2)
# prob3 = clfgtb.predict_proba(g2)
# print(prob3)
# print(pred3)
# print('above is nugs')

# df4 = pd.read_csv('po2018fullML.csv')
# df2 = pd.read_csv('po2017fullML&elo.csv')
# # df9 = pd.read_csv('po2017fullML&elo.csv')
# # df10 = pd.read_csv('po2018fullML&elo.csv')
# # df2 = pd.concat([df8, df9, df10])






# new_feature_cols = ['elo1_pre', 'elo2_pre']
# x_new = df2[new_feature_cols]
# y_new = df2['winner']
# # x.head()


# print(str(clfgtb.score(x_new, y_new)) + 'percent on first playoffs')
# probsgd = clfgtb.predict_proba(x_new)
# probsgd = probsgd.tolist()
# h = len(probsgd)

# h_lines = df2['h_ML']
# winners = df2['winner']
# winners = list(winners)

# h_lines = list(h_lines)
# #print(h_lines[0])

# a_lines = df2['ML']
# a_lines = list(a_lines)
# dates = df2['Date']
# dates = dates.tolist()
# # print(len(a_lines))
# # print(type(a_lines))
# # print(a_lines[0])
# net = 0
# home_gains = 0
# away_gains = 0
# home_losses = 0
# away_losses = 0
# n = 0

# gd_gains = 0
# gd_losses = 0

# print(len(probsgd))
# print(len(a_lines))
# print(len(h_lines))
# df = pd.read_csv('come2017season.csv')
# df3 = pd.read_csv('come2015season.csv')

# #print(df.columns)

# df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# df3 = df3.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)




# print(df.shape)


# print(df3.shape)

# df5 = pd.concat([df3, df])

# print(df5.shape)
# # df.head()

# # df.tail()

# # df.shape

# # df.columns

# # df.isnull().sum().max()

# # corrmat = df.corr()
# # f, ax = plt.subplots(figsize=(20,18))
# # sns.heatmap(corrmat, vmax=.8, square=True)

# # # k = 9
# # # cols = corrmat.nlargest(k, 'teamRslt')['teamRslt'].index
# # # f, ax = plt.subplots(figsize=(10,6))
# # # cm = np.corrcoef(df[cols].values.T)
# # # sns.set(font_scale=1.25)
# # # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # # plt.show()

# # #team points

# # k = 12
# # cols = corrmat.nlargest(k, 'teamPTS')['teamPTS'].index
# # f, ax = plt.subplots(figsize=(10,6))
# # cm = np.corrcoef(df[cols].values.T)
# # sns.set(font_scale=1.25)
# # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # plt.show()


# # #opponent points

# # k = 12
# # cols = corrmat.nlargest(k, 'opptPTS')['opptPTS'].index
# # f, ax = plt.subplots(figsize=(10,6))
# # cm = np.corrcoef(df[cols].values.T)
# # sns.set(font_scale=1.25)
# # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # plt.show()

# # #location; h/a

# # # k = 12
# # # cols = corrmat.nlargest(k, 'teamLoc')['teamLoc'].index
# # # f, ax = plt.subplots(figsize=(10,6))
# # # cm = np.corrcoef(df[cols].values.T)
# # # sns.set(font_scale=1.25)
# # # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # # plt.show()

# # # team days off

# # k = 12
# # cols = corrmat.nlargest(k, 'teamDayOff')['teamDayOff'].index
# # f, ax = plt.subplots(figsize=(10,6))
# # cm = np.corrcoef(df[cols].values.T)
# # sns.set(font_scale=1.25)
# # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # plt.show()


# # # team turnovers

# # k = 12
# # cols = corrmat.nlargest(k, 'teamTO')['teamTO'].index
# # f, ax = plt.subplots(figsize=(10,6))
# # cm = np.corrcoef(df[cols].values.T)
# # sns.set(font_scale=1.25)
# # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # plt.show()
# def combine(df1, df2)
# 	df1 = pd.read_csv('come2017season.csv')
# 	df2 = pd.read_csv('come2015season.csv')


# 	df1 = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# 	df2 = df3.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# 	df3 = pd.concat([df2, d1])



# def yo(odd):
#     # to find the adjusted odds multiplier 
#     # returns float
#     if odd == 0:
#         return 0
#     if odd >= 100:
#         return odd/100.
#     elif odd < 100:
#         return abs(100/odd)
# # #scatter plots

# # cols1 = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
# # sns.pairplot(df[cols1], size=2.5)
# # plt.show()



# #prepare x and y

# feature_cols = ['h_Avg_pts', 'Avg_pts', 'elo1_pre', 'elo2_pre']
# x = df5[feature_cols]
# y = df5['result']
# x.head()

# x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.05, random_state=2)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# #knn 

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train, y_train)
# pred = knn.predict(x_test)
# print(str(metrics.accuracy_score(y_test, pred)) + 'initial KN classifier model')
# #print(knn.predict_proba(x_test))


# #linear svm

# clf = LinearSVC(random_state=2)
# clf.fit(x_train, y_train)
# #print(clf.coef_)
# #print(clf.intercept_)
# pred = (clf.predict(x_test))
# #print(pred)
# print(str(metrics.accuracy_score(y_test, pred)) + 'original SVC model')

# #random forrest classifier

# clf = RandomForestClassifier()
# clf.fit(x_train, y_train)

# #print(clf.feature_importances_)

# pred = clf.predict(x_test)
# #print(pred)
# #print(clf.predict_proba(x_test))
# print(str(metrics.accuracy_score(y_test, pred)) + 'original RandomForestClassifier')

# # Gradient Treee Boosting

# clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)
# print(str(clfgtb.score(x_test, y_test)) +'original Gradient boosting')


# df2 = pd.read_csv('imeanlike.csv')
# df2 = df2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# # df2.head()


# # #prepare x and y
# #print(df2.columns)
# # CamelCase
# # under_scores

# new_feature_cols = ['h_Avg_pts', 'Avg_pts', 'elo1_pre', 'elo2_pre']
# x_new = df2[new_feature_cols]
# y_new = df2['result']
# # x.head()
# dates = df2['Date1']
# dates = list(dates)

# road_teams = df2['Team']
# road_teams = list(road_teams)

# check = df2['opp_id_y']
# check = list(check)

# # # gradient tree boosting
# #print(knn.predict_proba(x_new))
# # knn = KNeighborsClassifier(n_neighbors=5)
# # knn.fit(x, y)

# pred = knn.predict(x_new)
# print(str(metrics.accuracy_score(y_new, pred)) + 'knn on 2018')
# probsknn = knn.predict_proba(x_new)
# probsknn = probsknn.tolist()

# clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)
# print(str(clfgtb.score(x_new, y_new)) + 'gradient boosting on 2018')
# probsgd = clfgtb.predict_proba(x_new)
# probsgd = probsgd.tolist()

# probsrf = clf.predict_proba(x_new)
# probsrf = probsrf.tolist()
# # print(probs[10])
# # print(type(probs))
# h_lines = df2['h_ML']
# results = df2['result']
# results = list(results)

# h_lines = list(h_lines)
# #print(h_lines[0])

# a_lines = df2['ML']
# a_lines = list(a_lines)
# # print(len(a_lines))
# # print(type(a_lines))
# # print(a_lines[0])
# net = 0
# home_gains = 0
# away_gains = 0
# home_losses = 0
# away_losses = 0
# n = 0

# gd_gains = 0
# gd_losses = 0
# knn_gains = 0
# knn_losses = 0
# rf_gains = 0
# rf_losses = 0
# for i in range(260):
# 	if .6 > probsgd[i][1] * yo(h_lines[i]) - probsgd[i][0] > .1:
# 		print('bought')
# 		n += 1
# 		if results[i] == 'W':
# 			net += yo(h_lines[i])
# 			# print(road_teams[i])
# 			# print(check[i])
# 			# print(dates[i])
# 			# print(probsgd[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			gd_gains += yo(h_lines[i])
# 			#print(net)
# 		if results[i] == 'L':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			print(net)
		



# 	if  .4 >probsgd[i][0] * yo(a_lines[i]) - probsgd[i][1] > .1:
# 		n+= 1
# 		#print('bought')
# 		if results[i] == 'L':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			gd_gains += yo(a_lines[i])
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')
# 			# print(net)


# 		if results[i] == 'W':
# 			away_losses += 1
# 			gd_losses += 1
# 			net -=1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			# print(net)
	
# 	if 1 > probsrf[i][1] * yo(h_lines[i]) - probsrf[i][0] > .15:
# 		#print('bought')
# 		n += 1
# 		if results[i] == 'W':
# 			net += yo(h_lines[i])
# 			# print(road_teams[i])
# 			# print(check[i])
# 			# print(dates[i])
# 			# print(probsgd[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			rf_gains += yo(h_lines[i])
# 			# print(net)
# 		if results[i] == 'L':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses += 1
# 			rf_losses += 1
# 			net -=1
# 			# print(net)

	



# 	if  1 >probsrf[i][0] * yo(a_lines[i]) - probsrf[i][1] > .15:
# 		n+= 1
# 		# print('bought')
# 		if results[i] == 'L':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			rf_gains += yo(a_lines[i])

# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')
# 			# print(net)

# 		if results[i] == 'W':
# 			away_losses += 1
# 			rf_losses += 1
# 			net -=1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			# print(net)



# 	if 1 > probsknn[i][1] * yo(h_lines[i]) - probsknn[i][0] > .15:
# 		#print('bought')
# 		n += 1
# 		if results[i] == 'W':
# 			net += yo(h_lines[i])
# 			# print(road_teams[i])
# 			# print(check[i])
# 			# print(dates[i])
# 			# print(probsgd[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('WW')
# 			home_gains += yo(h_lines[i])
# 			knn_gains += yo(h_lines[i])
# 			# print(net)
# 		if results[i] == 'L':
# 			# print(gdprobs[i][1])
# 			# print('above is probs for home')
# 			# print(h_lines[i])
# 			# print('above is home line which we took')
# 			# print('LL')
# 			home_losses += 1
# 			knn_losses += 1
# 			net -=1
# 			# print(net)



# 	if  1 >probsknn[i][0] * yo(a_lines[i]) - probsknn[i][1] > .15:
# 		n+= 1
# 		# print('bought')
# 		if results[i] == 'L':
# 			net += yo(a_lines[i])
# 			away_gains += yo(a_lines[i])
# 			knn_gains += yo(a_lines[i])

# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('WW')
# 			# print(net)

# 		if results[i] == 'W':
# 			away_losses += 1
# 			knn_losses += 1
# 			net -=1
# 			# print(probs[i][0])
# 			# print('above is probs for away')
# 			# print(h_lines[i])
# 			# print('above is away line which we took')
# 			# print('LL')
# 			# print(net)


# print(net)
# print('above is net')
# print(home_gains - home_losses)
# print('above is home gains')
# print(away_gains - away_losses)
# print('above is away gains')
# print(gd_gains - gd_losses)
# print('above is gd gains')
# print(knn_gains - knn_losses)
# print('above is knn gains')
# print(rf_gains - rf_losses)
# print('above is rf gains')
# print(n)


		


# pred = clf.predict(x_new)
# print(str(metrics.accuracy_score(y_new, pred)) + 'Random Forest on 2018')

# g1 = [[114.4, 107.3, 1775, 1542]]
# pred = clfgtb.predict(g1)
# print(pred)
# print(clfgtb.predict_proba(g1))


# filename = 'nba_pred_modelv1.sav'
# pickle.dump(clfgtb, open(filename, 'wb'))

# nba_pred_modelv1 = pickle.load(open(filename, 'rb'))
# pred1 = nba_pred_modelv1.predict(g1)
# prob1 = nba_pred_modelv1.predict_proba(g1)
# #print(pred1)

# #print(prob1)
# # #games
# # games = ['PHX1 vs SAS2', 'DET1 vs PHI2', 'MIN1 vs BOS2', 'NY1 vs MIA2', 'TOR1 vs MIL2', 'CHI1 vs DAL2', 'UTA1 vs DEN2', 'WSH1 vs MEM2', 'ATL1 vs POR2', 'CHA1 vs LAL2']

# # g1 = [[101.3, 111.9, 22.3, 15.9, 11.3, 87.1]]
# # g2 = [[108.4, 106.7, 18.7, 14.4, 10.3, 86.0]]
# # g3 = [[103.4, 109.8, 18.0, 13.2, 10.3, 84.8]]
# # g4 = [[102.2, 108.0, 20.8, 15.3, 10.8, 86.0]]
# # g5 = [[105.9, 105.4, 22.1, 13.9, 9.4, 86.5]]
# # g6 = [[101.6, 108.7, 19.2, 13.8, 9.4, 88.6]]
# # g7 = [[107.9, 106.8, 20.1, 14.4, 8.4, 82.4]]
# # g8 = [[98.9, 106.4, 21.7, 13.7, 9.9, 86.1]]
# # g9 = [[102.5, 110.9, 19.7, 15.5, 9.5, 84.7]]
# # g10 = [[106.7, 107.4, 18.1, 13.1, 10.5, 86.4]]

# # nba_pred_modelv1 = pickle.load(open(filename, 'rb'))

# # pred1 = nba_pred_modelv1.predict(g1)
# # prob1 = nba_pred_modelv1.predict_proba(g1)
# # print(pred1)
# # print(prob1)

# # pred2 = nba_pred_modelv1.predict(g2)
# # prob2 = nba_pred_modelv1.predict_proba(g2)
# # print(pred2)
# # print(prob2)

# # pred3 = nba_pred_modelv1.predict(g3)
# # prob3 = nba_pred_modelv1.predict_proba(g3)
# # print(pred3)
# # print(prob3)

# # pred4 = nba_pred_modelv1.predict(g4)
# # prob4 = nba_pred_modelv1.predict_proba(g4)
# # print(pred4)
# # print(prob4)

# # pred5 = nba_pred_modelv1.predict(g5)
# # prob5 = nba_pred_modelv1.predict_proba(g5)
# # print(pred5)
# # print(prob5)

# # pred6 = nba_pred_modelv1.predict(g6)
# # prob6 = nba_pred_modelv1.predict_proba(g6)
# # print(pred6)
# # print(prob6)


# # pred7 = nba_pred_modelv1.predict(g7)
# # prob7 = nba_pred_modelv1.predict_proba(g7)
# # print(pred7)
# # print(prob7)

# # pred8 = nba_pred_modelv1.predict(g8)
# # prob8 = nba_pred_modelv1.predict_proba(g8)
# # print(pred8)
# # print(prob8)

# # pred9 = nba_pred_modelv1.predict(g9)
# # prob9 = nba_pred_modelv1.predict_proba(g9)
# # print(pred9)
# # print(prob9)

# # pred10 = nba_pred_modelv1.predict(g10)
# # prob10 = nba_pred_modelv1.predict_proba(g10)
# # print(pred10)
# # print(prob10)

# # d = {'Game': games, 'Prediction':[pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10], 'Probability (1, 2)': [prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8, prob9, prob10], 'Actual Result': [2 , 2, 2, 1, 1, 1, 2, 1, 2, 1]}
# # df3 = pd.DataFrame(data = d)
# # df3


