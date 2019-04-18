import pandas as pd 

df = pd.read_csv('2016fullwithrolling.csv')

dfplayoffs = df.dropna()

dfplayoffs = dfplayoffs.drop(columns='playoff')

dfplayoffs.to_csv('2016playoffs.csv')



# df2 = pd.read_csv('2014playoffs.csv')

# df2['Date'] = pd.to_datetime(df2["Date"]).dt.strftime("%Y%-m%d")
# # print(df2.columns)
# # print(df.columns)
# dates = df['Date']
# new_dates = []
# for date in dates:
# 	date = str(date)
# 	new_dates.append(date)

# df['Date'] = new_dates
# # print(df['Date'])
# # print(df2['Date'])



# mergedStuff = pd.merge(df, df2, on=['Date', 'a_team', 'h_team'], how='inner')
# print(mergedStuff.head())


# hometeams = {key: val for key, val in mergedStuff.groupby('h_team')}
# awayteams = {key: val for key, val in mergedStuff.groupby('a_team')}


# #print(hometeams)
# newguy = pd.DataFrame()

# guy = []
# for hometeam in hometeams:
# 	df1 = hometeams[hometeam]
# 	#print(df1)
# 	columns = ['Home Score', 'Away Score', 'fga', 'fg_pct', 'fg2a',	'fg2_pct', 'fg3a', 'fg3_pct', 'fta', 'ft_pct', 'opp_fga', 'opp_fg_pct', 'opp_fg2a', 'opp_fg2_pct', 'opp_fg3a', 'opp_fg3_pct', 'opp_fta', 'opp_ft_pct'] #put in columns we want to roll on
# 	for col in columns:
# 		df1[col + 'rol'] = df1[col].rolling(window=8).mean()
# 		guy.append(df1)
		

# newguy = pd.concat(guy)
# newguy = newguy.sort_index()	
# newguy = newguy.sort_values(by=['ranker'])

# bleh = []
# for awayteam in awayteams:
# 	df1 = awayteams[awayteam]

# 	columns = ['Home Score', 'Away Score', 'fga', 'fg_pct', 'fg2a',	'fg2_pct', 'fg3a', 'fg3_pct', 'fta', 'ft_pct', 'opp_fga', 'opp_fg_pct', 'opp_fg2a', 'opp_fg2_pct', 'opp_fg3a', 'opp_fg3_pct', 'opp_fta', 'opp_ft_pct'] #put in columns we want to roll on
# 	for col in columns: 
# 		df1[col + 'rol'] = df1[col].rolling(window=8).mean()
# 		#print(df1)
# 		bleh.append(df1)
		

# newbleh = pd.concat(bleh)

# newbleh = newbleh.sort_index()	
# newbleh = newbleh.sort_values(by=['ranker'])

# newdf = pd.concat([newguy, newbleh], axis=1)
# print(newdf.shape)
# print(newdf.columns)
# print(newdf.head())
# print(newdf['opp_ftarol'])
# print(newdf.tail())

# newdf = newdf.drop_duplicates()
# print(newdf.shape)
# print(newdf.columns)

# columns = ['Home Score', 'Away Score', 'mp', 'fg', 'fga', 'fg_pct', 'fg2', 'fg2a', 'fg2_pct', 'fg3', 'fg3a', 'fg3_pct', 'ft', 'fta', 'ft_pct', 'pts', 'opp_fg', 'opp_fga', 'opp_fg_pct', 'opp_fg2', 'opp_fg2a', 'opp_fg2_pct', 'opp_fg3', 'opp_fg3a', 'opp_fg3_pct', 'opp_ft', 'opp_fta', 'opp_ft_pct', 'opp_pts']

# newdf = newdf.drop(columns=columns)
# print(newdf.head())
# print(newdf.tail())
# newdf.to_csv('2019fullwithrolling.csv')
# # hometeams = df5['home_id']
# # awayteams = df5['opp_id']

# # home_teams = {key: val for key, val in df.groupby('home_id')}
# # away_teams = {key: val for key, val in df.groupby('opp_id')}

# # awayteams = away_teams

# # hometeams = home_teams



# for csv in csvs:
# 	df = pd.read_csv(csv)
# 	home_teams = {key: val for key, val in df.groupby('home_id')}
# 	away_teams = {key: val for key, val in df.groupby('opp_id')}

# 	awayteams = away_teams

# 	hometeams = home_teams
# 	guy = []
# 	for hometeam in hometeams:

# 		df1 = home_teams[hometeam]
# 		df2 = df1.drop(['Date', 'home_id', 'opp_id', 'w_l'],  axis=1)
# 		r = df2.rolling(window=10).mean()
# 		r['result']= df1['w_l']
# 		r['home_id']= df1['home_id']
# 		r['opp_id']= df1['opp_id']
# 		r['Date']= df1['Date']
# 		#print(r.columns)


# 		guy.append(r)

# 	newguy = pd.concat(guy)

# 	newguy = newguy.sort_index()
# 	#print(newguy)
# 	#print(newguy)


# 	newguy = pd.concat(guy)

# 	newguy = newguy.sort_index()
# 	#print(newguy)
# 	#print(newguy)



# 	bleh = []
# 	for awayteam in awayteams:
# 		df1 = away_teams[awayteam]
# 		df2 = df1.drop(['Date', 'home_id', 'opp_id', 'w_l'],  axis=1)
# 		r = df2.rolling(window=10).mean()

# 		r['home_id']= df1['home_id']
# 		r['opp_id']= df1['opp_id']
# 		r['Date']= df1['Date']
# 		#print(r.columns)


# 		bleh.append(r)



# 	newbleh = pd.concat(bleh)
# 	print(newbleh)


# 	newbleh = newbleh.sort_index()
# #print(newbleh)