# macros.py for highly usable data

data = '/data/'

url = 'https://www.baseball-reference.com'

teams_url = '/teams/'

schedule_suffix = '-schedule-scores.shtml'

mlb_teams_short = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
			'HOU', 'KCR', 'ANA', 'LAD', 'FLA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
			'PHI', 'PIT', 'SDP', 'SFG', 'SEA', 'STL', 'TBD', 'TEX', 'TOR', 'WSN']
 
mlb_teams_full = ['Arizona Diamondbacks', 'Atlanta Braves', 'Baltimore Orioles', 
				'Boston Red Sox', 'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds', 
				'Cleveland Indians', 'Colorado Rockies', 'Detroit Tigers', 'Houston Astros', 
				'Kansas City Royals', 'Los Angeles Angels', 'Los Angeles Dodgers', 
				'Miami Marlins', 'Milwaukee Brewers', 'Minnesota Twins', 'New York Mets', 
				'New York Yankees', 'Oakland Athletics', 'Philadelphia Phillies', 
				'Pittsburgh Pirates', 'San Diego Padres', 'San Francisco Giants', 
				'Seattle Mariners', 'St. Louis Cardinals', 'Tampa Bay Rays', 
				'Texas Rangers', 'Toronto Blue Jays', 'Washington Nationals']


mlb_csv_columns = ['game_id', 'inning', 'score', 'outs', 'robs', 'pitch_count', 'pitches', 'runs_outs', 'at_bat', 'wwpa', 'wwe', 'desc']


# Inn -- b is bottom, and t is top of the inning.
# Score
# ▲ -- Score from the perspective of the batting team.
# Out -- Number out at the start of the play.
# RoB -- Runners on Base at the start of the play.
# Click to see the defensive players and runners on base.
# Pit(cnt) -- Pitches (Balls-Strikes)
# Pitches Seen in PA with abbreviations
# Click to see the non-abbreviated pitch-by-pitch sequence
# C--called strike
# S--swinging strike
# F--foul
# B--ball
# X--ball put into play by batter

# T--foul tip
# K--strike (unknown type)
# I--intentional ball
# H--hit batter
# L--foul bunt
# M--missed bunt attempt
# N--no pitch (on balks and interference calls)
# O--foul tip on bunt
# P--pitchout
# Q--swinging on pitchout
# R--foul ball on pitchout
# U--unknown or missed pitch
# V--called ball because pitcher went to his mouth
# Y--ball put into play on pitchout
# 1--pickoff throw to first
# 2--pickoff throw to second
# 3--pickoff throw to third
# >--Indicates a runner going on the pitch
# +--following pickoff throw by the catcher
# *--indicates the following pitch was blocked by the catcher.
# .--marker for play not involving the batter
# R/O -- Number runs scored (R) and outs made (O) during the play.
# @Bat -- The team at bat
# wWPA -- Winning Team Win Probability Added (or Subtracted)
# Given average teams, this is the change in probability
# of eventual winner winning the game from the start of this play to the end of the play.
# wWE -- Winning Team Win Expectancy
# Given average teams, this is the probability
# of eventual winner winning the game at the end of this play.