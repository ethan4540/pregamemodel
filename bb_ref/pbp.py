import os
import sys

import bs4 
import requests as r
import macros as m
import openers as of


def get_table(page, table_id):  # given bs4 page and table id, finds table using bs4. returns tbody
	table = page.find('table', {'id': table_id})
	return table


def parse_table(table, split='th'):
	tbody = table.tbody
	rows = tbody.find_all('tr')
	data_rows = [[] for i in range(len(r))]
	for row in rows:

		row_class = row.get('class')
		
		if row_class == 'spacer':
			continue
		
		print(row.text)

		row_data = []
		
		things = row.find_all(split)
		
		for thing in things:
			row_data.append(thing)
		
			print(thing.text)
	
		data_rows.append(row_data)

	return data_rows

def write_table(table, fn, split='th'):
	try:
		tbody = table.tbody
	except AttributeError:
		return
	try:
		file = open('.' + m.data + fn + '.csv', 'w')
	except FileExistsError:
		print('skip')
		return

	thead = table.thead
	columns_row = thead.tr
	col_items = columns_row.find_all('th')
	for i, col in enumerate(col_items):
		
		file.write(col.text)

		if i == len(col_items) - 1:
			file.write('\n')
		else:
			file.write(',')

	rows = tbody.find_all('tr')
	for row in rows:
		row_class = row.get('class')
		if row_class is None:  # when the row class is none it is a data row
			row_data = row.find_all(split)
			for i, data_pt in enumerate(row_data):
				file.write(data_pt.text)
				
				if i == len(row_data) - 1:
					file.write('\n')
				else:
					file.write(',')

	print('{} written to {}'.format(fn, m.data))
	file.close()


def prev_day_link(page):
	return page.find('a', {'class' : 'button2 prev'})['href']	


def get_box_links(stop_date='/boxes/?year=2019&month=3&day=28'):	
	start_date = '/boxes/?year=2019&month=5&day=17'
	
	page = of.page(m.url + start_date) 
	prev_day = prev_day_link(page)

	box_links = []

	while prev_day != stop_date:
		date = prev_day.split('?')[1].split('&')  # eg /boxes/?year=2019&month=06&day=1 -> ['year=2019', 'month=06', 'day=1']
		
		year = date[0].split('=')[1]
		prev_year = int(year) - 1
		month = prev_day.split('?')[1].split('&')[0].split('=')[1]
		
		if month == '2':
			prev_day = '/boxes/?year=' + prev_year + '&month=12&day=01'

		links =  page.find_all('td' , {'class': 'right gamelink'})

		box_links += [elt.a['href'] for elt in links]
		page = of.page(m.url + prev_day)
		try:
			prev_day = prev_day_link(page)
		except TypeError:
			print('prev_day not found {}'.format(prev_day))

		print(prev_day)

	return box_links


def write_boxscores(links):
	file = open('tmp', 'w')
	for link in links:
		b = box(link)[0]


def box(url='/boxes/ANA/ANA201905180.shtml'):
	p = of.page(m.url + url)
	cs = comments(p)
	meta = box_meta(p)
	t1_batting = cs[15]
	t2_batting = cs[16]
	pitching = cs[20]
	lineup = cs[25]
	stats = [t1_batting, t2_batting, pitching]
	for i, stat in enumerate(stats):
		stats[i] = bs4.BeautifulSoup(stat, 'html.parser')


def box_meta(url='/boxes/ANA/ANA201905180.shtml'):
	game_id = url.split('/')[3].split('.')[0]

	page = of.page(m.url + url)
	cs = comments(page)

	params = [game_id]
	
	meta = page.find('div', {'class' : 'scorebox_meta'})
	meta_divs = meta.find_all('div')
	for i, div in enumerate(meta_divs):
		
		text = div.text.replace(',', '')
		if i > 5:
			break
		elif i > 0 and  i <= 4:
			
			data = text.split(': ')[1]
			params.append(data)
		else:
			params.append(text)

	other_meta = bs4_parse(cs[21])

	divs = other_meta.find_all('div')

	weather = divs[4].text.replace(',', '')
	weather = weather.split(': ')[1]

	umps = other_meta.div.div.text.split(', ')
	new_umps = []

	for i, elt in enumerate(umps):
	     new_umps.append(elt.split(' - ')[1])

	params += new_umps
	params.append(weather)

	return params


def all_metas():
	
	data_root = '.' + m.data
	
	box_urls = get_box_links()

	for url in box_urls:
		write_meta(url)


def write_meta(url='/boxes/TEX/TEX201905300.shtml'):

	# todo, do i want to overwrite files?

	game_id = url.split('/')[3].split('.')[0].replace(' ', '')
	print(game_id)
	data_root = '.' + m.data
	params = box_meta(url)
	date_split = params[1].split(' ')
	
	month = date_split[1]
	year = date_split[3]

	folder = 'boxes/' + year + '/' + month + '/ '+ game_id + '/'
	path = data_root + folder

	try:
		os.makedirs(path)
	except FileExistsError:
		print('file: {} exists'.format(game_id))
		pass

	fn = params[0] + '_meta.csv'
	file = open(path + fn, 'w')

	for j, list_type in enumerate([m.bb_ref_box_meta, params]):
		for i, field in enumerate(list_type):
			file.write(field)
			if i == len(m.bb_ref_box_meta) - 1:
				file.write('\n')
			else:
				file.write(',')

	file.close()

def batting_parse(bat_comment):
	soup = bs4.BeautifulSoup(bat_comment)
	tbody = soup.table.tbody
	rows = tbody.find_all('tr')
	for row in rows:
		
		row_class = row.get('class')
		
		if row_class == 'spacer':
			continue


def players(letter='a'):
	suffix = '/players/'
	# letter = 'a'
	letters = ['n', 's', 'm']
	player_list = []

	for let in letters:
		url = m.url + suffix + let # ter
		page = of.page(url)
		div_players = page.find('div', {'id' : 'div_players_'})
		players = div_players.find_all('a')
		player_list += players
		print(let)
		# letter = chr(ord(letter) + 1)
		
	return player_list


def player_game_log_urls(player_url='/players/c/cruzne02.shtml'):
	batting = []
	pitching = []
	fielding = []
	
	player_page = of.page(m.url + player_url)

	bot_nav = player_page.find('div', {'id' : 'bottom_nav_container'})
	
	ul_tags = bot_nav.find_all('ul')
	p_tags = bot_nav.find_all('p')

	for p in p_tags:
		if p.text == 'Batting Game Logs':
			batting = ul_tags[p_tags.index(p) - 2].find_all('a')
		if p.text == 'Pitching Game Logs':
			pitching = ul_tags[p_tags.index(p) - 2].find_all('a')
		if p.text == 'Fielding Game Logs':
			fielding = ul_tags[p_tags.index(p) - 2].find_all('a')
	
	for elt in [batting, pitching, fielding]:
		for item in elt:
			elt[elt.index(item)] = item['href']

	return [batting, pitching, fielding]

# find the index in the list of <p> tags which corresponds to pitching, batting, and fielding game logs 
# since the bottom nav container has one extra p tag, subtract one to these relevent ul tags in bot_nav
# find all a tags and append to url 

def game_logs(l='b'):
	all_data = []
	p_list = players(letter=l)

	for i, p in enumerate(p_list):
		url = p['href']
		write_player_history(url)


def player_history(player_url='/players/c/cruzne02.shtml'):
	ids = ['batting_gamelogs', 'pitching_gamelogs', '_0']
	
	# data is a list of 3 lists, where each element is a year of data for the given stats b, p, f respectively
	data = [[], [], []]

	all_years = player_game_log_urls(player_url)

	for i, stat_type in enumerate(all_years):  # each stat type is a list of urls to years
		for year in stat_type:
			try:
				page = of.page(m.url + year)
			except ConnectionError:
				continue
			data[i].append(get_table(page, ids[all_years.index(stat_type)]))

	return data


def url_to_player_id(player_url='/players/c/cruzne02.shtml'):
	almost = player_url.split('/')[3]
	player_id = almost.split('.')[0]
	return player_id


def write_player_history(player_url='/players/c/cruzne02.shtml'):
	types = ['b', 'p', 'f']
	letter_folder = player_url.split('/')[2]
	player_id = url_to_player_id(player_url)

	data_root = '.' + m.data
	folder = 'player_data2/' + letter_folder + '/' + player_id + '/'

	try:
		os.makedirs(data_root + folder)
	except FileExistsError:
		pass
	data = player_history(player_url)
	for i, stat_type in enumerate(data):
		for j, data_table in enumerate(stat_type):
			if data_table is None:
				continue

			year = data_table.caption.text.split(' ')[0]
			fn = folder + player_id + '_' + types[i] + '_' + year
			write_table(data_table, fn, 'td')


def team_links():
	page = of.page(m.url + m.teams_url)
	table = get_table(page, 'teams_active')
	tbody = table.tbody
	rows = tbody.find_all('tr')

	links = []
	for row in rows:
		if row.th == None:
			continue
		a = row.a
		links.append(a['href'])
	return links


def team_season_links(team_link, years=[2010, 2018]):
	urls = []
	tmp = years[0]
	while tmp <= years[1]:
		suffix = team_link + str(tmp) + m.schedule_suffix
		urls.append(suffix)
		tmp += 1
	return urls


def schedule_links(team_links):
	all_schedules = []
	for team_link in team_links:
		all_schedules += team_season_links(team_link)  # list of full urls
	return all_schedules


def game_links(schedule_links):
	links = []
	for schedule in schedule_links:
		full_link = m.url + schedule
		schedule_page = of.page(full_link)
		table = get_table(schedule_page, 'team_schedule')
		try:
			tbody = table.tbody
		except AttributeError:
			continue

		if tbody is None:
			continue

		suffixes = tbody.find_all('td', {'data-stat': 'boxscore'})
		for suffix in suffixes:
			link = suffix.a['href']
			links.append(link)
			print('.', end='')
	return links


def pages(urls):
	pages = []
	for url in urls:
		pages.append(of.page(url))
	return pages


def comments(page):
	comments = page.findAll(text=lambda text:isinstance(text, bs4.Comment))
	return comments


def pbp(boxscore_url):  # given boxscore url, returns pbp table as soup
	page = of.page(boxscore_url)
	comment_list = comments(page)
	pbp_raw = bs4.BeautifulSoup(comment_list[31], 'html.parser')
	pbp = get_table(pbp_raw, 'play_by_play')
	return pbp


def bs4_parse(text):
	return bs4.BeautifulSoup(text, 'html.parser')


def parse_pbp(table, game_id):
	tbody = table.tbody
	rows = tbody.find_all('tr')
	for row in rows:

		row_len = len(row)

		if row.get('id') is None:
			continue

		file.write(game_id + ',')

		for num, item in enumerate(row):
			print(item.text)
			file.write(item.text)
			if num == row_len - 1:
				file.write('\n')
			else:
				file.write(',')


def all_games():
	teams = team_links()
	schedules = schedule_links(teams)
	games = game_links(schedules)
	return games


def main():  # years is list
	file = open('./data/mlb3.csv', 'a+')
	game_links = all_games()
	all_pbp = []
	for i, game in enumerate(game_links):
		game_pbp = pbp(m.url + game)
		if game_pbp is None:
			continue
		parse_pbp(game_pbp, game, file)

	file.close()


def test_parse(url='https://www.baseball-reference.com/boxes/OAK/OAK201903200.shtml'):
	p = pbp(url)
	parse_pbp(p)
