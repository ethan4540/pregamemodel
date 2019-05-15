import bs4 
import requests as r
import macros as m
import openers as of


def get_table(page, table_id):  # given bs4 page and table id, finds table using bs4. returns tbody
	table = page.find('table', {'id': table_id})
	return table


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


def parse_pbp(table, game_id, file):
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
