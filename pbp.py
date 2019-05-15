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


def team_season_links(team_link, years=[2010, 2019]):
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
	file = open('./data/game_links.csv', 'a')
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
			file.write(link + '\n')
			links.append(link)
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
	comments = get_comments(page)
	pbp = bs4.BeautifulSoup(comments[31], 'html.parser')
	return pbp


def all_games():
	teams = team_links()
	schedules = schedule_links(teams)
	games = game_links(schedules)
	return games


def main(years=[2010, 2019]):  # years is list
	teams = team_links()  # list of urls
	schedules = schedule_links(teams, years)
	games = game_links(schedules)
	for link in games:
		full_url = m.url + link
		page = of.page(full_url)


# if __name__ == '__main__':
# 	years = [2010, 2019]
# 	# team = 'ARI'
	
# 	fn = '.' + m.data + 'ARI_BOXSCORE_LINKS' + '.csv'

# 	file = of.file(fn)
# 	urls = team_schedule_urls(years, team)

# 	pages = get_pages(urls)

# 	all_boxscore_urls = []
# 	for page in pages:
# 		urls = get_boxscores(page)
# 		if urls is None:
# 			continue
# 		all_boxscore_urls += urls

# 	for url in all_boxscore_urls:
# 		file.write(url)
# 		file.write('\n')

# file = open('./data/game_links.csv', 'a')

# for link in bls:
# 	file.write(link + '\n')
