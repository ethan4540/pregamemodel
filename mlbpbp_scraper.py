import bs4 
import requests as r

import macros as m
import open_functions as of

# to get all boxscore urls

# https://www.baseball-reference.com/teams/ARI/2018-schedule-scores.shtml




def team_schedule_urls(years, team):
	urls = []
	while years[0] < years[1]:
		full_url = m.url + team + '/' + str(years[0]) + m.games_suffix
		years[0] += 1
		urls.append(full_url)
	return urls


def get_pages(urls):
	pages = []
	for url in urls:
		pages.append(of.open_page(url))
	return pages


def get_comments(html):
	comments = soup.findAll(text=lambda text:isinstance(text, Comment))
	return comments


def get_boxscores(page):
	table = page.find('table', {'id' : 'team_schedule'})
	if table is None:
		return None
	try:		
		tbody = table.tbody
	except AttributeError:
		return None

	rows = []
	for row in tbody:
		try:
			suffix = row.find('td', {'data-stat': 'boxscore'})
			try:
				suffix = suffix.a['href']
			except AttributeError:
				continue
		except TypeError:
			continue
		rows.append(suffix)
	return rows


def boxscore(url):
	page = get_pages([url])
	comments = get_comments(page)
	html = bs4.BeautifulSoup(comments[31])





# if __name__ == '__main__':
# 	years = [2010, 2019]
# 	team = 'ARI'
	
# 	fn = '.' + m.data + 'ARI_BOXSCORE_LINKS' + '.csv'

# 	file = of.open_file(fn)
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