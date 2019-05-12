import bs4 
import requests as r

import macros as m
import open_functions as of

def get_teams_links():
	page = of.open_page(m.url)
	table = page.find('table', {'id': 'teams_active'})
	tbody = table.tbody
	links = tbody.find_all('a')

	three_letters = []
	for link in links:
		three_letters.append(link['href'][7:10])

	print(three_letters)



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
		suffix = row.find('td', {'data-stat': 'boxscore'})
		suffix = suffix.a['href']	
		rows.append(suffix)
	return rows


def boxscore(url):
	page = get_pages([url])
	comments = get_comments(page)
	html = bs4.BeautifulSoup(comments[31])


get_teams_links()


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