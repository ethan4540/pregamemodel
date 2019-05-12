import bs4 
import requests as r
import macros as m
import openers as of


def get_table(page, table_id):  # given bs4 page and table id, finds table using bs4. returns tbody
	table = page.find('table', {'id': table_id})
	return table.tbody


def get_team_links():
	page = of.page(m.url)
	tobdy = get_table(page, 'teams_active')
	rows = tbody.find_all('tr')
	team_names = []
	links = []
	for row in rows:
		if row.th == None:
			continue
		a = row.a
		links.append(a['href'])
	return links


def team_schedule_urls(years, team):
	urls = []
	while years[0] < years[1]:
		full_url = m.url + team + '/' + str(years[0]) + m.games_suffix
		years[0] += 1
		urls.append(full_url)
	return urls

def team_seasons():
	urls = []
	while years[0] < years[1]:
		full_url = m.url + team + '/' + str(years[0]) + m.games_suffix
		years[0] += 1
		urls.append(full_url)
	return urls


def get_pages(urls):
	pages = []
	for url in urls:
		pages.append(of.page(url))
	return pages


def get_comments(page):
	comments = page.findAll(text=lambda text:isinstance(text, bs4.Comment))
	return comments


def get_boxscores(page):
	rows = []
	for row in tbody:
		suffix = row.find('td', {'data-stat': 'boxscore'})
		suffix = suffix.a['href']	
		rows.append(suffix)
	return rows


def pbp(boxscore_url):  # given boxscore url, returns pbp table as soup
	page = get_pages([boxscore_url])
	comments = get_comments(page)
	pbp = bs4.BeautifulSoup(comments[31], 'html.parser')
	return pbp 


years = [2010, 2019]

def main(years):  # years is list
	start_year = years[0]
	teams = get_team_links()  # list of urls
	for team in teams:
		years[0] = start_year
		while years[0] < years[1]
			# TODO
			season = 
			years[0] += 1

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