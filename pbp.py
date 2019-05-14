import bs4 
import requests as r
import macros as m
import openers as of


def get_table(page, table_id):  # given bs4 page and table id, finds table using bs4. returns tbody
	table = page.find('table', {'id': table_id})
	return table.tbody


def get_team_links():
	page = of.page(m.url + m.teams_url)
	tbody = get_table(page, 'teams_active')
	rows = tbody.find_all('tr')

	links = []
	for row in rows:
		if row.th == None:
			continue
		a = row.a
		links.append(a['href'])
	# print(links)
	return links


def team_seasons(team_link, years):
	urls = []
	tmp = years[0]
	while tmp <= years[1]:
		full_url = m.url + team_link + str(tmp) + m.schedule_suffix
		urls.append(full_url)
		tmp += 1
	return urls


def get_pages(urls):
	pages = []
	for url in urls:
		pages.append(of.page(url))
	return pages


def get_comments(page):
	comments = page.findAll(text=lambda text:isinstance(text, bs4.Comment))
	return comments


def get_boxscores(schedule_page):
	rows = []
	tbody = get_table(schedule_page, 'team_schedule')
	for row in tbody:
		suffix = row.find('td', {'data-stat': 'boxscore'})
		suffix = suffix.a['href']	
		rows.append(suffix)
	return rows


def pbp(boxscore_url):  # given boxscore url, returns pbp table as soup
	page = of.page(boxscore_url)
	comments = get_comments(page)
	pbp = bs4.BeautifulSoup(comments[31], 'html.parser')
	return pbp 

def get_schedules(team_links, years)
	all_schedules = []
	for team_link in team_links:
		years = tmp_years
		all_schedules += team_seasons(team_link, years)  # list of full urls
	return all_schedules

# years = [2010, 2019]

def main(years=[2010, 2019]):  # years is list
	tmp_years = years
	team_links = get_team_links()  # list of urls
	schedule_links = get_schedules(team_links, years)
	for link in schedule_links:
		boxscores = get_boxscores(
			)

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
