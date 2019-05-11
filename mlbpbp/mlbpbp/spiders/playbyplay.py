import scrapy



class MlbSpider(scrapy.Spider):
    name = "baila"

    def start_requests(self):
        homeurl = [
            'http://www.espn.com/mlb/team/_/name/nyy/new-york-yankees'

        ]
        awayurl = ['http://www.espn.com/mlb/team/_/name/mil/milwaukee-brewers']
        for url in homeurl:
            yield scrapy.Request(url=url, callback=self.homeparse)

        for url in awayurl:
        	yield scrapy.Request(url=url, callback = self.awayparse)

    def homeparse(self, response):
    	links = response.xpath('//@href').extract()
    	for item in links:
    		if 'splits' in item:
    			splitslink = item

    	splitslink = 'http://www.espn.com' + splitslink
    	# print(splitslink)
    	# print('yolo')
    	urls = [splitslink]
    	x = response.css('div.game-meta > div ::text').extract()
    	y = response.css('div.game-info ::text').extract()
    	y = y[1:]
    	# print(x)
    	# print(y)
    	# print(len(y))
    	# print(len(x))
    	h = len(x)
    	gameresults = []
    	for i in range(h):
    		if x[i] == 'W' or x[i] == 'L':
    			gameresult = [x[i], x[i+1]]
    			gameresults.append(gameresult)

    	# print(gameresults)
    	# print(len(gameresults))
    	fullgames = []
    	for i in range(len(gameresults)):
    		if 'vs' not in y[i]:
    			fullgame = ['A', gameresults[i]]

    		if 'vs' in y[i]:
    			fullgame = ['H', gameresults[i]]

    		fullgames.append(fullgame)
    	homegames = []
    	awaygames = []
    	for i in range(len(fullgames)):
    		if fullgames[i][0] == 'H':
    			homegames.append(fullgames[i])

    		if fullgames[i][0] == 'A':
    			awaygames.append(fullgames[i])

    	# print(fullgames)
    	# print(len(homegames))
    	# print(len(awaygames))
    	l10wins = 0
 
    	l10winper = 0
    	l8hwins = 0
    	l8hwinper = 0
    	l8awins = 0
    	l8awinper = 0
    	l10scored = 0
    	l10allowed = 0
    	l8hscored = 0
    	l8hallowed = 0
    	l8ascored = 0
    	l8aallowed = 0
    	for i in range(10):
    		if fullgames[i][1][0] == 'W':
    			l10wins +=1
    			x = fullgames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[0])
    			runsallowed = int(y[1])
    			l10allowed += runsallowed
    			l10scored += runscored

    		if fullgames[i][1][0] == 'L':
    			x = fullgames[i][1][1]
    			y = x.split('-')

    			
    			runscored = int(y[1])
    			runsallowed = int(y[0])
    			l10scored +=runscored
    			l10allowed += runsallowed

    	l10runpergame = l10scored/10
    	l10allowedper = l10allowed/10
    	l10winper = l10wins/10

    	for i in range(8):
    		if homegames[i][1][0] == 'W':
    			l8hwins +=1
    			x = homegames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[0])
    			runsallowed = int(y[1])
    			l8hscored += runscored
    			l8hallowed += runsallowed

    		if homegames[i][1][0] == 'L':
    			x = homegames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[1])
    			runsallowed = int(y[0])
    			l8hscored += runscored
    			l8hallowed += runsallowed
    	l8hwinper = l8hwins/8
    	l8hrunper = l8hscored/8
    	l8hallowedper = l8hallowed/8

    	for i in range(8):
    		if awaygames[i][1][0] == 'W':
    			l8awins +=1
    			x = awaygames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[0])
    			runsallowed = int(y[1])
    			l8ascored += runscored
    			l8aallowed += runsallowed

    		if awaygames[i][1][0] == 'L':
    			x = awaygames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[1])
    			runsallowed = int(y[0])
    			l8ascored += runscored
    			l8aallowed += runsallowed
    	l8awinper = l8awins/8
    	l8arunper = l8ascored/8
    	l8aallowedper = l8aallowed/8






    	#print(l8awinper)

    	print(l8hwinper)
    	print('above is the home teams last 8 at home win %')
    	print(l10winper)
    	print('above is the home teams last 10 games win %')
    	print(l10runpergame)
    	print('above is the home teams last 10 runs per game')
    	print(l10allowedper)
    	print('above is the home teams last 10 allowed per game')
    	print(l8hrunper)
    	print('above is the home teams runs per game in their last 8 home games')
    	print(l8hallowedper)
    	print('above is the home teams runs allowed per game in their last 8 home games')
    	# print(l8arunper)
    	# print(l8aallowedper)
    	for url in urls:
    		yield scrapy.Request(url=url, callback=self.homesplits)

    def homesplits(self, response):
    	x = response.css('tr.Table2__tr.Table2__tr--sm.Table2__even ::text').extract()
    	# print(x)

    	totalgames = int(x[49])
    	totalwins = int(x[50])
    	averagerpg = int(x[53])/totalgames
    	homegames = int(x[63])
    	homewins = int(x[64])
    	averagehrpg = int(x[67])/homegames
    	awaygames = int(x[77])
    	awaywins = int(x[78])
    	averagearpg = int(x[81])/ awaygames
    	totalwinper = totalwins/totalgames
    	homewinper = homewins/homegames
    	awaywinper = awaywins/awaygames
    	print(averagerpg)
    	print('above is the home teams runs per game whole season')
    	print(averagehrpg)
    	print('above is the ohme teams run per game for all their home teams')
    	print(totalwinper)
    	print('above is the home teams winning percentage on the season')
    	print(homewinper)
    	print('above is the home teams winning percentage at home on the season')

    	links = response.xpath('//@href').extract()
    	for link in links:
    		if 'pitching' in link:
    			newlink = link

    	pitchinglink = 'http://www.espn.com' + newlink
    	yield scrapy.Request(url=pitchinglink, callback=self.homepitching)

    def homepitching(self, response):
    	x = response.css('tr.Table2__tr.Table2__tr--sm.Table2__even ::text').extract()
    	totalera = float(x[45])
    	homera = float(x[59])
    	awayera = float(x[73])
    	print(totalera)
    	print('above is the home teams average runs allowed on the year')
    	print(homera)
    	print('above is the home teams era at home on the year')



    def awayparse(self, response):
    	links = response.xpath('//@href').extract()
    	for item in links:
    		if 'splits' in item:
    			splitslink = item

    	splitslink = 'http://www.espn.com' + splitslink
    	# print(splitslink)
    	# print('yolo')
    	urls = [splitslink]
    	x = response.css('div.game-meta > div ::text').extract()
    	y = response.css('div.game-info ::text').extract()
    	y = y[1:]
    	# print(x)
    	# print(y)
    	# print(len(y))
    	# print(len(x))
    	h = len(x)
    	gameresults = []
    	for i in range(h):
    		if x[i] == 'W' or x[i] == 'L':
    			gameresult = [x[i], x[i+1]]
    			gameresults.append(gameresult)

    	# print(gameresults)
    	# print(len(gameresults))
    	fullgames = []
    	for i in range(len(gameresults)):
    		if 'vs' not in y[i]:
    			fullgame = ['A', gameresults[i]]

    		if 'vs' in y[i]:
    			fullgame = ['H', gameresults[i]]

    		fullgames.append(fullgame)
    	homegames = []
    	awaygames = []
    	for i in range(len(fullgames)):
    		if fullgames[i][0] == 'H':
    			homegames.append(fullgames[i])

    		if fullgames[i][0] == 'A':
    			awaygames.append(fullgames[i])

    	# print(fullgames)
    	# print(len(homegames))
    	# print(len(awaygames))
    	l10wins = 0
 
    	l10winper = 0
    	l8hwins = 0
    	l8hwinper = 0
    	l8awins = 0
    	l8awinper = 0
    	l10scored = 0
    	l10allowed = 0
    	l8hscored = 0
    	l8hallowed = 0
    	l8ascored = 0
    	l8aallowed = 0
    	for i in range(10):
    		if fullgames[i][1][0] == 'W':
    			l10wins +=1
    			x = fullgames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[0])
    			runsallowed = int(y[1])
    			l10allowed += runsallowed
    			l10scored += runscored

    		if fullgames[i][1][0] == 'L':
    			x = fullgames[i][1][1]
    			y = x.split('-')

    			
    			runscored = int(y[1])
    			runsallowed = int(y[0])
    			l10scored +=runscored
    			l10allowed += runsallowed

    	l10runpergame = l10scored/10
    	l10allowedper = l10allowed/10
    	l10winper = l10wins/10

    	for i in range(8):
    		if homegames[i][1][0] == 'W':
    			l8hwins +=1
    			x = homegames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[0])
    			runsallowed = int(y[1])
    			l8hscored += runscored
    			l8hallowed += runsallowed

    		if homegames[i][1][0] == 'L':
    			x = homegames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[1])
    			runsallowed = int(y[0])
    			l8hscored += runscored
    			l8hallowed += runsallowed
    	l8hwinper = l8hwins/8
    	l8hrunper = l8hscored/8
    	l8hallowedper = l8hallowed/8

    	for i in range(8):
    		if awaygames[i][1][0] == 'W':
    			l8awins +=1
    			x = awaygames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[0])
    			runsallowed = int(y[1])
    			l8ascored += runscored
    			l8aallowed += runsallowed

    		if awaygames[i][1][0] == 'L':
    			x = awaygames[i][1][1]
    			y = x.split('-')
    			runscored = int(y[1])
    			runsallowed = int(y[0])
    			l8ascored += runscored
    			l8aallowed += runsallowed
    	l8awinper = l8awins/8
    	l8arunper = l8ascored/8
    	l8aallowedper = l8aallowed/8






    	print(l8awinper)
    	print('above is the away teams last 8 winning percentage on the road')
    	# print(l8hwinper)
    	print(l10winper)
    	print('above is the away teams winning percentage in their last 10 games')
    	print(l10runpergame)
    	print('above is the away teams average runs per game in their last 10')
    	print(l10allowedper)
    	print('above is the away teams average runs allowed in their last 10 games')
    	#print(l8hrunper)
    	# print(l8hallowedper)

    	print(l8arunper)
    	print('above is the away teams average runs per game in their last 8 away')
    	print(l8aallowedper)
    	print('above is the away teams average runs allowed in their last 8')
    	for url in urls:
    		yield scrapy.Request(url=url, callback=self.awaysplits)

    def awaysplits(self, response):
    	x = response.css('tr.Table2__tr.Table2__tr--sm.Table2__even ::text').extract()
    	# print(x)

    	totalgames = int(x[49])
    	totalwins = int(x[50])
    	averagerpg = int(x[53])/totalgames
    	homegames = int(x[63])
    	homewins = int(x[64])
    	averagehrpg = int(x[67])/homegames
    	awaygames = int(x[77])
    	awaywins = int(x[78])
    	averagearpg = int(x[81])/ awaygames
    	totalwinper = totalwins/totalgames
    	homewinper = homewins/homegames
    	awaywinper = awaywins/awaygames
    	print(averagerpg)
    	print('above is the away teams average runs per game on the year')

    	print(averagearpg)
    	print('above is the away teams run per game for all their away games')
    	print(totalwinper)
    	print('above is the away teams winning percentage on the season')
    	print(awaywinper)
    	print('above is the away teams winning percentage on the raod on the season')

    	links = response.xpath('//@href').extract()
    	for link in links:
    		if 'pitching' in link:
    			newlink = link

    	pitchinglink = 'http://www.espn.com' + newlink
    	yield scrapy.Request(url=pitchinglink, callback=self.awaypitching)

    def awaypitching(self, response):
    	x = response.css('tr.Table2__tr.Table2__tr--sm.Table2__even ::text').extract()
    	totalera = float(x[45])
    	homera = float(x[59])
    	awayera = float(x[73])
    	print(totalera)
    	print('above is the away teams era on the year')
    	print(awayera)
    	print('above is the away teams era on the road on the year')

    	

