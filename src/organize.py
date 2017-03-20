import yaml
import os, sys
import csv
import numpy as np
import pickle

folderloc = "../data/odis/"

data = list()
headers = ["Date", "Team1", "Team2", "Toss_Winner", "Toss_Decision", "Venue", "Run1", "Over1", "Wicket1", "Run2", "Over2", "Wicket2", "Winner", "Winnerby", "Margin"]
data.append(headers)
teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,}

feed_list = pickle.load(open('../data/feed_list.pkl'))
for file in feed_list:
	print file
	if 'yaml' not in file:
		continue
	match = list()
	filepath = folderloc + file
	with open(filepath, 'r') as stream:
	    try:
	        file1 = yaml.load(stream)
	    except yaml.YAMLError as exc:
	        print(exc)

	if 'winner' not in file1["info"]["outcome"]:
		continue
	
	if file1["info"]["teams"][0] not in teams_id.keys():
			continue
	if file1["info"]["teams"][1] not in teams_id.keys():
			continue 

	match.append(file1["info"]["dates"][0])
	match.append(file1["info"]["teams"][0])
	match.append(file1["info"]["teams"][1])
	match.append(file1["info"]["toss"]["winner"])
	match.append(file1["info"]["toss"]["decision"])
	match.append(file1["info"]["venue"])
	

	## Innings 1
	fl1ind = file1["innings"][0]["1st innings"]["deliveries"]
	# print fl1ind

	match.append(np.sum([fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"] for i in range(len(fl1ind))]))
	match.append(fl1ind[len(fl1ind)-1].keys()[0])

	## wickets calculation
	bat = [fl1ind[i][fl1ind[i].keys()[0]]["batsman"] for i in range(len(fl1ind))]
	# print bat
	for i in range(len(fl1ind)):
		bat.append(fl1ind[i][fl1ind[i].keys()[0]]["non_striker"])
	bat = np.unique(bat)
	wickets = len(bat)-2
	if len(bat)==11:
		if float(fl1ind[len(fl1ind)-1].keys()[0]) < 50.0:
			wickets = 10
	match.append(wickets)


	## Innings 2
	fl1ind = file1["innings"][1]["2nd innings"]["deliveries"]

	match.append(np.sum([fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"] for i in range(len(fl1ind))]))
	match.append(fl1ind[len(fl1ind)-1].keys()[0])

	## wickets calculation
	bat = [fl1ind[i][fl1ind[i].keys()[0]]["batsman"] for i in range(len(fl1ind))]
	for i in range(len(fl1ind)):
		bat.append(fl1ind[i][fl1ind[i].keys()[0]]["non_striker"])
	bat = np.unique(bat)
	wickets = len(bat)-2
	if len(bat)==11:
		if float(fl1ind[len(fl1ind)-1].keys()[0]) < 50.0:
			wickets = 10
	match.append(wickets)

	if 'winner' in file1["info"]["outcome"]:
		match.append(file1["info"]["outcome"]["winner"])
		match.append(file1["info"]["outcome"]["by"].keys()[0])
		match.append(file1["info"]["outcome"]["by"][file1["info"]["outcome"]["by"].keys()[0]])
	else:
		match.append(0)
		match.append(0)
		match.append(0)

	data.append(match)

with open('../data/cricket.csv', 'wb') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
	wr.writerows(data)
