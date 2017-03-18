import yaml
import os, sys
import csv
import numpy as np
from collections import defaultdict
import pickle

folderloc = "../data/odis/"

data = list()
# headers = ["Date", "Team1", "Team2", "Toss_Winner", "Toss_Decision", "Venue", "Run1", "Over1", "Wicket1", "Run2", "Over2", "Wicket2", "Winner", "Winnerby", "Margin"]
# data.append(headers)

playerlist = defaultdict(dict)
for file in os.listdir(folderloc):
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

	## Innings 1
	fl1ind = file1["innings"][0]["1st innings"]["deliveries"]
	# print fl1ind, len(fl1ind)
	for i in range(len(fl1ind)):
		bat_bowl_pair = (str(fl1ind[i][fl1ind[i].keys()[0]]["batsman"]), str(fl1ind[i][fl1ind[i].keys()[0]]["bowler"]))
		if bat_bowl_pair not in playerlist.keys():
			playerlist[bat_bowl_pair] = 0
		playerlist[bat_bowl_pair] += int(fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"])

	## Innings 2
	fl1ind = file1["innings"][1]["2nd innings"]["deliveries"]
	for i in range(len(fl1ind)):
		bat_bowl_pair = (str(fl1ind[i][fl1ind[i].keys()[0]]["batsman"]), str(fl1ind[i][fl1ind[i].keys()[0]]["bowler"]))
		if bat_bowl_pair not in playerlist.keys():
			playerlist[bat_bowl_pair] = 0
		playerlist[bat_bowl_pair] += int(fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"])
	data.append(playerlist.copy())

with open('../data/player_data.pkl', 'w') as fp:
	pickle.dump(data, fp)

print len(playerlist)