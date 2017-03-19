import yaml
import os, sys
import csv
import numpy as np
from collections import defaultdict
import pickle
import copy

folderloc = "../data/odis/"

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,}

def save_dict():
	data = list()
	playerlist = defaultdict(dict)
	feed_list = pickle.load(open('../data/feed_list.pkl'))
	for file in feed_list:
		print file
		if 'yaml' not in file:
			continue
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

		## Innings 1
		fl1ind = file1["innings"][0]["1st innings"]["deliveries"]
		for i in range(len(fl1ind)):
			bat_bowl_pair = (str(fl1ind[i][fl1ind[i].keys()[0]]["batsman"]), str(fl1ind[i][fl1ind[i].keys()[0]]["bowler"]))
			if bat_bowl_pair not in playerlist.keys():
				# playerlist[bat_bowl_pair]["total"] = 0
				# playerlist[bat_bowl_pair]["matches"] = 0
				playerlist[bat_bowl_pair] = 0
			# playerlist[bat_bowl_pair]["total"] += int(fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"])
			playerlist[bat_bowl_pair] += int(fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"])
			# playerlist[bat_bowl_pair]["matches"] += 1

		## Innings 2
		fl1ind = file1["innings"][1]["2nd innings"]["deliveries"]
		for i in range(len(fl1ind)):
			bat_bowl_pair = (str(fl1ind[i][fl1ind[i].keys()[0]]["batsman"]), str(fl1ind[i][fl1ind[i].keys()[0]]["bowler"]))
			if bat_bowl_pair not in playerlist.keys():
				# playerlist[bat_bowl_pair]["total"] = 0
				# playerlist[bat_bowl_pair]["matches"] = 0
				playerlist[bat_bowl_pair] = 0
			# playerlist[bat_bowl_pair]["total"] += int(fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"])
			playerlist[bat_bowl_pair] += int(fl1ind[i][fl1ind[i].keys()[0]]["runs"]["total"])
			# playerlist[bat_bowl_pair]["matches"] += 1
		data.append(copy.deepcopy(playerlist))

	with open('../data/player_data.pkl', 'w') as fp:
		pickle.dump(data, fp)

def get_sorted_matches():
	flist = list()
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

		flist.append([file, file1["meta"]["created"]])

	flist = sorted(flist, key=lambda k:k[1])
	print flist
	with open('../data/feed_list.pkl', 'w') as fp:
		pickle.dump([i[0] for i in flist], fp)

# get_sorted_matches()
save_dict()