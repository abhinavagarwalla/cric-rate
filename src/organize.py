import yaml
import os, sys
import csv

folderloc = "/home/madhav/Downloads/test/"

data = list()
headers = ["Date", "Team1", "Team2", "Winner"]
data.append(headers)

for file in os.listdir(folderloc):
	print file
	match = list()
	filepath = folderloc + file
	with open(filepath, 'r') as stream:
	    try:
	        file1 = yaml.load(stream)
	    except yaml.YAMLError as exc:
	        print(exc)
	# print file1["info"]["teams"]
	match.append(file1["info"]["dates"])
	match.append(file1["info"]["teams"][0])
	match.append(file1["info"]["teams"][1])
	match.append(file1["info"]["outcome"]["winner"])
	data.append(match)

with open('cricket.csv', 'wb') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
	wr.writerows(data)