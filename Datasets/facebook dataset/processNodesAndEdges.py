import json
circles = ["0", "107", "348", "414", "686", "698", "1684", "1912", "3437", "3980"]
users = {}
attributes = ["concentration", "degree", "school", "hometown", "surname", "location,!work", "employer", "work-location", "work-position"]
for c in circles:
	listOfUsers = []
	#dealing separately with c
	subAttributes = {}
	featfile = open(c+".featnames")
	for a in attributes:
		subAttributes[a] = {}
	for line in featfile:
		line = line.split("\n")[0]
		for a in attributes:
			if "-" in a:
				if a.split("-")[0] in line and a.split("-")[1] in line:
					subAttributes[a][line.split(" ")[3]] = int(line.split(" ")[0])
			elif ",!" in a:
				if a.split(",!")[0] in line and a.split(",!")[1] not in line:
					subAttributes[a][line.split(" ")[3]] = int(line.split(" ")[0])
			else:
				if a in line:
					subAttributes[a][line.split(" ")[3]] = int(line.split(" ")[0])

	print subAttributes
	featVectorsFile = open(c+".feat")
	for line in featVectorsFile:
		line = line.split("\n")[0]
		line = line.split(" ")
		users[line[0]] = {}
		for a in attributes:
			users[line[0]][a] = []
			for x in subAttributes[a]:
				if line[subAttributes[a][x]+1] == "1":
					users[line[0]][a].append(x)
	egoFeatFile = open(c+".egofeat")
	for line in egoFeatFile:
		line = line.split("\n")[0]
		line = line.split(" ")
		users[c] = {}
		print c, line
		for a in attributes:
			users[c][a] = []
			for x in subAttributes[a]:
				if line[subAttributes[a][x]] == "1":
					users[c][a].append(x)


f_users = open("users_facebook.json", "w")
json.dump(users,f_users)
f_users.close()

edges = {}
edgesCombined = open("facebook_combined.txt")
rel = 0
for line in edgesCombined:
	e1 = line.split("\n")[0]
	edges[e1] = []
	relation = []
	line = line.split("\n")[0].split(" ")
	#print line
	for a in users[line[0]]:
		#print a, users[line[0]][a]
		if a in users[line[1]]:
			#print a, users[line[1]][a]
			if (len(list(set(users[line[0]][a]) & set(users[line[1]][a]))) > 0):
				relation.append(a)
	if len(relation) == 0:
		rel+=1
	edges[e1] = relation
f_edges = open("edges_facebook","w")
json.dump(edges,f_edges)
f_edges.close()
print rel


