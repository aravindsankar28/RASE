import json
import sys
user_attributes = {} 
# First indexed by the users, then indexed by attribute name and then has the list of attributes.

attributes = ["college", "employer", "location"]


for attribute in attributes:
	with open('EgoNetUIUC-LinkedinCrawl-Profile0101/'+attribute+'.txt') as f:
		lines = f.read().splitlines()
		i = 0
		while i < len(lines):
			user = lines[i]
			user = int(user[1:len(user)])
			i += 1
			num_attrs = int(lines[i])
			i +=1
			if user not in user_attributes:
				user_attributes[user] = {}
			for j in range(0, num_attrs):
				attr = lines[i]
				i += 1
				if attribute not in user_attributes[user]:
					user_attributes[user][attribute] = []
				user_attributes[user][attribute].append(attr)


print len(user_attributes)

network = {}

edges = set()
with open('EgoNetUIUC-LinkedinCrawl-Network_20150101/network.txt') as f:
	lines = f.read().splitlines()
	for line in lines:
		sp = line.split()
		a = int(sp[1][1:len(sp[1])])
		b = int(sp[2][1:len(sp[2])])
		if str(b)+" "+str(a) not in edges:
			edges.add(str(a)+" "+str(b))


# Read network.
count = 0
for line in edges:
	sp = line.split()
	a = int(sp[0])
	b = int(sp[1])
	edge_attributes = []
	if a not in user_attributes or b not in user_attributes:
		network[str(a)+" "+str(b)] = []
		continue

	for attribute in user_attributes[a]:
		a_attr_values = set(user_attributes[a][attribute])
		b_attr_values = set()

		if attribute in user_attributes[b]:
			b_attr_values = set(user_attributes[b][attribute])
		if len(a_attr_values & b_attr_values) > 0:
			edge_attributes.append(attribute)
	if str(b)+" "+str(a) not in network:
		network[str(a)+" "+str(b)] = edge_attributes
	if len(network[str(a)+" "+str(b)]) == 0:
		count += 1

print count, len(edges)

# Find number of links of each type
attributes_count = {}
for attribute in attributes:
	attributes_count[attribute] = 0

for edge in network:
	for att in network[edge]:
		attributes_count[att] += 1

with open('network.txt', 'w') as outfile:
    json.dump(network, outfile)

with open('user_attributes.txt', 'w') as outfile:
	string = json.dumps(user_attributes, encoding='latin1')
	outfile.write(string)
    #json.dump(user_attributes.decode('latin-1'), outfile)



with open('network_edgelist.txt','w') as outfile:	
	for edge in network:
		a = edge.split()[0]
		b = edge.split()[1]
		outfile.write(str(a)+" "+str(b)+"\n")


