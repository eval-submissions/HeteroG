import csv
import numpy as np
import os
import sys
csv_file = sys.argv[1]
context =dict()
mem_dict = dict()
if os.path.exists(csv_file):
    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            context[row[0]] = [float(i) for i in row[2:-1]]
            mem_dict[row[0]] = row[1]

with open("prof.log","r") as f:
    lines = f.readlines()
    for line in lines:
        if "import" not in line:
            continue
        else:
            line = line.strip()
            name = line.split(" ")[0]
            cost = line.split(", ")[-1].split("/")[0]
            mem = line.split(" ")[1].split(", ")[0].split("/")[0][1:]
            if mem_dict.get(name,None)!=None:
                assert(mem_dict[name]==mem)
            mem_dict[name] = mem
            if "us" in cost:
                cost = float(cost[:-2])/1000
            elif "ms" in cost:
                cost = float(cost[:-2])
            elif "sec" in cost:
                cost = float(cost[:-3])*1000
            else:
                raise Exception("ERROR")
            item = context.get(name,[])
            item.append(cost)
            context[name] = item

with open(csv_file, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key in context:
        item = [key]+[mem_dict[key]]+context[key]+[np.var(context[key])]
        spamwriter.writerow(item)


