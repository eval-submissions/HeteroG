import csv
import numpy as np
import os
context =dict()
mem_dict = dict()
if os.path.exists("prof.csv"):
    with open('prof.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            context[row[0]] = [float(i) for i in row[2:-1]]

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

with open('prof.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key in context:
        item = [key]+[mem_dict[key]]+context[key]+[np.var(context[key])]
        spamwriter.writerow(item)


