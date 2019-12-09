import numpy as np
import json
import re
import sys


prefix=sys.argv[1]
logfile = open(prefix+"/analysis.log","w")

with open(prefix+"/best_time.log", "r") as f:
    txt_dict = json.load(f)
    best_reward = txt_dict["time"]
    best_strategy = txt_dict["strategy"]
    name_cost_dict = txt_dict["cost"]
print("Time:",best_reward)
logfile.write("Time:{}\n".format(best_reward))

sorted_tuple =list()
for name, cost in name_cost_dict.items():
    sorted_tuple.append((int(cost[0]),name))

sorted_tuple.sort(reverse=True)

#strategy counter for all
counter_dict = dict()
for name,strategy in best_strategy.items():
    if counter_dict.get(str(strategy),None):
        counter_dict[str(strategy)]+=1
    else:
        counter_dict[str(strategy)]=1

for strategy,counter in counter_dict.items():
    print("Strategy:",strategy," Counter:",counter, "Ratio:",float(counter)/len(best_strategy))
    logfile.write("Strategy:{} Counter:{} Ratio:{}\n".format(strategy,counter,float(counter)/len(best_strategy)))



#top 100 operation details
sorted_tuple = sorted_tuple[:100]
for item in sorted_tuple:
    name = item[1]
    cost = name_cost_dict[name]
    strategy = best_strategy[name]
    print("Name:",name," Strategy:",strategy," Cost:",cost)
    logfile.write("Name:{} Strategy:{} Cost:{}\n".format(name,strategy,cost))

logfile.close()