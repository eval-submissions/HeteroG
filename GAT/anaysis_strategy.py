import numpy as np
import json
import re
with open("best_reward.log", "r") as f:
    txt = f.read()
    regex = re.compile(r'\\(?![/u"])')
    txt=regex.sub(r"\\\\", txt)
    txt = txt.replace("'", '"')
    txt_dict = json.loads(txt)
    best_reward = txt_dict["time"]
    best_strategy = txt_dict["strategy"]
    name_cost_dict = txt_dict["cost"]

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



#top 100 operation details
sorted_tuple = sorted_tuple[:100]
for item in sorted_tuple:
    name = item[1]
    cost = name_cost_dict[name]
    strategy = best_strategy[name]
    print("Name:",name," Strategy:",strategy," Cost:",cost)
