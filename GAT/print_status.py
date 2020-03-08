import os
import json
config_dict =dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)

inputs =config_dict.get("inputs",None)
if inputs:
    graph_num =len(inputs)
else:
    graph_num=7

def print_status():
    with open("time.log", "w") as f:
        for i in range(1, graph_num + 1):
            if os.path.exists("data/graph" + str(i) + "/time.log"):
                with open("data/graph" + str(i) + "/time.log", "r") as g:
                    txt = g.read()
                    txt = txt.strip()
                    txt = txt.split(",")
                    txt = txt[-min(10, len(txt)):]
                    f.write("learning rate:{} graph:{}, time:{}\n".format(config_dict["learning_rate"], i, str(txt)))
    with open("best_time.log", "w") as f:
        for i in range(1, graph_num + 1):
            if os.path.exists( "data/graph" + str(i) + "/best_time.log"):
                with open("data/graph" + str(i) + "/best_time.log", "r") as g:
                    txt = json.load(g)
                    time = txt["time"]
                    f.write("learning rate:{} graph:{}, best_time:{}\n".format(config_dict["learning_rate"], i, time))


print_status()