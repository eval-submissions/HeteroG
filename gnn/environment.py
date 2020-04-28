from tge import TGE

def evaluate(computation_graph, topo, decisions):
    return 200

    tge = TGE(computation_graph["gdef"], [dev for dev, _ in topo["devices"]])
    tge.custom(strategy)
    tge.fill_batchsize(48)
    tge.use_collective()
    tge.set_bandwidth(intra=topo["intra"], inter=topo["inter"])
    time, mem = tge.evaluate(computation_graph["prof_data"])

    pass
