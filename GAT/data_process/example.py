# coding=utf-8
from data_process.dataset import GraphDataset, WhiteSpaceTokenizer,NewWhiteSpaceTokenizer


def load_cora(data_dir,tokenizer):
    return  GraphDataset(data_dir, data_format=GraphDataset.FORMAT_EDGELIST, tokenizer=tokenizer)


def load_M10(data_dir, ignore_featureless_node=True):
    return GraphDataset(data_dir, ignore_featureless_node=ignore_featureless_node)


def load_dblp(data_dir, ignore_featureless_node=True):
    return GraphDataset(data_dir, ignore_featureless_node=ignore_featureless_node)
