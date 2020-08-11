import numpy as np
import time
import tensorflow as tf

from copy import copy

from data import get_all_data
from model import Model
from environment import evaluate_logp, evaluate
from utils import save, load

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

records = load("records")

for record in records:
    decisions = [ [1, 7] for _ in range(len(record["cgroups"])) ]
    info(record["best"][1], record["best_single"][1], evaluate(record, decisions))
