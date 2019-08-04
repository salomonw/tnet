import networkx as nx
from gurobipy import *
from utils import *
import numpy as np
import msa



def find_opt_lane_reversals(G):
	