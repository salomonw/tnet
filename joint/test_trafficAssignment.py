import sys
from Joint.utils import *
import Joint.tnet as tnet
import os 
import trafficAssignment.Convex_Combination as ta
import testing

netFile = "../networks/berlin-tiergarten_net.txt"
gFile = "../networks/berlin-tiergarten_trips.txt"

fcoeffs_truth = [1,0,0,0,0.20,0]
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)

#testing.
#print(tNet.g)