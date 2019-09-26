import pickle
import os
import csv
import json
import subprocess
import numpy as np
import time
from functools import update_wrapper

proto = pickle.HIGHEST_PROTOCOL

def zdump(obj, f_name):
    """
    save and compress an object with pickle.HIGHEST_PROTOCOL

    Parameters
    ----------
    obj: python object
    f_name: file name used to save compressed file 

    Returns
    -------
    A saved file

    """
    f = gzip.open(f_name,'wb', proto)
    pickle.dump(obj,f)
    f.close()

def zload(f_name):
	"""
    load compressed object with pickle.HIGHEST_PROTOCOL

    Parameters
    ----------
    obj: python object
    f_name: file name used to save compressed file 

    Returns
    -------
    the object in the file

    """
	f = gzip.open(f_name,'rb', proto)
	obj = pickle.load(f)
	f.close()
	return obj

def dump(obj, f_name):
    """
    save an object with pickle.HIGHEST_PROTOCOL

    Parameters
    ----------
    obj: python object
    f_name: file name used to save compressed file 

    Returns
    -------
    A saved file

    """
    f = open(f_name,'wb', proto)
    pickle.dump(obj,f)
    f.close()

def load(f_name):
	"""
    load file with pickle.HIGHEST_PROTOCOL

    Parameters
    ----------
    f_name: file name used to save compressed file 

    Returns
    -------
    the object in the file

    """
	f = open(f_name,'rb', proto)
	obj = pickle.load(f)
	f.close()
	return obj

def mkdir_n(dirName):
	"""
    Function to create directorry in case it does not exist

    Parameters
    ----------
    f_name: file name used to save compressed file 

    Returns
    -------
    the object in the file

    """
	if os.path.isdir(dirName) == False:
		os.mkdir(dirName)


def csv2dict(fname):
	"""
    read a csv fiel to a dict when kays and values are
    separate by a comma

    Parameters
    ----------
    fname: csv fname with two columns separated by a comma

    Returns
    -------
    a python dictionary

    """
	reader = csv.reader(open(fname, 'r'))
	d = {}
	for row in reader:
		k, v = row
		d[k] = v
	return d


def e_vect(n, i):
    """
    get a vector of zeros with a one in the i-th position

    Parameters
    ----------
    n: vector length 
    i: position

    Returns
    -------
    an array with zeros and 1 in the i-th position

    """    
    zeros = [0 for n_i in range(n)]
    zeros[i] = 1
    return zeros

def dict2json(dict_, fname):
    with open(fname, 'w') as fp:
        json.dump(dict_, fp)


def shell(command, printOut=True):
    """
    Run shell commands in Linux, decide if printing or not the output in console

    Parameters
    ----------
    command: text command
    printOut: decide if print output or not

    Returns
    -------
    None

    """   
    if printOut == False:   
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=None)
    else:
        proc = subprocess.Popen(command, shell=True)
        proc.wait(timeout=None)


def write_file(string, fname):
    f = open(fname, 'w')
    f.write(string)
    f.close()

def read_vector(fname, delimiter):
    x = np.loadtxt(fname, delimiter=delimiter)


def decorator(d):
    "Make function d a decorator: d wraps a function fn."
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d


@decorator
def timeit(f):
    """time a function, used as decorator"""
    def new_f(*args, **kwargs):
        bt = time.time()
        r = f(*args, **kwargs)
        et = time.time()
       # print("time spent on {0}: {1:.2f}s".format(f.__name__, et - bt))
        return r
    return new_f