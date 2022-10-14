from ast import arg
import os
import numpy as np
import math
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a',type=float)
    parser.add_argument('--b',type=float)
    parser.add_argument('--c',type=float)
    args = parser.parse_args()
    a = args.a
    b = args.b
    c = args.c
    print(a,b,c)