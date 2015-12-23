import shelve
from box import *

A = [1, 2, 3, 4];
b = {1:1, 2:0, 3:0}

print vars()
boxVariables('here.out', globals());
