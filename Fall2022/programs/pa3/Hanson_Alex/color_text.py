"""
Author: Alex Hanson
Date: 11/28/2022

Collection of functions to easily print with ascii color codes.
"""


# Print with color
def printRed(s, e="\n"): print("\033[91m{}\033[00m" .format(str(s)), end=e)
 
def printGreen(s, e="\n"): print("\033[92m{}\033[00m" .format(str(s)), end=e)
 
def printYellow(s, e="\n"): print("\033[93m{}\033[00m" .format(str(s)), end=e)
 
def printLightPurple(s, e="\n"): print("\033[94m{}\033[00m" .format(str(s)), end=e)
 
def printPurple(s, e="\n"): print("\033[95m{}\033[00m" .format(str(s)), end=e)
 
def printCyan(s, e="\n"): print("\033[96m{}\033[00m" .format(str(s)), end=e)
 
def printLightGray(s, e="\n"): print("\033[97m{}\033[00m" .format(str(s)), end=e)
 
def printBlack(s, e="\n"): print("\033[98m{}\033[00m" .format(str(s)), end=e)