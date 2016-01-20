# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# January 2016

""" 
"""
import argparse
import sys

def __init__(self, expressionData, patientLabels):
    """
    """
    self.expressionData = expressionData
    self.patientLabels = patientLabels

    
def createGlobalNetwork(self, threshold):
    """ Create the global (population-wise) co-expression network by computing Pearson's correlation
    over:
    - the whole population
    - the positive samples
    - the negative samples
    and then thresholding.
    An edge is kept if its weight (i.e. Pearson's correlation between the expression of the 2 genes)
    is greater than the threshold in any of the three networks.

    Store network in self.globalNetwork
    """


def checkScaleFree(self):
    """ Compute the scale-free criteria (Zhang et al., 2005) for the global network.
    Also plots the regression line.
    """


def normalizeExpressionData(self):
    """ Normalize self.expressionData so that each gene has a mean of 0
    and a standard deviation of 1.
    """


def createSamSpecLioness(self):
    """ Create sample-specific co-expression networks,
    using the LIONESS approach.
    """
    

def createSamSpecRegline(self):
    """ Create sample-specific co-expression networks,
    using the distance to the regression line.
    """

    
def main():
    """ Build sample-specific co-expression networks.

    Example:
        $ python CoExpressionNetwork.py 

    
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("name", help="Dataset name")
    parser.add_argument("out_dir", help="Where to store generated networks")

    

if __name__ == "__main__":
    main()
