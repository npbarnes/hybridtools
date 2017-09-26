import argparse
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from HybridReader2 import HybridReader2 as hr

class CoordType(int):
    """A special integer that lives a double life as a string.
    Used to input coordinates on the command line and automatically 
    translate that character into an integer for indexing arrays
    while maintaining the string representation.
    """
    def __new__(cls,c):
        if c == 'x':
            return super(CoordType, cls).__new__(cls, 0)
        elif c == 'y':
            return super(CoordType, cls).__new__(cls, 1)
        elif c == 'z':
            return super(CoordType, cls).__new__(cls, 2)
        else:
            raise ValueError("Coordinate must be one of 'x', 'y', or 'z'")

    def __repr__(self):
        if self == 0:
            return "CoordType('x')"
        elif self == 1:
            return "CoordType('y')"
        elif self == 2:
            return "CoordType('z')"
        else:
            raise ValueError

    def __str__(self):
        if self == 0:
            return 'x'
        elif self == 1:
            return 'y'
        elif self == 2:
            return 'z'
        else:
            raise ValueError

def LowerString(string):
    return string.lower()

class Variable:
    def __init__(self, name, coordinate=None):
        self.name = name
        self.coordinate = CoordType(coordinate) if coordinate is not None else None

    def __repr__(self):
        if self.coordinate is not None:
            return 'Variable(' + self.name + ',' + str(self.coordinate) + ')'
        else:
            return 'Variable(' + self.name + ')'

    def __str__(self):
        if self.coordinate is not None:
            return self.name + ' ' + str(self.coordinate)
        else:
            return self.name

class VariableAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed with VariableAction')
        super(VariableAction, self).__init__(option_strings, dest, nargs='+', **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1:
            setattr(namespace, self.dest, Variable(values[0]))
        elif len(values) == 2:
            setattr(namespace, self.dest, Variable(values[0], values[1]))
        else:
            raise argparse.ArgumentError(option_string, 'There must be exactly one or two values consumed')

class NormAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed with NormAction')
        super(NormAction, self).__init__(option_strings, dest, nargs='+', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values[0] == 'linear':
            if len(values) != 1:
                raise argparse.ArgumentError(option_string, 'Options for the linear norm are not supported')
            setattr(namespace, self.dest, None)

        elif values[0] == 'log':
            if len(values) != 1:
                raise argparse.ArgumentError(option_string, 'Options for the log norm are not supported')
            setattr(namespace, self.dest, LogNorm())

        elif values[0] == 'symlog':
            if len(values) > 2:
                raise argparse.ArgumentError(option_string, 
                        'Optionally specify the linear range, otherwise a default will be used')
            if len(values) == 2:
                setattr(namespace, self.dest, SymLogNorm(int(values[1])))
            elif len(values) == 1:
                setattr(namespace, self.dest, SymLogNorm(0.001))

        else:
            raise argparse.ArgumentError(option_string, 'Choose between linear, log, or symlog')



parser = argparse.ArgumentParser()
parser.add_argument('-p','--prefix', dest='prefix', default='databig', help='Name of the datafolder')
parser.add_argument('-v','--variable', action=VariableAction, dest='variable', required=True,
        help='Name of the variable whose data will be read')

parser.add_argument('--colormap', default='viridis', help='Choose a colormap for the plot')
parser.add_argument('--save', nargs='?', default=False, const=True, 
        help='Set flag to save instead of displaying. Optionally provide a filename.')

parser.add_argument('--norm', type=LowerString, action=NormAction, default='linear',
                    help='Specify what scale to use')

def get_pluto_coords(para):
    # Get grid spacing
    qx = para['qx']
    qy = para['qy']
    qzrange = para['qzrange']

    # Find the center index of the grid
    cx = para['nx']/2
    cy = para['ny']/2
    cz = para['zrange']/2

    # the offset of pluto from the center isn't always availible
    try:
        po = para['pluto_offset']
    except KeyError:
        print("Couldn't get pluto_offset. It has been assumed to be 30, but it probably isn't.")
        po = 30

    # Set constatnt for Pluto radius 
    Rp = 1187. # km

    # Shift grid so that Pluto lies at (0,0,0) and convert from km to Rp
    qx = (qx - qx[len(qx)/2 + po])/Rp
    qy = (qy - qy[len(qy)/2])/Rp
    qzrange = (qzrange - qzrange[len(qzrange)/2])/Rp

    infodict = {'px':qx,'py':qy,'pz':qzrange,'cx':cx,'cy':cy,'cz':cz, 'po':po}

    return infodict

