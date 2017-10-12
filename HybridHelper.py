import argparse
import numpy as np
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, ListedColormap
import colormaps as cmaps
import matplotlib.pyplot as plt
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
import matplotlib.ticker as plticker
from HybridReader2 import HybridReader2 as hr

import warnings

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
parser.add_argument('-v','--variable', action=VariableAction, dest='variable', required=True,
        help='Name of the variable whose data will be read. For vector quantaties you must provide a coordinate as well.')
parser.add_argument('-p','--prefix', dest='prefix', default='databig', help='Name of the data folder')

parser.add_argument('--colormap', default='viridis', help='Choose a registered colormap for the plot')
parser.add_argument('--save', nargs='?', default=False, const=True, 
        help='Set flag to save instead of displaying. Optionally provide a filename.')

parser.add_argument('--norm', type=LowerString, action=NormAction, default='linear',
                    help='Specify what scale to use and optionally a prameter.')

parser.add_argument('--vmin', type=float, default=None, help='Specify minimum for the colorbar')
parser.add_argument('--vmax', type=float, default=None, help='Specify maximum for the colorbar')

def parse_cmd_line():
    args = parser.parse_args()

    if args.save is True:
        args.save = str(args.variable)+'.png'

    return args

class MyLogNorm(Normalize):
    """
    Normalize a given value to the 0-1 range on a log scale
    """
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        result = np.ma.masked_less_equal(result, 0, copy=False)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin <= 0:
            raise ValueError("values must all be positive")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            # in-place equivalent of above can be much faster
            resdat = result.data
            mask = result.mask
            if mask is np.ma.nomask:
                mask = (resdat <= 0)
            else:
                mask |= resdat <= 0
            np.copyto(resdat, 1, where=mask)
            np.log(resdat, resdat)
            resdat -= np.log(vmin)
            resdat /= (np.log(vmax) - np.log(vmin))
            result = np.ma.array(resdat, mask=mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = np.ma.asarray(value)
            return vmin * np.ma.power((vmax / vmin), val)
        else:
            return vmin * pow((vmax / vmin), value)

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        A = np.ma.masked_less_equal(A, 0, copy=False)
        self.vmin = np.ma.min(A)
        self.vmax = np.ma.max(A)

    def autoscale_None(self, A):
        """autoscale only None-valued vmin or vmax."""
        if self.vmin is not None and self.vmax is not None:
            return
        A = np.ma.masked_less_equal(A, 0, copy=False)
        if self.vmin is None and A.size:
            self.vmin = A.min()
        if self.vmax is None and A.size:
            self.vmax = A.max()

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

def plot_setup(ax, data, params, direction, depth):
    infodict = get_pluto_coords(params)
    if direction == 'xy':
        depth = depth if depth is not None else infodict['cz']
        dslice = data[:,:,depth]
        x,y = infodict['px'], infodict['py']
        ax.set_xlabel('X ($R_p$)')
        ax.set_ylabel('Y ($R_p$)')

    elif direction == 'xz':
        depth = depth if depth is not None else infodict['cy']
        dslice = data[:,depth,:]
        x,y = infodict['px'], infodict['pz']
        ax.set_xlabel('X ($R_p$)')
        ax.set_ylabel('Z ($R_p$)')

    elif direction == 'yz':
        depth = depth if depth is not None else infodict['cx']
        dslice = data[depth,:,:]
        x,y = infodict['py'], infodict['pz']
        ax.set_xlabel('Y ($R_p$)')
        ax.set_ylabel('Z ($R_p$)')

    else:
        raise ValueError("direction must be one of 'xy', 'xz', or 'yz'")

    X,Y = np.meshgrid(x, y)

    ax.set_xlim(x[0],x[-1])
    ax.set_ylim(y[0],y[-1])

    return X, Y, dslice

def beta_plot(fig, ax, data, params, direction, depth=None, cax=None):
    X, Y, dslice = plot_setup(ax, data, params, direction, depth)

    # Setup custom colorbar
    cb_bounds = np.logspace(-1.5,2.5, 9)
    levels = cb_bounds[::2]
    ticks = cb_bounds[1::2]
    colors = plt.cm.get_cmap('viridis',len(levels)+2).colors
    cmap = ListedColormap(colors[1:-1],'beta_cmap')
    cmap.set_over(colors[-1])
    cmap.set_bad(colors[0])

    # Catch the stupid warnings I don't care about
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        mappable = ax.contourf(X.T,Y.T,dslice, levels=levels, norm=MyLogNorm(),
                                cmap=cmap, extend='both',
                                vmin=cb_bounds[0], vmax=cb_bounds[-1])

    if cax != 'None':
        cb = fig.colorbar(mappable, ax=ax, cax=cax)
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticks)

def direct_plot(fig, ax, data, params, direction, depth=None, cax=None, **kwargs):
    X, Y, dslice = plot_setup(ax, data, params, direction, depth)

    mappable = ax.pcolormesh(X,Y,dslice.transpose(), **kwargs)

    if cax != 'None':
        cb = fig.colorbar(mappable, ax=ax, cax=cax)
