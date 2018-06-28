import argparse
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, ListedColormap
from matplotlib import rcParams
import colormaps as cmaps
import matplotlib.pyplot as plt
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
import matplotlib.ticker as plticker
from HybridReader2 import HybridReader2 as hr
import spice_tools
from numpy.ma import masked_array
import matplotlib.cm as cm
from scipy.interpolate import griddata
from streamplot import streamplot

import warnings

# Set constant for Pluto radius 
Rp = 1187. # km

def streams(ax, x, y, u, v, *args, **kwargs):
    """Make a streamplot on a non-uniform grid.
    It works by interpolating onto a uniform grid before calling
    the usual streamplot function. Arguments beyond the basic ones
    will be passed along to streamplot."""
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x,y)
    else:
        assert x.ndim == 2
        assert y.ndim == 2

    x_unif = np.linspace(x.min(), x.max(), len(x))
    y_unif = np.linspace(y.min(), y.max(), len(y))

    X_unif, Y_unif = np.meshgrid(x_unif,y_unif)

    px = X.flatten()
    py = Y.flatten()
    pu = u.flatten()
    pv = v.flatten()

    gu = griddata(zip(px,py), pu, (X_unif,Y_unif))
    gv = griddata(zip(px,py), pv, (X_unif,Y_unif))

    return streamplot(ax, x_unif, y_unif, gu, gv, *args, **kwargs)

class CoordType(int):
    """A special integer that lives a double life as a string.
    Used to input coordinates on the command line and automatically
    translate that character into an integer for indexing arrays
    while maintaining the string representation.
    """
    def __new__(cls, c):
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

parser.add_argument('--xlim', type=float, default=None, nargs=2, help='Set the x data limits')
parser.add_argument('--ylim', type=float, default=None, nargs=2, help='Set the y data limits')
parser.add_argument('--zlim', type=float, default=None, nargs=2, help='Set the z data limits')
parser.add_argument('--mccomas', action='store_true', 
    help='Set to arrange the plot in the (-x, transverse) plane instead of the default (x,y) plane')

parser.add_argument('--fontsize', type=float, default=None)
parser.add_argument('--titlesize', type=float, default=25)
parser.add_argument('--labelsize', type=float, default=20)
parser.add_argument('--ticklabelsize', type=float, default=15)
parser.add_argument('--refinement', type=int, default=0)
parser.add_argument('--no-traj', dest='traj', action='store_false')
parser.add_argument('--separate-figures', dest='separate', action='store_true')
parser.add_argument('--title', default=None)
parser.add_argument('--title2', default=None)
parser.add_argument('--units', default='')

def parse_cmd_line():
    args = parser.parse_args()

    if args.fontsize:
        raise RuntimeError("The --fontsize argument is depreciated, use --titlesize, --labelsize, and --ticklabelsize.")

    if args.save is True:
        args.save = str(args.variable)

    if args.title is None:
        args.title = str(args.variable)

    if args.title2 is None:
        args.title2 = args.title

    return args

def init_figures(args):
    if args.separate:
        fig1 = plt.figure()
        fig2 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)

    else:
        fig1, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True,
                figsize=(2*rcParams['figure.figsize'][0], 0.8*rcParams['figure.figsize'][1]))
        fig1.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)
        fig2 = fig1

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

    return fig1, fig2, ax1, ax2

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

    # Shift grid so that Pluto lies at (0,0,0) and convert from km to Rp
    qx = (qx - qx[len(qx)/2 + po])/Rp
    qy = (qy - qy[len(qy)/2])/Rp
    qzrange = (qzrange - qzrange[len(qzrange)/2])/Rp

    infodict = {'px':qx,'py':qy,'pz':qzrange,'cx':cx,'cy':cy,'cz':cz, 'po':po}

    return infodict

def get_next_beta_slice(hn, hT, hB, direction, coordinate=None, depth=None):
    infodict = get_pluto_coords(hn.para)
    n = hn.get_next_timestep()[-1]
    T = hT.get_next_timestep()[-1]
    B = hB.get_next_timestep()[-1]

    # Convert units
    n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
    T = 1.60218e-19 * T                  # eV -> J
    B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T

    # Compute B \cdot B
    B2 = np.sum(B**2, axis=-1)

    # Compute plasma beta
    data = n*T/(B2/(2*1.257e-6))

    if direction == 'xy':
        depth = depth if depth is not None else infodict['cz']
        return data[:,:,depth]

    elif direction == 'xz':
        depth = depth if depth is not None else infodict['cy']
        return data[:,depth,:]

    elif direction == 'yz':
        depth = depth if depth is not None else infodict['cx']
        return data[depth,:,:]

def get_next_slice(h, direction, coordinate=None, depth=None):
    infodict = get_pluto_coords(h.para)
    data = h.get_next_timestep()[-1]
    if not h.isScalar:
        assert coordinate is not None
        data = data[:,:,:,args.variable.coordinate]

    if direction == 'xy':
        depth = depth if depth is not None else infodict['cz']
        return data[:,:,depth]

    elif direction == 'xz':
        depth = depth if depth is not None else infodict['cy']
        return data[:,depth,:]

    elif direction == 'yz':
        depth = depth if depth is not None else infodict['cx']
        return data[depth,:,:]

def plot_setup(ax, data, params, direction, depth, time_coords=False, fontsize=None, mccomas=False, titlesize=25, labelsize=20, ticklabelsize=15):
    infodict = get_pluto_coords(params)
    if direction == 'xy':
        depth = depth if depth is not None else infodict['cz']
        dslice = data[:,:,depth]
        x,y = infodict['px'], infodict['py']
        ax.set_xlabel('X ($R_p$)', fontsize=labelsize)
        ax.set_ylabel('Transverse ($R_p$)' if mccomas else 'Y ($R_p$)', fontsize=labelsize)

    elif direction == 'xz':
        depth = depth if depth is not None else infodict['cy']
        dslice = data[:,depth,:]
        x,y = infodict['px'], infodict['pz']
        ax.set_xlabel('X ($R_p$)', fontsize=labelsize)
        ax.set_ylabel('Z ($R_p$)', fontsize=labelsize)

    elif direction == 'yz':
        depth = depth if depth is not None else infodict['cx']
        dslice = data[depth,:,:]
        x,y = infodict['py'], infodict['pz']
        ax.set_xlabel('Y ($R_p$)', fontsize=labelsize)
        ax.set_ylabel('Z ($R_p$)', fontsize=labelsize)

    else:
        raise ValueError("direction must be one of 'xy', 'xz', or 'yz'")

    if time_coords:
        # Don't pass in mccomas=True since the x variable is always internal coordinates in this function
        # assuming data is coming directly from the simulation.
        x = [spice_tools.time_at_pos(xx*Rp, mccomas=False) for xx in x]
    X,Y = np.meshgrid(x, y)

    if mccomas:
        if direction == 'xy':
            X = -X
            Y = -Y
        elif direction == 'xz':
            X = -X
        elif direction == 'yz':
            X = -X

    ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)

    return X, Y, dslice

def beta_plot(fig, ax, data, params, direction, depth=None, cax=None, fontsize=None, mccomas=False, refinement=0, titlesize=25, labelsize=20, ticklabelsize=15):
    X, Y, dslice = plot_setup(ax, data, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize)

    # Setup custom colorbar
    levels = np.logspace(-1.5,2.5, 5+refinement)
    ticks = np.logspace(-1,2,4)
    # This can be handled by get_cmap in matplotlib v1.5.2 and greater
    # writing out this line this way is to correct a bug in v1.5.1.
    colors = cmaps.viridis(np.linspace(0,1,len(levels)+2))
    cmap = ListedColormap(colors[1:-1],'beta_cmap')
    cmap.set_over(colors[-1])
    cmap.set_bad(colors[0])

    # Catch the stupid warnings I don't care about
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        mappable = ax.contourf(X.T, Y.T, dslice, levels=levels, norm=MyLogNorm(),
                                cmap=cmap, extend='both',
                                vmin=levels[0], vmax=levels[-1])

    if cax != 'None':
        cb = fig.colorbar(mappable, ax=ax, cax=cax)
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticks)

    return mappable

def bs_hi_plot(fig, ax, n_tot, n_h, n_ch4, ux, swspeed, hdensity, params, direction, mccomas=False, depth=None, time_coords=False, fontsize=None, titlesize=25, labelsize=20, ticklabelsize=15, **kwargs):
    """Plot bowshock, plutopause, and heavy ion tail defined as:
    bowshock: >20% slowing of the solar wind (defined explicitly in McComas 2016)
    plutopause: >70% exclusion of H+ (proxy for solar wind particles) (defined indirectly in McComas 2016)
    heavy ion tail: >50% heavy ions (defined indirectly in McComas 2016)
    """
    X, Y, n = plot_setup(ax, n_tot, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize)
    X, Y, h = plot_setup(ax, n_h, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize)
    X, Y, ch4 = plot_setup(ax, n_ch4, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize)
    X, Y, v = plot_setup(ax, ux, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize)

    bs_cont = ax.contourf(X.T, Y.T, v, levels=[-0.8*swspeed, 0], colors='b')
    pp_cont = ax.contourf(X.T, Y.T, h, levels=[0, 0.3*hdensity], colors='m')
    hi_cont = ax.contourf(X.T,Y.T, ch4/n, levels=[.5,1], colors='r')

    return bs_cont, pp_cont, hi_cont

def redblue_density_plot(fig, ax, h, he, ch4, params, direction, red_cax, blue_cax, depth=None, time_coords=False, **kwargs):
    X, Y, h   = plot_setup(ax, h, params, direction, depth, time_coords)
    X, Y, he = plot_setup(ax, he, params, direction, depth, time_coords)
    X, Y, ch4 = plot_setup(ax, ch4, params, direction, depth, time_coords)

    mass_tot = h+4*he+16*ch4

    mtot   = masked_array(mass_tot, mask=16*ch4/mass_tot>0.5)
    mheavy = masked_array(mass_tot, mask=16*ch4/mass_tot<=0.5)

    #b = ax.pcolormesh(X,Y,mtot.transpose(), cmap='Blues', **kwargs)
    #r = ax.pcolormesh(X,Y,mheavy.transpose(), cmap='Reds', **kwargs)

    b = ax.pcolormesh(X,Y,mtot.transpose(), cmap='cool', **kwargs)
    r = ax.pcolormesh(X,Y,mheavy.transpose(), cmap='hot', **kwargs)

    fig.colorbar(b, cax=blue_cax)
    fig.colorbar(r, cax=red_cax, format="")
    return b,r

def redblue_plot(fig, ax, heavy, params, direction, depth=None, time_coords=False):
    X, Y, heavy   = plot_setup(ax, heavy, params, direction, depth, time_coords)

    ratio = np.where(heavy == 0, 0.1, 0.9)

    mappable = ax.pcolormesh(X,Y,ratio.transpose(), cmap='coolwarm', vmin=0, vmax=1)
    return mappable

def direct_plot(fig, ax, data, params, direction, depth=None, cax=None, time_coords=False, fontsize=None, mccomas=False, titlesize=25, labelsize=20, ticklabelsize=15, cbtitle='', **kwargs):
    X, Y, dslice = plot_setup(ax, data, params, direction, depth, time_coords, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize)

    mappable = ax.pcolormesh(X,Y,dslice.transpose(), **kwargs)

    if cax != 'None':
        if cax == None:
            if 'SymLogNorm' in repr(kwargs['norm']):
                cb = fig.colorbar(mappable, ax=ax, shrink=0.7, ticks=plticker.SymmetricalLogLocator(linthresh=0.01, base=10))
            elif 'LogNorm' in repr(kwargs['norm']):
                cb = fig.colorbar(mappable, ax=ax, shrink=0.7, ticks=plticker.LogLocator())
            else:
                cb = fig.colorbar(mappable, ax=ax, shrink=0.7)
        else:
            cb = fig.colorbar(mappable, cax=cax)

        cb.ax.set_title(cbtitle)

    return mappable, X, Y, dslice

def traj_plot(fig, ax, direction, mccomas=False):
    traj, o, times = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 60., mccomas=mccomas)
    traj = traj/1187.

    if direction == 'xy':
        x = traj[:, 0]
        y = traj[:, 1]
    elif direction == 'xz':
        x = traj[:, 0]
        y = traj[:, 2]
    ax.plot(x, y, color='black', linewidth=2, scalex=False, scaley=False)
