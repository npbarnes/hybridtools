import argparse
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, CenteredNorm, ListedColormap
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from HybridReader2 import HybridReader2 as hr
import spice_tools
from numpy.ma import masked_array
import matplotlib.cm as cm
from scipy.interpolate import griddata
from streamplot import streamplot

import warnings

# Set constant for Pluto radius 
#Rp = 1187. # km
SCALE = 1. # km

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
        norm_args = [float(v) for v in values[1:]]
        if values[0] == 'linear':
            setattr(namespace, self.dest, Normalize(*norm_args))
        elif values[0] == 'log':
            setattr(namespace, self.dest, LogNorm(*norm_args))
        elif values[0] == 'symlog':
            setattr(namespace, self.dest, SymLogNorm(*norm_args))
        elif values[0] == 'centered':
            setattr(namespace, self.dest, CenteredNorm(*norm_args))
        else:
            raise argparse.ArgumentError(option_string, 'Choose between linear, log, symlog, and centered')

def limittype(s):
    if s == 'auto':
        return None
    else:
        return float(s)

parser = argparse.ArgumentParser()
parser.add_argument('-v','--variable', action=VariableAction, dest='variable', required=True,
        help='Name of the variable whose data will be read. For vector quantaties you must provide a coordinate as well.')
parser.add_argument('-p','--prefix', dest='prefix', default='data', help='Name of the data folder')

parser.add_argument('--colormap', default='viridis', help='Choose a registered colormap for the plot')
parser.add_argument('--save', nargs='?', default=False, const=True, 
        help='Set flag to save instead of displaying. Optionally provide a filename.')

parser.add_argument('--norm', type=LowerString, action=NormAction, default=None,
                    help='Specify what scale to use and optionally a prameter.')

parser.add_argument('--vmin', type=float, default=None, help='Specify minimum for the colorbar')
parser.add_argument('--vmax', type=float, default=None, help='Specify maximum for the colorbar')

parser.add_argument('--xlim', type=limittype, default=None, nargs=2, help='Set the x data limits')
parser.add_argument('--ylim', type=limittype, default=None, nargs=2, help='Set the y data limits')
parser.add_argument('--zlim', type=limittype, default=None, nargs=2, help='Set the z data limits')
parser.add_argument('--mccomas', action='store_true', 
    help='Set to arrange the plot in the (-x, transverse) plane instead of the default (x,y) plane')

parser.add_argument('--titlesize', type=float, default=25)
parser.add_argument('--labelsize', type=float, default=20)
parser.add_argument('--ticklabelsize', type=float, default=15)
parser.add_argument('--refinement', type=int, default=0)
parser.add_argument('--traj', dest='traj', action='store_true')
#parser.add_argument('--separate-figures', dest='separate', action='store_true')
parser.add_argument('--xy', action='store_true', default=None)
parser.add_argument('--xz', action='store_true', default=None)
parser.add_argument('--yz', action='store_true', default=None)
parser.add_argument('--title', default=None)
parser.add_argument('--title2', default=None)
parser.add_argument('--units', default='')
parser.add_argument('--style', help='Matplotlib style to use for the plot')
parser.add_argument('-s','--step', dest='stepnum', type=int, default=-1,
        help='The specific step number to read. Negative numbers count from the end')
parser.add_argument('--no-aspect', dest='equal_aspect', action='store_false')

parser.add_argument('--force-version', type=int, default=None)
parser.add_argument('--scale-factor', type=float, default=1.0)

def get_pcolormesh_args(mesh):
    x = mesh._coordinates[0,:,0]
    y = mesh._coordinates[:,0,1]
    c = mesh.get_array().reshape(len(y)-1,len(x)-1)

    return x,y,c

def build_pcolormesh_format_coord(mesh):
    x,y,c = get_pcolormesh_args(mesh)
    return build_format_coord(x,y,c)

def build_format_coord(xx,yy,C):
    def format_coord(x,y):
        nocolor = "x={0:1.4f}, y={1:1.4f}".format(x, y)

        if xx.ndim == 2:
            X = xx[0,:]
        else:
            X = xx

        if yy.ndim == 2:
            Y = yy[:,0]
        else:
            Y = yy

        if X[0] < X[-1]:
            col = np.searchsorted(X, x)
        else:
            col = X.size - np.searchsorted(X[::-1], x, side='right')

        if Y[0] < Y[-1]:
            row = np.searchsorted(Y, y)
        else:
            row = Y.size - np.searchsorted(Y[::-1], y, side='right')

        # I don't know why this line needs to be here.
        # I was off by one and this fixes it.
        row -= 1
        col -= 1
        if row < 0 or col < 0 or row >= C.shape[0] or col >= C.shape[1]:
            return nocolor

        return nocolor+", color={0:.4e}".format(C[row,col])

    return format_coord

def parse_cmd_line():
    args = parser.parse_args()
    global SCALE
    SCALE = args.scale_factor


    if args.save is True:
        args.save = str(args.variable)

    if args.style:
        plt.style.use(args.style)

    if args.title is None:
        args.title = str(args.variable)

    if args.title2 is None:
        args.title2 = args.title

    args.directions = []
    if args.xy:
        args.directions.append('xy')
    if args.xz:
        args.directions.append('xz')
    if args.yz:
        args.directions.append('yz')

    if len(args.directions) == 0:
        args.xy = True
        args.xz = True
        args.directions = ['xy','xz']
    elif len(args.directions) == 3:
        print("Warning: hvar will only plot the first two directions")

    return args

def init_figures(args):
    figs = []
    axs = []
    for d in args.directions:
        figs.append(plt.figure())
        axs.append(figs[-1].add_subplot(111))
        axs[-1].set_aspect('equal', adjustable='datalim')

    return figs, axs

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

def get_coords(para):
    # Get grid spacing
    qx = para['qx']
    qy = para['qy']
    qzrange = para['qzrange']

    # Find the center index of the grid
    cx = para['nx']//2
    cy = para['ny']//2
    cz = para['zrange']//2

    # the offset of pluto from the center isn't always availible
    try:
        po = para['pluto_offset']
    except KeyError:
        print("Couldn't get pluto_offset. It has been assumed to be 0, but it probably isn't.")
        po = 0

    # Shift grid so that Pluto lies at (0,0,0)
    qx = (qx - qx[len(qx)//2 + po])
    qy = (qy - qy[len(qy)//2])
    qzrange = (qzrange - qzrange[len(qzrange)//2])

    infodict = {'px':qx,'py':qy,'pz':qzrange,'cx':cx,'cy':cy,'cz':cz, 'po':po}

    return infodict

def get_scaled_coords(para, scale):
    infodict = get_coords(para)
    infodict['px'] /= scale
    infodict['py'] /= scale
    infodict['pz'] /= scale

    return infodict

def get_pluto_coords(para):
    return get_scaled_coords(para, SCALE)

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

def data_slice(para, data, direction, coordinate=None, depth=None):
    infodict = get_pluto_coords(para)
    if direction == 'xy':
        depth = depth if depth is not None else infodict['cz']
        return data[:,:,depth]

    elif direction == 'xz':
        depth = depth if depth is not None else infodict['cy']
        return data[:,depth,:]

    elif direction == 'yz':
        depth = depth if depth is not None else infodict['cx']
        return data[depth,:,:]
    else:
        raise ValueError("direction must be one of xy, xz, or yz")

def get_next_slice(h, direction, coordinate=None, depth=None):
    data = h.get_next_timestep()[-1]
    if not h.isScalar:
        assert coordinate is not None
        data = data[:,:,:,args.variable.coordinate]

    return data_slice(h.para, data, direction, coordinate, depth)

def get_next_slice(h, direction, coordinate=None, depth=None):
    data = h.get_next_timestep()[-1]
    if not h.isScalar:
        assert coordinate is not None
        data = data[:,:,:,args.variable.coordinate]
    return data_slice(h.para, data, direction, coordinate, depth)

def plot_setup(ax, data, params, direction, depth, time_coords=False, fontsize=None, mccomas=False, titlesize=25, labelsize=20, ticklabelsize=15, skip_labeling=False):
    infodict = get_pluto_coords(params)
    if direction == 'xy':
        depth = depth if depth is not None else infodict['cz']
        dslice = data[:,:,depth]
        x,y = infodict['px'], infodict['py']
        if not skip_labeling:
            ax.set_xlabel('$X$', fontsize=labelsize)
            ax.set_ylabel('Transverse' if mccomas else 'Y', fontsize=labelsize)

    elif direction == 'xz':
        depth = depth if depth is not None else infodict['cy']
        dslice = data[:,depth,:]
        x,y = infodict['px'], infodict['pz']
        if not skip_labeling:
            ax.set_xlabel('$X$', fontsize=labelsize)
            ax.set_ylabel('$Z$', fontsize=labelsize)

    elif direction == 'yz':
        default = np.abs(infodict['px'] - (-15.0)).argmin()
        depth = depth if depth is not None else default
        print('X = {}'.format(infodict['px'][depth]))
        dslice = data[depth,:,:]
        x,y = infodict['py'], infodict['pz']
        if not skip_labeling:
            ax.set_xlabel('$Y$', fontsize=labelsize)
            ax.set_ylabel('$Z$', fontsize=labelsize)

    else:
        raise ValueError("direction must be one of 'xy', 'xz', or 'yz'")

    if time_coords:
        # Don't pass in mccomas=True since the x variable is always internal coordinates in this function
        # assuming data is coming directly from the simulation.
        x = [spice_tools.time_at_pos(xx*SCALE, mccomas=False) for xx in x]
    X,Y = np.meshgrid(x, y)

    if mccomas:
        if direction == 'xy':
            X = -X
            Y = -Y
        elif direction == 'xz':
            X = -X
        elif direction == 'yz':
            X = -X

    if not skip_labeling:
        ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)

    return X, Y, dslice

def beta_plot(fig, ax, data, params, direction, depth=None, cax=None, fontsize=None, mccomas=False, limits=None, refinement=0, titlesize=25, labelsize=20, ticklabelsize=15, skip_labeling=False, cbar_orientation='vertical'):
    X, Y, dslice = plot_setup(ax, data, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize, skip_labeling=skip_labeling)

    if limits is None:
        limits = (-1,2)

    # Setup custom colorbar
    levels = np.logspace(limits[0] - .5,limits[1] + .5, (limits[1]-limits[0]+1)*(refinement+1)+1)
    ticks = np.logspace(limits[0],limits[1],limits[1]-limits[0]+1)
    viridis = cm.get_cmap('viridis', len(levels)+2)
    cmap = ListedColormap(viridis.colors[1:-1],'beta_cmap')
    cmap.set_over(viridis.colors[-1])
    cmap.set_bad(viridis.colors[0])

    # Catch the stupid warnings I don't care about
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        mappable = ax.contourf(X.T, Y.T, dslice, levels=levels, norm=MyLogNorm(),
                                cmap=cmap, extend='both',
                                vmin=levels[0], vmax=levels[-1])

    if cax != 'None':
        cb = fig.colorbar(mappable, ax=ax, cax=cax, orientation=cbar_orientation)
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticks)

    return mappable, X, Y, dslice

def bs_hi_plot(fig, ax, n_tot, n_h, n_ch4, ux, swspeed, hdensity, params, direction, mccomas=False, depth=None, time_coords=False, fontsize=None, titlesize=25, labelsize=20, ticklabelsize=15, skip_labeling=False):
    """Plot bowshock, plutopause, and heavy ion tail defined as:
    bowshock: >20% slowing of the solar wind (defined explicitly in McComas 2016)
    plutopause: >70% exclusion of H+ (proxy for solar wind particles) (defined indirectly in McComas 2016)
    heavy ion tail: >5e12 heavy ions per cubic kilometer (McComas just talks about a heavy ion dominated tail, but that's not exactly what I wanted to show).
    """
    X, Y, n = plot_setup(ax, n_tot, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize, skip_labeling=skip_labeling)
    X, Y, h = plot_setup(ax, n_h, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize, skip_labeling=skip_labeling)
    X, Y, ch4 = plot_setup(ax, n_ch4, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize, skip_labeling=skip_labeling)
    X, Y, v = plot_setup(ax, ux, params, direction, depth, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize, skip_labeling=skip_labeling)

    bs_cont = ax.contourf(X.T, Y.T, v, levels=[-0.8*swspeed, 0], colors='b')
    pp_cont = ax.contourf(X.T, Y.T, h, levels=[0, 0.3*hdensity], colors='m')
    amount = np.ones_like(ch4)
    amount[ch4<5e12] = 0.0
    hi_cont = ax.contourf(X.T,Y.T, amount, levels=[.5,1], colors='r')

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

def scientific_format(digits=2):
    fmt_str = '{{:.{}e}}'.format(digits)
    def fmt(x, pos):
        a, b = fmt_str.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a,b)
    return fmt

def direct_plot(fig, ax, data, params, direction, depth=None, cax=None, time_coords=False, fontsize=None, mccomas=False, titlesize=25, labelsize=20, ticklabelsize=15, cbtitle='', skip_labeling=False, **kwargs):
    X, Y, dslice = plot_setup(ax, data, params, direction, depth, time_coords, fontsize=fontsize, mccomas=mccomas, titlesize=titlesize, labelsize=labelsize, ticklabelsize=ticklabelsize, skip_labeling=skip_labeling)

    mappable = ax.pcolormesh(X,Y,dslice.transpose(), **kwargs)

    fmt = plticker.FuncFormatter(scientific_format(digits=1))
    if cax != 'None':
        if cax == None:
            if 'SymLogNorm' in repr(kwargs['norm']):
                cb = fig.colorbar(mappable, ax=ax, ticks=plticker.SymmetricalLogLocator(linthresh=0.01, base=10))
            elif 'LogNorm' in repr(kwargs['norm']):
                cb = fig.colorbar(mappable, ax=ax, shrink=0.7, ticks=plticker.LogLocator())
            else:
                cb = fig.colorbar(mappable, ax=ax, shrink=0.7, format=fmt)
        else:
            cb = fig.colorbar(mappable, cax=cax, format=fmt)

        cb.ax.set_title(cbtitle, fontsize=ticklabelsize)
        cb.ax.tick_params(labelsize=16)

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
