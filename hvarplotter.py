#!/usr/bin/env python
import numpy as np
from HybridReader2 import HybridReader2 as hr
from HybridReader2 import NoSuchVariable
from HybridHelper import parser, parse_cmd_line, init_figures, direct_plot, bs_hi_plot, traj_plot, get_pluto_coords, streams, build_format_coord, build_pcolormesh_format_coord
from HybridParams import HybridParams as hp
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams

def var_sanity_check(isScalar, coord):
    if isScalar and coord is not None:
        raise ValueError("Don't specify a coordinate for scalars.")
    if not isScalar and coord is None:
        raise ValueError("Must specify a coordinate for vectors.")

def plot_variable(figs, axs, args):
    # Special cases have special considerations when loading data,
    # e.g. unit conversions, or a product of two variables etc.
    if args.variable.name == 'pressure':
        try:
            hn = hr(args.prefix, 'np_tot', force_procs=args.n)
        except NoSuchVariable:
            hn = hr(args.prefix, 'np', force_procs=args.n)

        para = hn.para
        n = hn.get_timestep(args.stepnum)[-1]
        try:
            T = hr(args.prefix, 'temp_tot', force_procs=args.n).get_timestep(args.stepnum)[-1]
        except NoSuchVariable:
            T = hr(args.prefix, 'temp_p', force_procs=args.n).get_timestep(args.stepnum)[-1]

        # Convert units
        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = 1.60218e-19 * T                  # eV -> J

        data = n*T

    elif args.variable.name == 'bmag':
        hb = hr(args.prefix, 'bt', force_procs=args.n)
        para = hb.para
        B = hb.get_timestep(args.stepnum)[-1]
        B = para['ion_amu']*1.6726219e-27/1.60217662e-19 * B # ion gyrofrequency -> T
        Bmag = np.sqrt(np.sum(B**2, axis=-1))
        data = Bmag

    elif args.variable.name == 'Emag':
        hE = hr(args.prefix, 'E', force_procs=args.n)
        para = hE.para
        E = hE.get_timestep(args.stepnum)[-1]
        E = para['ion_amu']*1.6726219e-27/1.60217662e-19 * E # ion acceleration -> V/m
        Emag = np.sqrt(np.sum(E**2, axis=-1))
        data = Emag

    elif args.variable.name == 'ajmag':
        haj = hr(args.prefix, 'aj', force_procs=args.n)
        para = haj.para
        aj = haj.get_timestep(args.stepnum)[-1]
        ajmag = np.sqrt(np.sum(aj**2, axis=-1))
        data = ajmag

    elif args.variable.name == 'ajpar':
        haj = hr(args.prefix, 'aj', force_procs=args.n)
        hbt = hr(args.prefix, 'bt', force_procs=args.n)
        para = haj.para
        aj = haj.get_timestep(args.stepnum)[-1]
        bt = hbt.get_timestep(args.stepnum)[-1]
        btmag = np.sqrt(np.sum(bt**2, axis=-1))

        data = np.sum(aj*bt, axis=-1)/btmag

    elif args.variable.name == 'fmach':
        hn = hr(args.prefix, 'np', force_procs=args.n)
        para = hn.para
        n = hn.get_timestep(args.stepnum)[-1]
        T = hr(args.prefix, 'temp_tot', force_procs=args.n).get_timestep(args.stepnum)[-1]
        B = hr(args.prefix, 'bt', force_procs=args.n).get_timestep(args.stepnum)[-1]
        u = hr(args.prefix, 'up', force_procs=args.n).get_timestep(args.stepnum)[-1]
        ux = -u[:,:,:,0]

        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = 1.60218e-19 * T                  # eV -> J
        B = para['ion_amu']*1.6726219e-27/1.60217662e-19 * B # ion gyrofrequency -> T

        B2 = np.sum(B**2, axis=-1)

        c   = 3e8       # m/s
        mu0 = 1.257e-6  # H/m
        mp  = 1.602e-19 # kg
        gamma = 3

        # Upstream alfven velocity
        us_va = np.sqrt(B2[-1,0,0]/(mu0*mp*n[-1,0,0]))

        # Upstream ion acousitic speed
        us_vs = np.sqrt(gamma*T[-1,0,0]/mp)

        # Upstream fastmode velocity
        us_vf = c*np.sqrt((us_vs**2 + us_va**2)/(c**2 + us_va**2))

        data = ux/us_vf

    elif args.variable.name == 'upmag':
        h = hr(args.prefix,'up', force_procs=args.n)

        data = h.get_timestep(args.stepnum)[-1]
        data = np.linalg.norm(data, axis=-1)
        para = h.para

    else: # Generic case. Plots the variable directly
        h = hr(args.prefix,args.variable.name, force_procs=args.n)
        var_sanity_check(h.isScalar, args.variable.coordinate)

        data = h.get_timestep(args.stepnum)[-1]
        if not h.isScalar:
            data = data[:,:,:,args.variable.coordinate]
        if str(args.variable).startswith('bt'):
            data *= h.para['ion_amu']*1.6726219e-27/1.60217662e-19 # ion gyrofrequency -> T
        para = h.para

    for fig, ax, d in zip(figs, axs, args.directions):
        m, X, Y, C = direct_plot(fig, ax, data, para, d, 
                cmap=args.colormap,
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize, 
                cbtitle=args.units)
        ax.format_coord = build_pcolormesh_format_coord(m)
    
def get_1d_scalar(h):
    steps, times, data = h.get_all_timesteps()

    return steps, times, data[:,:,0,0]

def get_1d_vector(h):
    steps, times, data = h.get_all_timesteps()

    return steps, times, data[:,:,0,0,:]

def get_1d_magnetude(h):
    steps, times, data = h.get_all_timesteps()

    return steps, times, np.linalg.norm(data[:,:,0,0,:], axis=-1)

def plot_1d_variable(fig, ax, args):

    # Get upstream values
    para = hp(args.prefix, force_version=args.force_version).para
    B0 = para['b0_init']           # T  (yes, b0_init is already in Tesla. No need to convert.)
    n0 = para['nf_init']/1000**3   # m^-3

    ## Handy parameters common to most plots
    c = 3e8 # speed of light in m/s
    q = 1.602e-19 # C
    m = para['ion_amu']*1.6726e-27 # kg
    q_over_m = q/m # C/kg
    e0 = 8.854e-12 # F/m
    mu_0 = 1.257e-6

    # Some upstream parameters
    omega_pi = np.sqrt(n0*q*q_over_m/e0) # rad/s
    lambda_i = c/omega_pi # m
    omega_ci = q_over_m*B0 # rad/s

    ## Special cases first, the general case is the else block at the bottom
    if args.variable.name == 'pressure':
        steps, times, n = get_1d_scalar(hr(args.prefix, 'np_tot', force_version=args.force_version, force_procs=args.n))
        _, _, T = get_1d_scalar(hr(args.prefix, 'temp_tot', force_version=args.force_version, force_procs=args.n))

        # Convert units
        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = q * T                  # eV -> J

        x = 1000*para['qx']/lambda_i # position in units of lambda_i
        t = times*omega_ci # time in units of omega_ci^-1

        pressure = n*T

        mesh = ax.pcolormesh(t, x, pressure.T, 
                cmap=args.colormap,
                norm=args.norm,
                vmin=args.vmin,
                vmax=args.vmax)
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)


    elif args.variable.name == 'beta':
        steps, times, n = get_1d_scalar(hr(args.prefix, 'np_tot', force_version=args.force_version, force_procs=args.n))
        _, _, T = get_1d_scalar(hr(args.prefix, 'temp_tot', force_version=args.force_version, force_procs=args.n))
        _, _, B = get_1d_vector(hr(args.prefix, 'bt', force_version=args.force_version, force_procs=args.n))


        # Convert units
        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = q * T                  # eV -> J
        B = B/q_over_m # ion gyrofrequency -> T

        # Compute B \cdot B
        B2 = np.sum(B**2, axis=-1)

        # Compute plasma beta
        beta = n*T/(B2/(2*mu_0))

        x = 1000*para['qx']/lambda_i # position in units of lambda_i
        t = times*omega_ci # time in units of omega_ci^-1

        mesh = ax.pcolormesh(t, x, beta.T, 
                cmap=args.colormap,
                norm=args.norm,
                vmin=args.vmin,
                vmax=args.vmax)
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)

    elif args.variable.name == 'bmag':
        steps, times, B = get_1d_magnetude(hr(args.prefix, 'bt', force_version=args.force_version, force_procs=args.n))

        B = B/q_over_m # ion gyrofrequency -> T

        x = 1000*para['qx']/lambda_i # position in units of lambda_i
        t = times*omega_ci # time in units of omega_ci^-1

        mesh = ax.pcolormesh(t, x, B.T, 
                cmap=args.colormap,
                norm=args.norm,
                vmin=args.vmin,
                vmax=args.vmax)
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)

    elif args.variable.name == 'brat':
        steps, times, B = get_1d_magnetude(hr(args.prefix, 'bt', force_version=args.force_version, force_procs=args.n))

        B = B/q_over_m # ion gyrofrequency -> T

        B = B/para['b0_init'] # Normalize

        x = 1000*para['qx']/lambda_i # position in units of lambda_i
        t = times*omega_ci # time in units of omega_ci^-1

        mesh = ax.pcolormesh(t, x, B.T, 
                cmap=args.colormap,
                norm=args.norm,
                vmin=args.vmin,
                vmax=args.vmax)
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)

    elif args.variable.name == 'upmag':
        steps, times, up = get_1d_magnetiude(hr(args.prefix, 'up', force_version=args.force_version, force_procs=args.n))

        x = 1000*para['qx']/lambda_i # position in units of lambda_i
        t = times*omega_ci # time in units of omega_ci^-1

        mesh = ax.pcolormesh(t, x, up.T, 
                cmap=args.colormap,
                norm=args.norm,
                vmin=args.vmin,
                vmax=args.vmax)
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)

    else:
        h = hr(args.prefix,args.variable.name, force_version=args.force_version, force_procs=args.n)
        var_sanity_check(h.isScalar, args.variable.coordinate)

        if h.isScalar:
            steps, times, data = get_1d_scalar(h)
        else:
            steps, times, data = get_1d_vector(h)
            data = data[:,:,args.variable.coordinate]

        x = 1000*para['qx']/lambda_i # position in units of lambda_i
        t = times*omega_ci # time in units of omega_ci^-1

        mesh = ax.pcolormesh(t, x, data.T, 
                cmap=args.colormap,
                norm=args.norm,
                vmin=args.vmin,
                vmax=args.vmax)
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)

    ax.format_coord = build_pcolormesh_format_coord(mesh)

    return mesh

