#!/usr/bin/env python
import numpy as np
from HybridReader2 import HybridReader2 as hr
from HybridReader2 import NoSuchVariable
from HybridHelper import parser, parse_cmd_line, init_figures, direct_plot, beta_plot2, bs_hi_plot, traj_plot, get_pluto_coords, streams, build_format_coord
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams

def var_sanity_check(isScalar, coord):
    if isScalar and coord is not None:
        raise ValueError("Don't specify a coordinate for scalars.")
    if not isScalar and coord is None:
        raise ValueError("Must specify a coordinate for vectors.")

def plot_variable(fig1,fig2, ax1,ax2, args):
    ## Special cases first, the general case is direct_plot() at the bottom
    if args.variable.name == 'bs':
        hup = hr(args.prefix,'up')
        hn_tot = hr(args.prefix,'np')
        hn_h = hr(args.prefix,'np_He')
        hn_ch4 = hr(args.prefix,'np_CH4')

        ux = hup.get_timestep(args.stepnum)[-1]
        n_tot = hn_tot.get_timestep(args.stepnum)[-1]
        n_h = hn_h.get_timestep(args.stepnum)[-1]
        n_ch4 = hn_ch4.get_timestep(args.stepnum)[-1]

        ux = ux[:,:,:,0]
        para = hup.para

        bs_hi_plot(fig1, ax1, n_tot, n_h, n_ch4,ux, 401, 2.7e12, para, 'xy', 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)
        bs_hi_plot(fig2, ax2, n_tot, n_h, n_ch4,ux, 401, 2.7e12, para, 'xz', 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)

    elif args.variable.name == 'pressure':
        try:
            hn = hr(args.prefix, 'np_tot')
        except NoSuchVariable:
            hn = hr(args.prefix, 'np')

        para = hn.para
        n = hn.get_timestep(args.stepnum)[-1]
        try:
            T = hr(args.prefix, 'temp_tot').get_timestep(args.stepnum)[-1]
        except NoSuchVariable:
            T = hr(args.prefix, 'temp_p').get_timestep(args.stepnum)[-1]

        # Convert units
        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = 1.60218e-19 * T                  # eV -> J

        data = n*T

        m1, X1, Y1, C1 = direct_plot(fig1, ax1, data, para, 'xy', 
                cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize, 
                cbtitle=args.units)
        m2, X2, Y2, C2 = direct_plot(fig2, ax2, data, para, 'xz', 
                cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize, 
                cbtitle=args.units)

    elif args.variable.name == 'beta':
        hn = hr(args.prefix, 'np')
        para = hn.para
        n = hn.get_timestep(args.stepnum)[-1]
        try:
            T = hr(args.prefix, 'temp_tot').get_timestep(args.stepnum)[-1]
        except NoSuchVariable:
            T = hr(args.prefix, 'temp_p').get_timestep(args.stepnum)[-1]

        B = hr(args.prefix, 'bt').get_timestep(args.stepnum)[-1]

        # Convert units
        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = 1.60218e-19 * T                  # eV -> J
        B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T

        # Compute B \cdot B
        B2 = np.sum(B**2, axis=-1)

        # Compute plasma beta
        data = n*T/(B2/(2*1.257e-6))

        m1, X1, Y1, C1 = beta_plot2(ax1, data, para, 'xy', mccomas=args.mccomas)
        m2, X2, Y2, C2 = beta_plot2(ax2, data, para, 'xz', mccomas=args.mccomas)

        args.variable = 'Plasma Beta'

    elif args.variable.name == 'bmag':
        hb = hr(args.prefix, 'bt')
        para = hb.para
        B = hb.get_timestep(args.stepnum)[-1]
        B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T
        Bmag = np.sqrt(np.sum(B**2, axis=-1))
        data = Bmag

        m1, X1, Y1, C1 = direct_plot(fig1, ax1, data, para, 'xy', cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)
        m2, X2, Y2, C2 = direct_plot(fig2, ax2, data, para, 'xz', 
                cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)

    elif args.variable.name == 'fmach':
        hn = hr(args.prefix, 'np')
        para = hn.para
        n = hn.get_timestep(args.stepnum)[-1]
        T = hr(args.prefix, 'temp_tot').get_timestep(args.stepnum)[-1]
        B = hr(args.prefix, 'bt').get_timestep(args.stepnum)[-1]
        u = hr(args.prefix, 'up').get_timestep(args.stepnum)[-1]
        ux = -u[:,:,:,0]

        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = 1.60218e-19 * T                  # eV -> J
        B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T

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

        m1, X1, Y1, C1 = direct_plot(fig1, ax1, data, para, 'xy', 
                cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)
        m2, X2, Y2, C2 = direct_plot(fig2, ax2, data, para, 'xz', 
                cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)

    elif args.variable.name == 'ratio':
        h = hr(args.prefix, 'np')
        ch4 = hr(args.prefix, 'np_CH4')

        h_data = h.get_timestep(args.stepnum)[-1]
        ch4_data = ch4.get_timestep(args.stepnum)[-1]

        para = h.para

        ratio_plot(fig1, ax1, h_data, ch4_data, para, 'xy', 
                norm=args.norm, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)
        ratio_plot(fig2, ax2, h_data, ch4_data, para, 'xz', 
                norm=args.norm, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize)

    else:
        h = hr(args.prefix,args.variable.name)
        var_sanity_check(h.isScalar, args.variable.coordinate)

        data = h.get_timestep(args.stepnum)[-1]
        if not h.isScalar:
            data = data[:,:,:,args.variable.coordinate]
        if str(args.variable).startswith('bt'):
            data = 1e9 * 1.6726219e-27/1.60217662e-19 * data # proton gyrofrequency -> nT
        para = h.para

        m1, X1, Y1, C1 = direct_plot(fig1, ax1, data, para, 'xy', 
                cmap=args.colormap,
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize, 
                cbtitle=args.units)
        m2, X2, Y2, C2 = direct_plot(fig2, ax2, data, para, 'xz', 
                cmap=args.colormap, 
                norm=args.norm, 
                vmin=args.vmin, 
                vmax=args.vmax, 
                mccomas=args.mccomas, 
                titlesize=args.titlesize, 
                labelsize=args.labelsize, 
                ticklabelsize=args.ticklabelsize, 
                cbtitle=args.units)


    try:
        # Custom format_coord shows the value under the mouse in the status line
        # in addition to the coordinates of that point
        ax1.format_coord = build_format_coord(X1, Y1, C1)
        ax2.format_coord = build_format_coord(X2, Y2, C2)
    except NameError:
        pass
