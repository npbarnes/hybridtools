import argparse
import numpy as np
from HybridReader2 import HybridReader2 as hr

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-p','--prefix', dest='prefix', default='databig', help='Name of the datafolder')
parser.add_argument('-v','--variable', dest='variable', default='np', help='Name of the variable whose data will be read')
parser.add_argument('-s','--step', dest='stepnum', type=int, default=-1, help='The specific step number to read')

def convert_sim_coords_to_pluto_coords(hybrid_object):
    # Get grid spacing
    qx = hybrid_object.para['qx']
    qy = hybrid_object.para['qy']
    qzrange = hybrid_object.para['qzrange']

    # Find the center index of the grid
    cx = hybrid_object.para['nx']/2
    cy = hybrid_object.para['ny']/2
    cz = hybrid_object.para['zrange']/2

    # the offset of pluto from the center isn't always availible
    try:
        po = hybrid_object.para['pluto_offset']
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

def last_timestep(prefix, variable):
    # Get data
    hybrid_object = hr(prefix, variable)
    data = hybrid_object.get_last_timestep()[-1]

    # Get grid spacing
    qx = hybrid_object.para['qx']
    qy = hybrid_object.para['qy']
    qzrange = hybrid_object.para['qzrange']

    # Find the center index of the grid
    cx = hybrid_object.para['nx']/2
    cy = hybrid_object.para['ny']/2
    cz = hybrid_object.para['zrange']/2

    # the offset of pluto from the center isn't always availible
    try:
        po = hybrid_object.para['pluto_offset']
    except KeyError:
        print("Couldn't get pluto_offset. It has been assumed to be 30, but it probably isn't.")
        po = 30

    # Set constatnt for Pluto radius 
    Rp = 1187. # km

    # Shift grid so that Pluto lies at (0,0,0) and convert from km to Rp
    qx = (qx - qx[len(qx)/2 + po])/Rp
    qy = (qy - qy[len(qy)/2])/Rp
    qzrange = (qzrange - qzrange[len(qzrange)/2])/Rp

    infodict = {'qx':qx,'qy':qy,'qzrange':qzrange,'cx':cx,'cy':cy,'cz':cz, 'po':po}
    return data, infodict

