import numpy as np
from HybridReader2 import HybridReader2 as hr

def last_timestep_setup(argv):
    # Get data
    try:
        prefix = argv[1]
    except IndexError:
        prefix = 'databig'
    try: 
        variable = argv[2]
    except IndexError:
        variable = 'np'

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

