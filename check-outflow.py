#!/usr/bin/env python
import numpy as np
from sys import argv
import FortranFile as ff
#from HybridHelper import parser
from HybridParams import HybridParams
from os.path import join
import matplotlib.pyplot as plt
import argparse
from progress import printProgressBar 

parser = argparse.ArgumentParser()
parser.add_argument('prefix')

args = parser.parse_args()
p = HybridParams(args.prefix)

mass_out = np.zeros(2*p.para['nt'])
part_out = np.zeros(2*p.para['nt'])

kg_per_amu = 1.6605e-27

# for each processor
for n in range(p.para['num_proc']):
    f = ff.FortranFile(join(p.particle,"c.outflowing_"+str(n+1)+".dat"))

    # for each half timestep
    for i in range(2*p.para['nt']):
        try:
            mrat = f.readReals()
        except ff.NoMoreRecords:
            break
        max_i = i
        beta_p = f.readReals()
        tags = f.readReals()

        # Each of the arrays must have the same length
        assert len(mrat) == len(beta_p) and len(mrat) == len(tags)
        # If that length is zero than there was no outflow
        if len(mrat) == 0:
            continue

        # for each macro particle
        for m,b,t in zip(mrat, beta_p, tags):
            if t != 0:
                part_out[i] += 1/(b*p.para['beta'])
                mass_out[i] += kg_per_amu/m * 1/(p.para['beta']*b)
    printProgressBar(n+1, p.para['num_proc'])

print(p.para['dt'])

plt.plot(part_out)

plt.show()
