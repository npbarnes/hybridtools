#!/usr/bin/python
import numpy as np
from sys import argv
import FortranFile as ff
from HybridHelper import parser
from HybridParams import HybridParams
from os.path import join

args = parser.parse_args()
p = HybridParams(args.prefix)

mass_out = np.zeros((2*p.para['nt'], p.para['num_proc']))
part_out = np.zeros((2*p.para['nt'], p.para['num_proc']))

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
                part_out[i, n] += 1/(b*p.para['beta'])
                mass_out[i, n] += kg_per_amu/m * 1/(p.para['beta']*b)

mass_out = np.sum(mass_out, axis=1)
part_out = np.sum(part_out, axis=1)

print max_i
last_n = max_i - max_i/20
print np.sum(part_out[maxi-last_n:max_i])/(last_n*p.para['dt']/2)
