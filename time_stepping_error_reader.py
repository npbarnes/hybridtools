#!/usr/bin/python
import argparse
import HybridHelper as hh
from HybridReader2 import HybridReader2 as hr
import FortranFile as ff
import matplotlib.pyplot as plt
import numpy

hh.parser.add_argument('filenum', type=int, help='Which time_stepping_error file to open')
args = hh.parser.parse_args()

params = hr(args.prefix, args.variable).para

# open desired FortranFile
f = ff.FortranFile('./'+args.prefix+'/c.time_stepping_error_'+str(args.filenum)+'.dat')

def read_entry(f):
    entry = {}
    entry['step'] = f.readInts(); assert len(entry['step']) == 1; entry['step'] = entry['step'][0]
    entry['rank'] = f.readInts(); assert len(entry['rank']) == 1; entry['rank'] = int(entry['rank'][0])
    entry['indicies'] = f.readInts() - 1 # subtract 1 to convert from 1-based indexing to 0-based indexing
    entry['coords'] = f.readReals()
    entry['np'] = f.readReals().reshape( (params['nx'],params['ny'],params['nz'])  , order='F')
    entry['b0'] = f.readReals().reshape( (params['nx'],params['ny'],params['nz'],3), order='F')
    entry['b1'] = f.readReals().reshape( (params['nx'],params['ny'],params['nz'],3), order='F')
    entry['bt'] = f.readReals().reshape( (params['nx'],params['ny'],params['nz'],3), order='F') 
    return entry

def read_all_entries(f):
    ret = []
    while(True):
        try:
            ret.append(read_entry(f))
        except ff.NoMoreRecords:
            break
        except ff.IntegrityError:
            print "Integrity Error"
            break
    return ret

entries = read_all_entries(f)

ks = []
for entry in entries:
    ks.append(entry['indicies'][2])

plt.plot(ks, marker='o', linestyle='None')
plt.show()

