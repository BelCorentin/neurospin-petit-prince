#! /usr/bin/env python3
# Time-stamp: <2023-05-11 11:05:17 christophe@pallier.org>

""" Extract the first column of tsv files.
"""

import sys
import os.path as op
import csv

for tsvfile in sys.argv[1:]:
    with open(tsvfile, "rt") as fin:
        basename, _ = op.splitext(tsvfile)
        outfname = op.join(basename + '.txt')
        with open(outfname, "wt") as fout:
            csvreader = csv.reader(fin, delimiter="\t")
            next(csvreader)  # skip header
            for row in csvreader:
                tok = row[0]
                if tok[0] in '-"«':
                    tok = tok[1:]
                tok = tok.replace("»", "")
                tok = tok.replace("-", " ")
                tok = tok.replace('"', " ")
                if not len(tok):
                    next
                    
                if tok[len(tok)-1:] in ".?!:":
                    print(tok, end='\n', file=fout)
                else:
                    print(tok, end=' ', file=fout)

