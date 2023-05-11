#! /usr/bin/env python3
# Time-stamp: <2023-05-11 11:05:17 christophe@pallier.org>

"""French Tokenizer based on stanza.

Usage:
  frtok.py FILE1 [FILE2 ...] 
"""

import sys
import os.path as op
import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize')

if __name__ == '__main__':
    for text_file in sys.argv[1:]:
        basename, _ = op.splitext(text_file)
        output_filename = op.join(basename + '-tokenized.txt')

        with open(text_file, 'rt') as f:
            txt = f.readlines()

        with open(output_filename, 'wt') as outfile:
            for line in txt:
                doc = nlp(line)
                for sent in doc.sentences:
                    print(*[f'{token.text}' for token in sent.tokens], sep=' ', file=outfile)
