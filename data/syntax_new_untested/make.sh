#! /usr/bin/bash
# Time-stamp: <2023-05-11 11:46:41 christophe@pallier.org>

# extract the texts from lpp stimulation files, the tokenize and parse them


python tsv_to_txt.py run*.tsv

python frtok.py run*.txt

MTGDIR="$HOME/git_stuff/mtgpy"

for f in run*-tokenized.txt
do
    OUTPUT="$(basename -s .txt $f).syntax.txt" 
    python "$MTGDIR/src/mtg.py" eval "$MTGDIR/models/french_all_lr_0.00004_B_16_H_250_diff_0.3_flaubert/" "$f" "$OUTPUT" 
done
