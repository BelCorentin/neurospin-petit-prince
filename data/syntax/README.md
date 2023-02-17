Generation of syntactic parses
==============================

I used [Mind the Gap](https://gitlab.com/mcoavoux/mtgpy/) and a model on French trained by Maximin Coavoux.

The `*syntax.txt` files were obtained by: 

```
for f in ../text/*.txt;  
do 
	OUTPUT="$(basename -s .txt $f)".syntax.txt 
	python src/mtg.py eval models/french_all_lr_0.00004_B_16_H_250_diff_0.3_flaubert/ "$f" "$OUTPUT" 
done
```

