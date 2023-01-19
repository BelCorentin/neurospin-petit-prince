import grequests  # pip install grequests
from pathlib import Path
from tqdm import tqdm
import textgrid  # pip install git+https://github.com/kylerbrown/textgrid#egg=textgrid
import typing as tp
import numpy as np
import re 
from urllib.request import urlopen


def online_MAUS_aligner(
    txt_files, wav_files, language, ext="TextGrid", save_dir=None,
):
    """
    phonemes description:
    https://clarin.phonetik.uni-muenchen.de/BASWebServices
    /services/runMAUSGetInventar?LANGUAGE=nld-NL
    phoneme segmentation:
    web interface: BASWebServices/interface/clarinIFrame
    rest api: BASWebServices/services/help
    """
    assert len(txt_files) == len(wav_files)
    queue = list()
    for i, (txt_file, wav_file) in enumerate(zip(txt_files, wav_files)):
        txt_file = Path(txt_file)
        wav_file = Path(wav_file)
        assert txt_file.exists()
        assert wav_file.exists()
        assert txt_file.name[:-4] == wav_file.name[:-4]

        # FIXME save_dir_ should be compulsory
        save_dir_ = wav_file.parent if save_dir is None else save_dir
        align_file = save_dir_ / wav_file.name.replace(".wav", f".{ext}")
        if align_file.exists():
            continue

        url = "http://clarin.phonetik.uni-muenchen.de/BASWebServices"
        url += "/services/runMAUSBasic"
        r = grequests.post(
            url,
            params=dict(LANGUAGE=language, OUTFORMAT=ext),
            files=[
                ("SIGNAL", open(wav_file, "rb")),
                ("TEXT", open(txt_file, "r")),
            ],
        )
        queue.append(dict(request=r, wav_file=wav_file, align_file=align_file))

    # Parallel url requests
    if len(queue):
        print(f"Retrieving {len(queue)} segmentations via BASWebServices...")
    else:
        return []

    pbar = tqdm(total=len(queue))
    starts = list(range(0, len(queue), 100)) + [len(queue)]
    bads = list()
    for start, stop in zip(starts, starts[1:]):
        subqueue = [queue[i] for i in range(start, stop)]
        ans = grequests.map([r["request"] for r in subqueue])
        wavs = [r["wav_file"] for r in subqueue]
        aligns = [r["align_file"] for r in subqueue]
        for r, wav_file, align_file in zip(ans, wavs, aligns):

            # check status
            if "<success>" not in r.text:
                print(f"error with {wav_file.name}")
                bads.append(wav_file.name)
                continue
            success = re.search("<success>(.*)</success>", r.text).group(1)
            if success != "true":
                print(f"error with {wav_file.name}")
                bads.append(wav_file.name)
                continue
            link = re.search("<downloadLink>(.*)</downloadLink>", r.text)

            # download
            with open(str(align_file), "wb") as f:
                f.write(urlopen(link.group(1)).read())
            pbar.update(1)
    if len(bads):
        print(f"{len(bads)}/{len(queue)} files did not get aligned:", bads)
    return bads


def read_textgrid(fname):
    """Parse TextGrid Praat file and generates a dataframe containing both
    words and phonemes"""
    # FIXME: merge with textgrid functions from MOUS schoffelen dataset
    tgrid = textgrid.read_textgrid(str(fname))
    parts: tp.Dict[str, tp.Any] = {}
    for p in tgrid:
        if p.name != "" and p.name != "<p:>":  # Remove empty entries
            parts.setdefault(p.tier, []).append(p)

    # Separate orthographics, phonetics, and phonemes
    words = parts["ORT-MAU"]
    words_ph = parts["KAN-MAU"]
    phonemes = parts["MAU"]
    if len(words) != len(words_ph):
        raise RuntimeError(
            "Orthographics and phonetics have different lengths."
        )

    # Def concatenate orthographics and phonetics
    rows: tp.List[tp.Dict[str, tp.Any]] = []
    for word_id, (w_r, w_ph_r) in enumerate(zip(words, words_ph)):
        if w_r.start != w_ph_r.start or w_r.stop != w_ph_r.stop:
            raise RuntimeError(f"Mismatch: {w_r} and {w_ph_r}")
        rows.append(
            dict(
                event_type="word",
                onset=w_r.start,
                duration=w_r.stop - w_r.start,
                word_id=word_id,
                name=w_r.name,
                word_ph=w_ph_r.name,
                word=w_r.name,
            )
        )

    # Add timing of individual phonemes
    starts = np.array([i["onset"] for i in rows])
    # phonemes and starts are both ordered so this could be further optimized
    # if needs be
    for phoneme in phonemes:
        idx = np.where(phoneme.start < starts)[0]
        idx = idx[0] - 1 if idx.size else len(rows) - 1
        row = rows[idx]
        rows.append(
            dict(
                event_type="phoneme",
                onset=phoneme.start,
                duration=phoneme.stop - phoneme.start,
                word_id=row["word_id"],
                word=row["word"],
                word_ph=row["word_ph"],
                name=phoneme.name,
            )
        )
    # not sure why sorting is needed, but otherwise a sample is dropped
    rows.sort(key=lambda x: x["onset"])
    return rows

with open('./wav.txt') as f:
    list_wav = (f.read().splitlines())
    list_wav.sort()


with open('./txt.txt') as f:
    list_txt = (f.read().splitlines())
    list_txt.sort()

grids = online_MAUS_aligner(txt_files=list_txt,
                            wav_files=list_wav,
                            language='fr', save_dir=Path('.'))
