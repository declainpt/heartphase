# Heartphase: A cardiac coherence standard

[Heartphase](https://heartphase.com) takes a raw Lead I ECG (from an Apple Watch) then reconstructs a 3D phase space portrait and computes a coherence score, providing visual and numerical feedback about the degree of coherence between all heartbeats in the ECG.

#### Code

Heartphase 0.0.7 has five main files:

* `heartphase.py` — Reconstructs a 3D phase space portrait and computes a coherence score.
* `animation.py` — Animates the raw ECG and the 3D phase space portrait in synchrony.
* `fraction.py` — Isolates and animates a selected heartbeat in the 3D phase space portrait.
* `selection.py` — Simple tool for manual labelling of PQRST components in ECG.
* `variation.py` — Computes the variance of all features (waveforms, segments and intervals) of all heartbeats.

As of Heartphase 0.0.3, Heartphase (`heartphase.py`) includes Conjugation, which time-reverses each segmented heartbeat and computes the coherence score for each time-reversed pair. These time-reversed coherence scores serve to confirm the coherence scores of the original heartbeats.

As of Heartphase 0.0.6, Fraction (`fraction.py`) only loads an ECG labelled using Selection (`selection.py`) in order to highlight the P-wave, QRS-complex and T-wave of a selected heartbeat.

The code is experimental and needs several improvements, e.g. more robust heartbeat segmentation and component detection. The two ECG samples (`ecg-sample.csv` and `ecg-sample.csv`) can be used to used to demo the code. As of Heartphase 0.0.5, `ecg-sample-pqrst.csv` is included as an example of an ECG labelled using `selection.py`.

To run the code, make sure you have installed the dependencies:

```
pip install numpy pandas scipy matplotlib tqdm
```
If you choose to use your own ECG, **remove** the following from the exported CSV to leave only the amplitude values (starting on row 1):
```
Name
Date of Birth,
Recorded Date,
Classification,
Symptoms,
Software Version,
Device,
Sample Rate,


Lead,Lead I
Unit,µV
```

#### Working paper

The working paper available at [heartphase.com](https://heartphase.com/) provides greater context about Heartphase.

Supplements:

* [Supplement 1 — Animation](https://www.heartphase.com/supplements/f115817a00086655cf073a4e14428288ac736a9e1ab8b07a152b7a0c44704989.pdf): Provides context for `animation.py`
* [Supplement 2 — Conjugation](https://www.heartphase.com/supplements/9c3641b2c4fa9ffa47d6c6c081a2e61ad18db747ba1307eaf33303aba1f5d669.pdf): Provides context for Conjugation.
* [Supplement 3 — Fraction](https://www.heartphase.com/supplements/9c221fc7eb1c0da714e1e69d72e532c201c28a0eb964b955db8fcdb00438f354.pdf): Provides context for `fraction.py`.
* [Supplement 4 — Selection](https://www.heartphase.com/supplements/832990a791093f1cf22a72d3535a8df491dc9c1bb666902ed0201ad2aba2c953.pdf): Provides context for `selection.py`.
* [Supplement 5 — Fraction 2](https://www.heartphase.com/supplements/cd7ad52ce06a3b7ffcba14169cb9380330c4e22e94df4c49b9da32a2af55e84c.pdf): Provides context for an update to `fraction.py`.
* [Supplement 6 — Variation](https://www.heartphase.com/supplements/f483117d718570970e834e4e4b900dee3addfb720ebaf88f1fe45e247dc3923b.pdf): Provides context for `variation.py`.
* [Supplement 7 — Atrial Fibrillation](https://www.heartphase.com/supplements/350bf51d3c27a4460d3e58754a7b4ebcbfc29e019f4923ab85bbe97f3e01d81b.pdf): Provides context for Heartphases with Atrial Fibrillation (AFib).

#### License and contribution
Heartphase 0.0.7 is released under the terms of the MIT license. Contributors are welcomed.

#### Contact
Follow [@Heartphase](https://x.com/heartphase) (and [@declainpt](https://x.com/declainpt)) on 𝕏.
