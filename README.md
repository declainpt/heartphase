# Heartphase: A coherence standard

[Heartphase](https://heartphase.com) takes a raw Lead I ECG (from an Apple Watch) then reconstructs a 3D phase space portrait and computes a coherence score, providing visual and numerical feedback about the degree of coherence between all heartbeats in the ECG.

#### Code

Heartphase 0.0.3 has two main files:

* `heartphase.py` — Reconstructs a 3D phase space portrait and computes a coherence score.
* `animation.py` — Animates the raw ECG and the 3D phase space portrait in synchrony.

As of Heartphase 0.0.3, Heartphase includes Conjugation, which time-reverses each segmented heartbeat and computes the coherence score for each time-reversed pair. These time-reversed coherence scores serve to confirm the coherence scores of the original heartbeats.

The code is experimental and needs several improvements, e.g. more robust heartbeat segmentation and component detection. The two ECG samples (`ecg-sample.csv` and `ecg-sample.csv`) can be used to used to demo the code. 

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

#### License and contribution
Heartphase 0.0.3 is released under the terms of the MIT license. Contributors are welcomed.

#### Contact
Follow [@Heartphase](https://x.com/heartphase) (and [@declainpt](https://x.com/declainpt)) on 𝕏.