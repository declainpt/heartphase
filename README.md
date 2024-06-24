# Heartphase: A coherence standard

[Heartphase](https://heartphase.com) takes a raw Lead I ECG (from an Apple Watch) then reconstructs a 3D phase space portrait and computes a coherence score, providing visual and numerical feedback about the degree of coherence between all heartbeats in the ECG.

#### Code
The code is experimental and needs several improvements, e.g. more robust heartbeat segmentation and component detection. The `ecg-sample.csv` can be used to used to demo the code. 

To run the code, make sure you have installed the dependencies:

```
pip install numpy pandas scipy matplotlib
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

#### License and contribution
Heartphase 0.0.1 is released under the terms of the MIT license. Contributors are welcomed.

#### Contact
Follow [@Heartphase](https://x.com/heartphase) (and [@declainpt](https://x.com/declainpt)) on 𝕏.

#### Gift

You can support Heartphase at this Bitcoin address: `bc1q2zqnmswxj85rw4w4g6d9jexkzth53c5usx9677`