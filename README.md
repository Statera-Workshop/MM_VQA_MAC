# MM_VQA_MAC

**MAC on Sort-of-CLEVR - convergence study (PAI-SG Multimodal Wave 2)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One model of the visual question answering convergence comparison in [MM_Wave2_SortOfCLEVR](https://github.com/Statera-Workshop/MM_Wave2_SortOfCLEVR).

## Method

MAC (Hudson & Manning, ICLR 2018), reduced: same stem, knowledge base 361 x 64; 4 steps of control (from the question vector - there are no words, so no word attention), read attention over the knowledge base and linear write; output MLP over [question, memory]. The two position-invariant biases before the read softmax are omitted because they cannot affect it.

**Common setup** (identical for the five Wave 2 repositories; full comparison in [MM_Wave2_SortOfCLEVR](https://github.com/Statera-Workshop/MM_Wave2_SortOfCLEVR)).
Sort-of-CLEVR generated with `code/gen_sortofclevr.py` (no download): 75 x 75 RGB images with 6 objects (one per
colour, square or circle), 10 questions per image as 11-d vectors (colour, relational or not, subtype), 10 answers;
5 non-relational questions (shape, left?, top?) and 5 relational ones (shape of the closest / furthest object, count of
the same shape). Train 10,000 images (100,000 questions), test 2,000 (20,000). All models from scratch: Adam 1e-4
constant, batch 64, budgets 20 and 60 epochs, seeds 777 / 778, fp32, deterministic cuDNN. Test accuracy is evaluated
twice per epoch below 30 epochs and every epoch otherwise, each time in plain eval mode and after re-estimating the
BatchNorm statistics on 50 training batches (the series' evaluation practice); the two agree within 0.004 here.
Reported value = last evaluation.

## Results (Sort-of-CLEVR test, mean of seeds; range in brackets)

| Model | Epochs | Seeds | Final accuracy, plain (range) | Final, BN re-estimated (range) | Relational | Non-relational | Peak (epoch) | E90 (epochs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAC | 20 | 2 | 0.694 (0.027) | 0.690 (0.036) | 0.717 | 0.662 | 0.690 (20.0) | 7.8 |
| MAC | 60 | 2 | 0.877 (0.023) | 0.880 (0.018) | 0.770 | 0.989 | 0.883 (59.5) | 29.5 |

**Finding.** Slow like the Relation Network, reaching FiLM's level at 60 epochs; still rising at the end of the budget.

## Weights

Release **v1.0** (4 files, SHA-256 in `SHA256SUMS.txt`): `sortofclevr_mac_e20_s777.pth`, `sortofclevr_mac_e20_s778.pth`, `sortofclevr_mac_e60_s777.pth`, `sortofclevr_mac_e60_s778.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_c as M
m = M.build('mac')
m.load_state_dict(torch.load('sortofclevr_mac_e60_s777.pth', weights_only=True), strict=True)
```

## Run

```bash
cd code && bash smoke_c.sh        # generates the data, gates, 1-epoch smoke
python mm_c.py --model mac --epochs 60 --seed 777
python analyze_c.py
```
