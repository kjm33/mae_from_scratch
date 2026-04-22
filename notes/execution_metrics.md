2 GPUS 32x8 - small model - batch_size=512 - the 0.7 plateau effect overcome:

Epoch 1/50 avg_loss=0.7737 time=14.0s
Epoch 2/50 avg_loss=0.6110 time=13.8s
Epoch 3/50 avg_loss=0.5557 time=14.0s
Epoch 4/50 avg_loss=0.5437 time=14.2s
Epoch 5/50 avg_loss=0.5391 time=14.3s
Epoch 6/50 avg_loss=0.5359 time=14.5s
Epoch 7/50 avg_loss=0.5341 time=14.6s
Epoch 8/50 avg_loss=0.5904 time=14.7s
Epoch 9/50 avg_loss=0.7039 time=14.7s
Epoch 10/50 avg_loss=0.7051 time=14.8s
Epoch 11/50 avg_loss=0.7034 time=14.8s
Epoch 12/50 avg_loss=0.6766 time=14.9s
Epoch 13/50 avg_loss=0.6063 time=15.0s
Epoch 14/50 avg_loss=0.5651 time=15.0s
Epoch 15/50 avg_loss=0.5469 time=15.0s
Epoch 16/50 avg_loss=0.5415 time=15.1s
Epoch 17/50 avg_loss=0.5381 time=15.1s
Epoch 18/50 avg_loss=0.5356 time=15.1s
Epoch 19/50 avg_loss=0.5342 time=15.1s
Epoch 20/50 avg_loss=0.5336 time=15.1s
Epoch 21/50 avg_loss=0.5330 time=15.1s
Epoch 22/50 avg_loss=0.5350 time=15.2s
Epoch 23/50 avg_loss=0.5315 time=15.1s
Epoch 24/50 avg_loss=0.5304 time=15.2s
Epoch 25/50 avg_loss=0.5298 time=15.2s
Epoch 26/50 avg_loss=0.5294 time=15.2s
Epoch 27/50 avg_loss=0.5289 time=15.2s
Epoch 28/50 avg_loss=0.5282 time=15.2s
Epoch 29/50 avg_loss=0.5278 time=15.2s
Epoch 30/50 avg_loss=0.5269 time=15.2s
Epoch 31/50 avg_loss=0.5271 time=15.2s
Epoch 32/50 avg_loss=0.5272 time=15.2s
Epoch 33/50 avg_loss=0.5262 time=15.3s
Epoch 34/50 avg_loss=0.5259 time=15.2s
Epoch 35/50 avg_loss=0.5254 time=15.2s
Epoch 36/50 avg_loss=0.5247 time=15.2s
Epoch 37/50 avg_loss=0.5248 time=15.3s
Epoch 38/50 avg_loss=0.5246 time=15.2s
Epoch 39/50 avg_loss=0.5241 time=15.3s
Epoch 40/50 avg_loss=0.5237 time=15.2s
Epoch 41/50 avg_loss=0.5235 time=15.3s
Epoch 42/50 avg_loss=0.5237 time=15.3s
Epoch 43/50 avg_loss=0.5224 time=15.3s
Epoch 44/50 avg_loss=0.5229 time=15.3s
Epoch 45/50 avg_loss=0.5225 time=15.3s
Epoch 46/50 avg_loss=0.5229 time=15.3s
Epoch 47/50 avg_loss=0.5226 time=15.2s
Epoch 48/50 avg_loss=0.5226 time=15.3s
Epoch 49/50 avg_loss=0.5226 time=15.3s
Epoch 50/50 avg_loss=0.5221 time=15.3s

----

with CUDA graps but without tokens interaction (lower memory usage)
DALI pipeline: 238378 images from ./data/yiddish_lines.npy  (memmap, no per-epoch decode)
Epoch 1/6 avg_loss=0.8010 time=24.8s
Epoch 2/6 avg_loss=0.7065 time=6.8s
Epoch 3/6 avg_loss=0.7064 time=6.8s
Epoch 4/6 avg_loss=0.7046 time=6.8s
Epoch 5/6 avg_loss=0.7044 time=6.9s
Epoch 6/6 avg_loss=0.7043 time=6.9s
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38/38 loss 0.7043 VRAM 20.4 GB 0:00:04
        Training Summary — None
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.02 h  (59 s)       │
│ Total steps   │ 228                  │
│ Avg steps/sec │ 3.86                 │
│ Peak VRAM     │ 20.41 GB  (20899 MB) │
│ Avg loss      │ 0.7212               │
└───────────────┴──────────────────────┘

WITHOUT CUDA graphs
(venv) ➜  mae_from_scratch git:(main) ✗ ./profile_ncu.sh 15_large_batch_size
==PROF== Connected to process 387949 (/usr/bin/python3.12)
DALI pipeline: 238378 images from ./data/yiddish_lines.npy  (memmap, no per-epoch decode)
Epoch 1/6 avg_loss=0.7908 time=12.6s
Epoch 2/6 avg_loss=0.7099 time=11.5s
Epoch 3/6 avg_loss=0.7052 time=11.6s
Epoch 4/6 avg_loss=0.7026 time=11.6s
Epoch 5/6 avg_loss=0.7023 time=11.7s
Epoch 6/6 avg_loss=0.7023 time=11.7s
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46/46 loss 0.7023 VRAM 16.0 GB 0:00:11
        Training Summary — None
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.02 h  (71 s)       │
│ Total steps   │ 276                  │
│ Avg steps/sec │ 3.91                 │
│ Peak VRAM     │ 16.04 GB  (16429 MB) │
│ Avg loss      │ 0.7188               │
└───────────────┴──────────────────────┘

WITH CUDA graphs
DALI pipeline: 238378 images from ./data/yiddish_lines.npy  (memmap, no per-epoch decode)
Epoch 1/6 avg_loss=0.7759 time=19.5s
Epoch 2/6 avg_loss=0.7064 time=8.8s
Epoch 3/6 avg_loss=0.7021 time=8.8s
Epoch 4/6 avg_loss=0.7020 time=8.9s
Epoch 5/6 avg_loss=0.7019 time=8.9s
Epoch 6/6 avg_loss=0.7015 time=8.9s
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46/46 loss 0.7015 VRAM 22.2 GB 0:00:05
        Training Summary — None
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.02 h  (64 s)       │
│ Total steps   │ 276                  │
│ Avg steps/sec │ 4.32                 │
│ Peak VRAM     │ 22.22 GB  (22757 MB) │
│ Avg loss      │ 0.7150               │
└───────────────┴──────────────────────┘

          Training Summary —
         13_token_interaction
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value              ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.04 h  (139 s)    │
│ Total steps   │ 6                  │
│ Avg steps/sec │ 0.04               │
│ Peak VRAM     │ 7.73 GB  (7911 MB) │
│ Avg loss      │ 1.1897             │
└───────────────┴────────────────────┘

Epoch 1/6 avg_loss=1.5408 time=5.1s
Epoch 2/6 avg_loss=1.3443 time=0.3s
Epoch 3/6 avg_loss=1.2397 time=0.2s
Epoch 4/6 avg_loss=1.1734 time=0.2s
Epoch 5/6 avg_loss=1.1207 time=0.2s
Epoch 6/6 avg_loss=1.0866 time=0.2s
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 loss 1.0866 VRAM 7.7 GB 0:00:00
       Training Summary — None
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value              ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.00 h  (6 s)      │
│ Total steps   │ 6                  │
│ Avg steps/sec │ 0.99               │
│ Peak VRAM     │ 7.73 GB  (7913 MB) │
│ Avg loss      │ 1.2509             │
└───────────────┴────────────────────┘

DALI pipeline: 7279 images from ./data/yiddish_lines.npy  (memmap, no per-epoch decode)
Epoch 1/6 avg_loss=0.9564 time=12.8s
Epoch 2/6 avg_loss=0.7597 time=0.3s
Epoch 3/6 avg_loss=0.7441 time=0.3s
Epoch 4/6 avg_loss=0.7409 time=0.3s
Epoch 5/6 avg_loss=0.7355 time=0.3s
Epoch 6/6 avg_loss=0.7299 time=0.3s
Profiling single step...
Trace saved to ./runs/9_ultra_light_model_2026_04_14__19_09_42.pt.trace.json
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28/28 loss 0.7299 VRAM 0.5 GB 0:00:00
         Training Summary —
         9_ultra_light_model
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.00 h  (14 s)    │
│ Total steps   │ 168               │
│ Avg steps/sec │ 11.73             │
│ Peak VRAM     │ 0.52 GB  (533 MB) │
│ Avg loss      │ 0.7777            │
└───────────────┴───────────────────┘

Epoch 1/6 avg_loss=0.8703 time=230.1s
Epoch 2/6 avg_loss=0.7405 time=3.4s
Epoch 3/6 avg_loss=0.7384 time=3.4s
Epoch 4/6 avg_loss=0.7376 time=3.4s
Epoch 5/6 avg_loss=0.7370 time=3.4s
Epoch 6/6 avg_loss=0.7365 time=3.4s
Profiling single step...
Trace saved to ./runs/8_changing_path_to_32x8_2026_04_14__18_40_08.pt.trace.json
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28/28 loss 0.7365 VRAM 5.4 GB 0:00:02
          Training Summary —
       8_changing_path_to_32x8
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value              ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.07 h  (248 s)    │
│ Total steps   │ 168                │
│ Avg steps/sec │ 0.68               │
│ Peak VRAM     │ 5.38 GB  (5507 MB) │
│ Avg loss      │ 0.7600             │
└───────────────┴────────────────────┘

Epoch 1/6 avg_loss=0.9005 time=25.9s
Epoch 2/6 avg_loss=0.6295 time=11.7s
Epoch 3/6 avg_loss=0.6189 time=11.8s
Epoch 4/6 avg_loss=0.6064 time=11.9s
Epoch 5/6 avg_loss=0.5991 time=11.9s
Epoch 6/6 avg_loss=0.5943 time=12.0s
Profiling single step...
Trace saved to ./runs/7_dali_loader/
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28/28 loss 0.5943 VRAM 16.6 GB 0:00:10
    Training Summary — 7_dali_loader
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.02 h  (86 s)       │
│ Total steps   │ 168                  │
│ Avg steps/sec │ 1.96                 │
│ Peak VRAM     │ 16.56 GB  (16963 MB) │
│ Avg loss      │ 0.6581               │
└───────────────┴──────────────────────┘

Dataset ready. Loaded: 7279 images.
Epoch 1/6 avg_loss=0.9056 time=18.5s
Epoch 2/6 avg_loss=0.6270 time=12.6s
Epoch 3/6 avg_loss=0.6189 time=12.4s
Epoch 4/6 avg_loss=0.6069 time=12.6s
Epoch 5/6 avg_loss=0.5986 time=12.7s
Epoch 6/6 avg_loss=0.5943 time=12.7s
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 29/29 loss 0.5932 VRAM 16.0 GB 0:00:12
  Training Summary — 3_8_bit_optimizer
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.02 h  (81 s)       │
│ Total steps   │ 174                  │
│ Avg steps/sec │ 2.14                 │
│ Peak VRAM     │ 15.98 GB  (16360 MB) │
│ Avg loss      │ 0.6586               │
└───────────────┴──────────────────────┘

2_accelerator_disabled
Dataset ready. Loaded: 7279 images.
Epoch 1/6 avg_loss=0.8779 time=79.7s
Epoch 2/6 avg_loss=0.6272 time=12.4s
Epoch 3/6 avg_loss=0.6187 time=12.3s
Epoch 4/6 avg_loss=0.6056 time=12.4s
Epoch 5/6 avg_loss=0.5978 time=12.5s
Epoch 6/6 avg_loss=0.5891 time=12.5s
  Epoch 6/6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 29/29 loss 0.5747 VRAM 16.6 GB 0:00:12
           Training Summary —
         2_accelerator_disabled
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.04 h  (142 s)      │
│ Total steps   │ 174                  │
│ Avg steps/sec │ 1.23                 │
│ Peak VRAM     │ 16.59 GB  (16987 MB) │
│ Avg loss      │ 0.6527               │
└───────────────┴──────────────────────┘

1_inital_reference_version
Epoch 1/10 avg_loss=0.8832 time=17.2s
Epoch 2/10 avg_loss=0.6285 time=12.5s
Epoch 3/10 avg_loss=0.6212 time=12.3s
Epoch 4/10 avg_loss=0.6102 time=12.3s
Epoch 5/10 avg_loss=0.5990 time=12.4s
Epoch 6/10 avg_loss=0.5903 time=12.5s
Epoch 7/10 avg_loss=0.5780 time=12.6s
Epoch 8/10 avg_loss=0.5757 time=12.6s
Epoch 9/10 avg_loss=0.5754 time=12.6s
Epoch 10/10 avg_loss=0.5743 time=12.6s
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value                ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total time    │ 0.04 h  (130 s)      │
│ Total steps   │ 290                  │
│ Avg steps/sec │ 2.24                 │
│ Peak VRAM     │ 16.59 GB  (16991 MB) │
│ Avg loss      │ 0.6236               │
└───────────────┴──────────────────────┘
