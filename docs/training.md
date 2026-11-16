# Training protocol

Official ISIC configuration here means released train/validation/test partitions, not an organizer-mandated optimizer or augmentation recipe.

| Dataset | Train | Validation | Test | Input |
| --- | --- | --- | --- | --- |
| ISIC2017 | 2000 images | 150 images | 600 images | 256x256 |
| ISIC2018 | 2594 images | 100 images | 1000 images | 256x256 |
| Synapse | 18 cases / 2211 listed slices | None | 12 cases | 224x224 |

Presets use 300 epochs, batch size 32, seed 42, AdamW (lr 5e-4, weight decay 1e-3), cosine scheduling to 1e-6, and encoder freezing for 10 epochs. Online augmentation uses independent horizontal/vertical flips and random multiples of 90-degree rotation. Masks use nearest-neighbor resizing. ISIC RGB uses ImageNet normalization. Synapse images use cubic resizing and intensities transformed from [0,1] to [-1,1].

Each head uses 0.4 CE + 0.6 Dice (BCE with logits for binary). Dice uses squared denominators and includes background for multiclass tasks. Total loss is `(L4 + 0.2*(L0+L1+L2+L3))/1.8`. Adjust `aux_weight` for comparisons.

## Synapse labels

Training predicts eight organs plus background.

| Target | Organ | Original source ID |
| --- | --- | --- |
| 0 | Background and excluded organs | 0,5,9,10,12,13 |
| 1 | Aorta | 8 |
| 2 | Gallbladder | 4 |
| 3 | Left kidney | 3 |
| 4 | Right kidney | 2 |
| 5 | Liver | 6 |
| 6 | Pancreas | 11 |
| 7 | Spleen | 1 |
| 8 | Stomach | 7 |

`original13` applies this mapping to both training and test data. Use `indexed9` only for this exact target ordering. A different eight-organ ordering requires explicit conversion; values 0..8 alone do not establish semantic equivalence. The supplied train list has 2211 entries and is used unchanged.

## Cache and checkpoints

Cache identity includes source metadata, preprocessing version, input size, and Synapse label format. Only deterministic decoding, resizing and mapping are cached; augmentation runs online. The default puts cached tensors on the selected GPU; CPU caching is supported.

The multi-GPU launcher snapshots source before starting jobs. Every job saves configuration, history, logs, last checkpoint and (for ISIC) validation-best checkpoint. Resume restores optimizer, scheduler and random states. Keep the same total epoch budget when resuming cosine scheduling. Data, caches, runs, weights and machine-specific paths are excluded from Git.

## Evaluation

The last checkpoint is the default test checkpoint. ISIC best.pt is selected solely by validation Dice. Binary metrics are per-image means at input resolution: Dice, IoU, accuracy, sensitivity and specificity. These are not native-resolution challenge scores or thresholded challenge Jaccard.

Synapse slice predictions are restored to the original grid and assembled into volumes. Reports include per-case/organ and mean Dice, plus HD95 in voxel units. With one empty set HD95 is undefined (null), and its undefined-pair count is reported alongside the mean over defined pairs. Physical-spacing HD95 requires explicit spacing and is not claimed here.

Run `--test-only --checkpoint PATH` to evaluate a fixed checkpoint. CUDA selective scan has no CPU-only model execution path.
