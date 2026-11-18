# DIT5411 Character Recognition

THEi DIT5411 Machine Learning assignment: recognise handwritten traditional Chinese characters with TensorFlow.

**Repository name:** `DIT5411-FongKaChung`

**Dataset:** 13,065 traditional Chinese characters, average ~50 samples each (300×300).  
Source: [Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

## Assignment requirements and how they are met

| Requirement | Implementation |
| --- | --- |
| TensorFlow, multiple layers, **13,065 output neurons** | Final dense layer is `logits_13065` in every model (`dit5411/models.py`) |
| First 40 samples of each character for training, the rest for testing | `dit5411/prepare.py` (`split_files`). Filenames are sorted numerically so `2` comes before `10`. |
| At least 200 training images per character using rotation, shearing and scaling | `dit5411/augment.py`. Each of the 40 originals is kept and four OpenCV variants are added (40 × 5 = 200). Test images are never augmented. |
| Several architectures; keep the one with the highest test recognition rate | Three models: `mlp`, `small_cnn`, `fast_resnet`. The best is selected by test accuracy. |

## Augmented image creation method

Augmentation is performed with OpenCV geometric transforms (the same tools used in the [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV) material: `getRotationMatrix2D` and `warpAffine`).

For every original training image:

1. Resize to 128×128 (twice the network input size) so the warp retains stroke detail.
2. Apply one seeded geometric transform. With 40 originals, each image receives all four variants:
   - **Rotation**: 8–16 degrees, clockwise or counter-clockwise.
   - **Shear**: 0.12–0.28 in either direction, plus at most 3 degrees of rotation.
   - **Scale**: 1.08–1.22 (or the reciprocal for zoom-out), plus at most 3 degrees of rotation.
   - **Combined**: rotation 5–12°, shear 0.08–0.18, scale 0.88–1.12.
3. Fill empty corners with the median colour of the four corners (so a white page stays white and a black page stays black).
4. Resize to 64×64 grayscale using area resampling.

The rotation angles are deliberately small. A large rotation can make one Chinese character look like another and would train the wrong label.

The 40 resized originals are kept, giving exactly **200 training images per character**. The same random seed (`2026`) produces the same augmented set every time. Test images are only resized; they receive no geometric transforms.

Example of the four transforms (drawn glyph for clarity):

![Rotation, shear, scale and combined](docs/augmentation_examples.jpg)

After the full dataset is packed, real handwriting examples are written to `docs/real_augmentation_examples.jpg`.

Packed records are stored as raw bytes (64×64 uint8 pixels + little-endian class index) in a single sequential file. This keeps the GPU busy; millions of individual PNG files would make training data-bound.

## AI models

All three models take a 64×64×1 uint8 image, rescale it to [0, 1] on the GPU, and finish with a float32 dense layer of **13,065 neurons** (one per character).

| Model | Architecture | Role |
| --- | --- | --- |
| `mlp` | Flatten → Dense(1024) → Dense(512) → Dense(13065). Dropout 0.3. | Accuracy floor. No spatial filters. |
| `small_cnn` | Six 3×3 convolutions, batch normalisation, stride-2 downsampling, global average pooling, Dense(13065). | Small convolutional baseline. |
| `fast_resnet` | Residual stages at 64, 128, 256 and 512 channels, global average pooling, Dense(13065). ~0.5 billion multiply-adds per image. | Recommended model. Highest test recognition rate on the RTX 5080 run below. |

Common training settings for all models:

- Mixed precision (`mixed_float16`) so Tensor Cores run the convolutions; logits stay float32.
- XLA (`jit_compile`) so kernels are compiled for the actual GPU (including Blackwell / compute capability 12.0).
- AdamW (weight decay 1e-4), gradient clip norm 1.0.
- Learning rate `1e-3 * sqrt(batch/256)`, capped at `4e-3`, with one epoch of warmup followed by cosine decay.
- Label smoothing 0.05 (13,065 classes and only 200 images each would overfit a hard one-hot).
- Automatic batch size measured from the GPU memory budget (16 GB), then rounded down to a multiple of 64.
- `steps_per_execution=16` to reduce Python overhead.
- Sequential reader with an 8 MB buffer, parallel parsing and prefetch. Augmentation is done once offline, not inside the training loop.

The checkpoint keeps the epoch with the highest **test** accuracy (the assignment recognition rate). There is no separate validation split.

## Recognition accuracy on the test set

Results obtained on the full 13,065-character dataset (test set = held-out original images, never augmented). Twelve epochs on an NVIDIA RTX 5080. Epoch time is the steady time of one full pass over the 2,613,000 training images, including the test pass.

| Model | Output neurons | Parameters | Test recognition rate | Top-5 | Epoch time |
| --- | --- | --- | --- | --- | --- |
| mlp | 13065 | 11,422,473 | 0.01% | 0.04% | 6.3 s |
| small_cnn | 13065 | 3,932,073 | 98.28% | 99.83% | 24.7 s |
| **fast_resnet** | **13065** | **17,823,657** | **99.10%** | **99.90%** | **128.6 s** |

**fast_resnet details** (from `runs/fast_resnet/result.json`):

- Parameters: 17,823,657
- Test recognition rate: **99.10%**
- Top-5 accuracy: **99.90%**
- Training samples: 2,613,000 (13,065 × 200)
- Test samples: 162,593
- Steady epoch time: 128.6 s

`result.json` also lists the 20 characters with the lowest per-class accuracy. These are useful for discussing remaining error cases (rare characters, visually similar glyphs, or characters with very few test samples).

The model with the highest test recognition rate is kept as the final model (`fast_resnet` in the run shown above).

## How to reproduce

### 1. Environment (Ubuntu on WSL2 recommended for RTX 50-series)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
python train.py --check-gpu
```

### 2. Prepare the dataset

```bash
# After unpacking the full 13,065-character set to ~/tchd/cleaned_data
python prepare_dataset.py --data-dir "$HOME/tchd/cleaned_data" --out-dir data/packed
```

### 3. Train

```bash
python train.py --model fast_resnet --data-dir data/packed --epochs 12
# or compare all three models
python train.py --model all --data-dir data/packed --epochs 12
```

### 4. Evaluate and predict

```bash
python evaluate.py --run runs/fast_resnet
python predict.py --run runs/fast_resnet --image path/to/character.png
```

Output files for the report:

- `runs/<model>/summary.txt` — layer list
- `runs/<model>/history.csv` — accuracy per epoch
- `runs/<model>/result.json` — test recognition rate, top-5, parameter count
- `runs/comparison.json` — results of all three models (when `--model all` is used)

- Source: https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

```bibtex
@misc{AI.FREE2020,
  author = {Po-Chuan Chen},
  title = {Traditional Chinese Handwriting Dataset},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset}}
}
```

Original collection: Department of EECS, Southern Taiwan University of Science and Technology, https://scidm.nchc.org.tw/dataset/stusteecs_chinese_mnist

## References

- AI.FREE Team, Traditional Chinese Handwriting Dataset. https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
- OpenCV geometric transforms used for the augmented images. https://github.com/darshitajain/Introduction-to-OpenCV
- Keras / TensorFlow handwritten-digit project used as a report-format reference. https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow
- TensorFlow mixed precision and XLA. https://www.tensorflow.org/guide/mixed_precision
- NVIDIA RTX 5080 specifications. https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/
