# Running SGAR-MOT

SGAR-MOT is based on ByteTrack with one extra stage, the Object Recovery Protocol (ORP). This repository holds only the code we contribute; ByteTrack and SAM 2 are installed from their own repositories. It will be necessary to add our modifications to run our tracker.

All the code lives under `src/` folder; the repository root contains the main README, documentation, figures and the `Dockerfile` with the definition of a Docker image to run the code. We present the folder structure:

```
src/
  sgar_mot/                                 the contribution, as an installable package
    mir/                                    folder with the Mask Instance Resolver (the transformer)
    orp/                                    folder with the Object Recovery Protocol (SAM 2 + MIR)
    data/, utils/                           folders containing MIR dataset and helpers
    tools/train_mir.py                      MIR training script
    tools/build_mir_dataset.py              builds the MIR training samples with SAM 2
  configs/mir_vitbase.yml                   MIR architecture and weights
  bytetrack/                                folder containing files to ADD to a ByteTrack checkout
    yolox/tracker/sgar_tracker.py           ByteTrack + ORP
    yolox/evaluators/sgar_evaluator.py      evaluation loop
    tools/track_sgar.py                     entry point for the tracker
    tools/mine_unmatched_detections.py      mines the MIR training situations (used to create the MIR training dataset)
    exps/example/mot/*.py                   YOLOX experiment files per dataset
  sam2_sgar_mot/                            folder containing files to REPLACE in a SAM 2 checkout
    sam2/sam2_video_predictor.py            normal SAM 2 video predictor + limit_frames, + add_frames_to_state
    sam2/sam2_image_predictor.py            normal SAM 2 image predictor + get_image_mask_embeddings function
    sam2/utils/misc.py                      normal SAM 2 misc.py file + windowed frame loading, more efficient for our case
docker/Dockerfile                           fille with the necessary instructions to build a docker image 
docs/TUTORIAL.md                            this file
```

Nothing in `src/bytetrack/` overwrites a ByteTrack file, so the baseline tracker stays runnable in the same checkout and both can be compared side by side. The three files in `src/sam2_sgar_mot/` do replace their SAM 2 counterparts: they are the SAM 2 sources with our modifications, each marked in place with `[SGAR-MOT]`. ORP needs them — it consumes a sequence frame by frame, and decoding the whole
video up-front is not affordable inside a tracker.

## 1. Installation

The released code was verified against the original implementation using Python 3.10.12, Pytorch 2.4.1+cu121, NumPy 1.23.5 and CUDA 12.1. We have used, as commented in the manuscript, a single NVIDIA A100 GPU.

We recommend to use a docker installation, more direct. The Docker image can be built using `docker/Dockerfile`, from the root of this repository:

```bash
docker build -f docker/Dockerfile -t sgar-mot:v1 .

docker run --gpus all --ipc host -it \
    -v /path/to/datasets:/workspace/ByteTrack/datasets \
    -v /path/to/pretrained:/workspace/ByteTrack/pretrained \
    sgar-mot:v1
```

The working directory inside the container is `/workspace/ByteTrack`, with the tracker files and `configs/` already in place. Mount with your datasets and checkpoints.

If you want to do a manual installation, you can follow the commands below:

```bash
# 1. ByteTrack (brings YOLOX, motmetrics, lap, cython_bbox, pycocotools)
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python3 setup.py develop
pip install cython_bbox pycocotools
cd ..

# 2. SAM 2. Pin the 2.0 release (used in our adaptation)
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 7e1596c0b6462eb1d1ba7e1492430fed95023598   # SAM 2.0, the revision SGAR-MOT was developed against
pip install -e .
cd ..

# 3. SGAR-MOT
git clone https://github.com/ManuelBGomez/SGAR-MOT.git
cd SGAR-MOT
pip install -e .
pip install -r requirements.txt

# 4. Drop the tracker files into the ByteTrack checkout (nothing is overwritten)
cp -r src/bytetrack/* ../ByteTrack/

# 5. Replace the three modified SAM 2 sources in the SAM 2 checkout, and reinstall
#    so that the CUDA extension is rebuilt
cp -r src/sam2_sgar_mot/sam2/* ../sam2/sam2/
cd ../sam2 && SAM2_BUILD_ALLOW_ERRORS=0 pip install -e . && cd -
python3 -c "from sam2 import _C"   
```

Everything below is run **from the ByteTrack root**.

### Checkpoints

We recommend to save all checkpoints necessary to run the tracker inside a `pretrained/` folder:

| Checkpoint of | File name | Where from |
|---|---|---|
| YOLOX, MOT20 | `pretrained/bytetrack_x_mot20.pth.tar` | [ByteTrack model zoo](https://github.com/ifzhang/ByteTrack#model-zoo) |
| YOLOX, SportsMOT | `pretrained/bytetrack_sportsmot.pth.tar` | [MixSort model zoo](https://github.com/MCG-NJU/MixSort) |
| YOLOX, VisDrone | — | not available, see below |
| SAM 2 (large) | `pretrained/sam2_hiera_large.pt` | [Public file from Meta](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) |
| MIR | `pretrained/mir_sgar_mot.pth` | [Google Drive](https://drive.google.com/file/d/1sOU-mJH23MoUqAp7jLW9lrFcFfYhYRw5/view?usp=sharing) (361 MB) |

We provide the MIR weights that we obtained from the training process described in the manuscript. After downloading it, you can check that you got the file intact by running:

```bash
md5sum pretrained/mir_sgar_mot.pth
# 5785b737845dfc0424f26c86cde6a01d
```

**VisDrone had no released weights.** Its results were obtained with a YOLOX detector trained from scratch on VisDrone, run separately from the tracker, and then loaded into the tracker script. In order to obtain them:

1. Train a detector on VisDrone with [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) — its [training instructions](https://github.com/Megvii-BaseDetection/YOLOX#train-custom-data) cover custom datasets; VisDrone has to be converted to COCO format first, and `src/bytetrack/exps/example/mot/yolox_x_visdrone_test.py` is the experiment file the tracking side expects (1 class, test size 800x1440).
2. Run it over the split and write **one `<sequence>.json` per video** into a folder.
3. Pass that folder with `--det-folder` (see § 2) to our [tracker](./tools/track_sgar.py).

Each JSON is a flat list of detections with the fields the loader reads:

```json
[{"image_id": i, "bbox": [x1, y1, x2, y2], "score": s, "category_id": c}, ...]
```

Relevant format conventions:
* `bbox` is consumed as **corner coordinates in image space** (`x1, y1, x2, y2`).
* The loader sorts the unique `image_id` values and takes the *n*-th for frame *n*, so the ids only need to sort into frame order.
* The pipeline still builds the detector even when it does not use it, so `-c` must point at *some* loadable YOLOX checkpoint.

### Data layout

ORP reads the frames of the sequence from disk, so `--data-path` must be the directory that contains `<sequence>/img1/*.jpg`, and it must resolve to the same images the dataloader reads (`<datasets dir>/<dataset>/<split>/`).

## 2. Run Tracker experiments

```bash
# MOT20 test -- submission only, the split has no public ground truth
python3 tools/track_sgar.py \
    -f exps/example/mot/yolox_x_mot20_test.py \
    -c pretrained/bytetrack_x_mot20.pth.tar \
    --data-path datasets/MOT20/test/ \
    --sam-checkpoint pretrained/sam2_hiera_large.pt \
    --mir-config configs/mir_vitbase.yml \
    -expn mot20_test -b 1 -d 1 --fp16 --fuse --offload-cpu --mot20 --test

# SportsMOT test -- submission only, the split has no public ground truth
python3 tools/track_sgar.py \
    -f exps/example/mot/yolox_x_sportsmot_test.py \
    -c pretrained/bytetrack_sportsmot.pth.tar \
    --data-path datasets/SportsMOT/test/ \
    --sam-checkpoint pretrained/sam2_hiera_large.pt \
    --mir-config configs/mir_vitbase.yml \
    -expn sportsmot_test -b 1 -d 1 --fp16 --fuse --offload-cpu --test

# VisDrone test -- tracked from precomputed detections (-c is unused here, but a
# loadable checkpoint is still required)
python3 tools/track_sgar.py \
    -f exps/example/mot/yolox_x_visdrone_test.py \
    -c <any YOLOX checkpoint> \
    --data-path datasets/VisDrone/test/ \
    --det-folder datasets/VisDrone/dets/test \ # Or the folder where they are saved.
    --sam-checkpoint pretrained/sam2_hiera_large.pt \
    --mir-config configs/mir_vitbase.yml \
    -expn visdrone_test -b 1 -d 1 --fp16 --fuse --offload-cpu --test

# the same pipeline without ORP, to reproduce the baseline
python3 tools/track_sgar.py ... --no-orp -expn mot20_test_baseline
```

Results are written to `YOLOX_outputs/<expn>/track_results/<sequence>.txt` in MOTChallenge format, one file per sequence, flushed when the sequence ends. A sequence whose `.txt` already exists is skipped, so an interrupted run can be resumed by launching the same command again.

### Additional options

The table contains some additional options that can be used to try other experiments.

| Flag | Default | Meaning |
|---|---|---|
| `--gamma` | 0.5 | probability above which MIR recovers a track (γ) |
| `--min-track-frames` | 10 | frames a track must have survived to be eligible (F) |
| `--no-orp` | off | run the baseline tracker |
| `--offload-cpu` | off | keep SAM 2 video frames in CPU memory (recommended to try if there are out-of-memory errors) |
| `--det-folder` | none | track from `<sequence>.json` detections instead of running YOLOX |
| `--track_thresh`, `--track_buffer`, `--match_thresh`, `--mot20` | ByteTrack defaults | untouched from the baseline |

### Metrics

The numbers in the paper when there were public annotations available were computed with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (MOTA, HOTA, IDF1, FP, FN, IDSW) over the per-sequence `.txt` files.

The rest of results that involved non-public annotations were obtained by uploading the results to the corresponding evaluation servers (in the case of MOT20 and SportsMOT).

## 3. Train MIR

MIR is trained on samples mined from MOTSynth: frames where the initial association of the tracker failed, each stored with the three mask embeddings (detection, previous and current frame), their boxes and a label saying whether the track should have been recovered. 

For this, it is necessary to have available MOTSynth dataset. It is a heavy dataset compared to the others, so enough disk space should be available. In any case, we provide the code we used for both extracting the detections and train the module.

### 3.1 Mine the frames where the detector fails

The following script runs YOLOX over every MOTSynth video and matches its detections against the ground truth with the same two-round association the tracker uses, recording per object the frames where it was missed while still visible:

```bash
python3 tools/mine_unmatched_detections.py \
    --exp-file exps/example/mot/yolox_x_ablation_motsynth.py \
    -c pretrained/bytetrack_ablation.pth.tar \
    --data_path datasets/MOTSynth/
```

This writes one `<video>.json` per sequence under `datasets/MOTSynth/annotations/unmatched_detections_per_id/`.

### 3.2 Build the samples

For each of those situations, SAM 2 produces the three time-aligned masks, which are embedded and stored. Every situation yields three samples: the positive one, a negative built by swapping in a neighbouring identity (an ID switch), and a negative built by displacing the box — the `type` field is 0, 1 and 2 respectively. Set the paths and the video range in the constants at the top of `src/sgar_mot/tools/build_mir_dataset.py`, then:

```bash
python3 -m sgar_mot.tools.build_mir_dataset
```

One pickle per video is written to `SAVE_PATH`; this is what the dataset class reads.

### 3.3 Splitting and training of MIR

Put the pickle files of the validation videos in a separate directory — the paper holds out 11 sequences (617, 619, 631, 642, 653, 657, 662, 672, 680, 682 and 759), the ones with the poorest baseline tracking metrics — and point `--train-set` and `--val-set` at the two directories.

```bash
python3 -m sgar_mot.tools.train_mir \
    --train-set data/mir/train --val-set data/mir/val \
    --config configs/mir_vitbase.yml \
    --out-dir outputs --exp-name mir_vitbase
```

600 epochs, batch size 1,024, binary cross-entropy, SGD with lr 0.1 — the configuration in the paper, all of it in `configs/mir_vitbase.yml`. Checkpoints and TensorBoard logs are written to `outputs/<exp-name>/`. `SUMMARY.txt` names the epoch with the lowest validation loss. `--n-splits` controls how many groups the training sequences are split into, so that only a fraction of the samples is held in memory at a time.

## 4. Troubleshooting

* **`cv2.imread` returns `None`, or recovered boxes look blank.** `--data-path` must end in `/` (the script adds it) and point at the split directory that directly contains `<sequence>/img1/`. It has to be the same images the YOLOX exp file reads.
* **`FileNotFoundError` on the MIR weights.** The path in `configs/mir_vitbase.yml` is relative to the working directory, which is the ByteTrack root.
* **`_C.so: failed to map segment from shared object`.** SAM 2's compiled extension cannot be loaded from a filesystem mounted `noexec` (many NFS shares are). SAM 2 then silently skips mask post-processing -- the run continues but the masks differ. Keep the SAM 2 checkout on a local, exec-capable filesystem.