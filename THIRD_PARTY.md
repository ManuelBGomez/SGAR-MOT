# Third-party code

The code written for SGAR-MOT, as well as the released MIR checkpoint, are under the PolyForm Noncommercial License 1.0.0 (see `LICENSE`). Te files below are **not**: they keep the license of the project they come from, which our license does not restrict.

| Files | Component | License |
|---|---|---|
| `src/sam2_sgar_mot/**` | [SAM 2](https://github.com/facebookresearch/sam2) © Meta Platforms, Inc. | Apache 2.0 — `src/sam2_sgar_mot/LICENSE` |
| `src/sgar_mot/mir/positional_encodings/torch_encodings.py` | [multidim-positional-encoding](https://github.com/tatp22/multidim-positional-encoding), © 2020 Peter Tatkowski | MIT — `licenses/multidim-positional-encoding-MIT.txt` |
| parts of `src/bytetrack/**` | [ByteTrack](https://github.com/ifzhang/ByteTrack) © 2021 Yifu Zhang | MIT — `licenses/ByteTrack-MIT.txt` |
| parts of `src/bytetrack/yolox/**` | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), © Megvii, Inc. (inside ByteTrack) | Apache 2.0 — `licenses/Apache-2.0.txt` |

The SAM 2 files are the upstream sources **adding our modifications**, kept so that they can be copied over a SAM 2 checkout; every change is marked with `[SGAR-MOT]` label in place. SGAR-MOT is built on ByteTrack and adopts its association strategy, configuration and detector — in each file a header is provided indicating which parts come from where. **Neither project is redistributed here**: both are installed from their own repositories (see [`docs/TUTORIAL.md`](./docs/TUTORIAL.md)).

Model checkpoints other than MIR, and all datasets (MOT20, SportsMOT, VisDrone, MOTSynth), are not redistributed and carry the terms of the projects they come from.