SGAR-MOT is released under the PolyForm Noncommercial License 1.0.0 (SPDX identifier: PolyForm-Noncommercial-1.0.0), reproduced in full below.

Required Notice: Copyright (c) 2026 Universidade de Santiago de Compostela

WHAT THESE TERMS COVER
----------------------

These terms cover the code written for SGAR-MOT, that is, everything in this repository except the third-party files listed in the next section:

  * src/sgar_mot/       -- MIR and ORP (except the file noted below)
  * src/bytetrack/      -- the tracker, evaluator, entry points and exps
  * src/configs/, docker/, docs/, requirements.txt, pyproject.toml

The released MIR checkpoint (pretrained/mir_sgar_mot.pth) is part of "the software" as these terms use the word, and is covered by them on the same footing as the code, so that one definition of "noncommercial purpose" governs the whole release. To the extent that the trained weights are protected by copyright or related rights, they are licensed under these terms; where they are not so protected, the authors ask that you respect the same limits. The weights were trained on MOTSynth, whose own terms are noncommercial.


WHAT THESE TERMS DO NOT COVER
-----------------------------

Third-party code is redistributed here under its own licence, and nothing in this file restricts it:

  * src/sam2_sgar_mot/

        SAM 2, Copyright (c) Meta Platforms, Inc. and affiliates, with our modifications marked [SGAR-MOT]. Apache License 2.0; the licence text is in src/sam2_sgar_mot/LICENSE.

  * src/sgar_mot/mir/positional_encodings/torch_encodings.py

        Copyright (c) 2020 Peter Tatkowski. MIT licence; see licenses/multidim-positional-encoding-MIT.txt.

  * portions of src/bytetrack/yolox/tracker/sgar_tracker.py,
    src/bytetrack/yolox/evaluators/sgar_evaluator.py,
    src/bytetrack/tools/track_sgar.py and src/bytetrack/exps/example/mot/*.py

        Derived from ByteTrack, Copyright (c) 2021 Yifu Zhang (MIT, see licenses/ByteTrack-MIT.txt), and, in the evaluator, from YOLOX Copyright (c) Megvii, Inc. and its affiliates (Apache License 2.0, see licenses/Apache-2.0.txt). Each file states in its header which parts come from where. Those parts remain available to you under their original licences.

THIRD_PARTY.md records every component in detail.

----------------------
----------------------
<br></br>

# PolyForm Noncommercial License 1.0.0

<https://polyformproject.org/licenses/noncommercial/1.0.0>

## Acceptance

In order to get any license under these terms, you must agree to them as both strict obligations and conditions to all your licenses.

## Copyright License

The licensor grants you a copyright license for the software to do everything you might do with the software that would otherwise infringe the licensor's copyright in it for any permitted purpose.  However, you may only distribute the software according to [Distribution License](#distribution-license) and make changes or new works based on the software according to [Changes and New Works License](#changes-and-new-works-license).

## Distribution License

The licensor grants you an additional copyright license to distribute copies of the software. Your license to distribute covers distributing the software with changes and new works permitted by [Changes and New Works License](#changes-and-new-works-license).

## Notices

You must ensure that anyone who gets a copy of any part of the software from you also gets a copy of these terms or the URL for them above, as well as copies of any plain-text lines beginning with `Required Notice:` that the licensor provided with the software.  For example:

> Required Notice: Copyright Yoyodyne, Inc. (http://example.com)

## Changes and New Works License

The licensor grants you an additional copyright license to make changes and new works based on the software for any permitted purpose.

## Patent License

The licensor grants you a patent license for the software that covers patent claims the licensor can license, or becomes able to license, that you would infringe by using the software.

## Noncommercial Purposes

Any noncommercial purpose is a permitted purpose.

## Personal Uses

Personal use for research, experiment, and testing for the benefit of public knowledge, personal study, private entertainment, hobby projects, amateur pursuits, or religious observance, without any anticipated commercial application, is use for a permitted purpose.

## Noncommercial Organizations

Use by any charitable organization, educational institution, public research organization, public safety or health organization, environmental protection organization, or government institution is use for a permitted purpose regardless of the source of funding or obligations resulting from the funding.

## Fair Use

You may have "fair use" rights for the software under the law. These terms do not limit them.

## No Other Rights

These terms do not allow you to sublicense or transfer any of your licenses to anyone else, or prevent the licensor from granting licenses to anyone else.  These terms do not imply any other licenses.

## Patent Defense

If you make any written claim that the software infringes or contributes to infringement of any patent, your patent license for the software granted under these terms ends immediately. If your company makes such a claim, your patent license ends immediately for work on behalf of your company.

## Violations

The first time you are notified in writing that you have violated any of these terms, or done anything with the software not covered by your licenses, your licenses can nonetheless continue if you come into full compliance with these terms, and take practical steps to correct past violations, within 32 days of receiving notice.  Otherwise, all your licenses end immediately.

## No Liability

***As far as the law allows, the software comes as is, without any warranty or condition, and the licensor will not be liable to you for any damages arising out of these terms or the use or nature of the software, under any kind of legal claim.***

## Definitions

The **licensor** is the individual or entity offering these terms, and the **software** is the software the licensor makes available under these terms.

**You** refers to the individual or entity agreeing to these terms.

**Your company** is any legal entity, sole proprietorship, or other kind of organization that you work for, plus all organizations that have control over, are under the control of, or are under common control with that organization.  **Control** means ownership of substantially all the assets of an entity, or the power to direct its management and policies by vote, contract, or otherwise.  Control can be direct or indirect.

**Your licenses** are all the licenses granted to you for the software under these terms.

**Use** means anything you do with the software requiring one of your licenses.
