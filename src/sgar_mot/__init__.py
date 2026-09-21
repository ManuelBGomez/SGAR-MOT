"""SGAR-MOT: Segmentation-Guided Association Refinement in Multiple Object Tracking.

The package holds the two components introduced by the paper:

- sgar_mot.mir  -- the Mask Instance Resolver (MIR), a ViT-based network that
  decides whether three time-aligned masks belong to the same object.
- sgar_mot.orp  -- the Object Recovery Protocol (ORP), which couples SAM 2
  with MIR to recover tracks left unmatched by the association step.
"""
