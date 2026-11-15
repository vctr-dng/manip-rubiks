"""Batched GPU simulation on MuJoCo Warp.

The CPU environments in :mod:`manip_rubiks.environments` remain the reference
implementation (docs/project-draft.md, risk 4). Everything here reproduces their
behaviour for thousands of worlds at once, with PyTorch tensors that share
memory with the Warp simulation.
"""
