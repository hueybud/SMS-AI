"""Citrus RL gym package.

Phase A (this commit): IPC roundtrip with the AIController IpcBackend.
Phase B: BC equivalence test (PyTorch checkpoint over IPC == local ONNX).
Phase C: PPO + teacher-KL learner over batched envs.

See SMS AI/rl_gym_design.md for the design doc.
"""
