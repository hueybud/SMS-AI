## Overview

This project attempts to build an AI for Super Mario Strikers (SMS) by using Imitation Learning to train on the replays of experts with future plans to enhance the AI-model with Reinforcement Learning

## Architecture

A 2-layer LSTM processes one game frame at a time (440 features: game state + previous action), maintaining hidden state across the match. Its output feeds an autoregressive controller head that predicts each button and stick axis in sequence — each decision conditioning the next — using behavioral cloning on expert replays.