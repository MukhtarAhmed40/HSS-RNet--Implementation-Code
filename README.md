# HSS-RNet: Scalable Long-Range Sequential Modeling for Network Intrusion Detection

Official PyTorch implementation of **HSS-RNet**, a hybrid selective state-space framework for scalable network intrusion detection and secure traffic analytics.

## Overview

HSS-RNet integrates:
- **Selective State-Space (Mamba) Backbone** — Efficient linear-time sequential modeling with input-dependent state transitions
- **BiGRU Refinement Module** — Recovers fine-grained local temporal patterns oversmoothed by linear SSM dynamics
- **Traffic-Aware Adaptive Gating (TAAG)** — Dynamically emphasizes informative temporal patterns and suppresses noise

The framework also includes **S-SSM**, a structured state-space counterpart with DPLR parameterization and HiPPO initialization, enabling systematic comparison of fixed vs. selective state-space dynamics.

## Architecture
