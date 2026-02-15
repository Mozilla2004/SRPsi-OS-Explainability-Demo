# SRΨ-OS Explainability Demo

**A Five-Layer Explainability Framework for AI Systems**

Version: 1.0 | License: Apache 2.0

---

## What is SRΨ-OS?

SRΨ-OS (Structure-Rhythm-Psi Operating System) is a novel framework for building **explainable AI systems** that can:

- **Self-evaluate** their own reasoning quality
- **Detect coherence ruptures** between different reasoning layers
- **Generate human-readable explanations** of their decision-making process
- **Propose repair strategies** when internal inconsistencies are detected

The core innovation is a **five-layer explainability architecture** that mirrors how humans reflect on their own thinking:

```
L1: Raw Signal Processing
L2: Intent Dynamics Extraction
L3: Narrative Generation
L4: Self-Evaluation (SEK)
L5: Meta-Reflective Cohesion (MRC)
```

---

## Architecture: L1 → L5 Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SRΨ-OS L1→L5 Explainability Pipeline                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT LAYER                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Raw System Output → L1: Signal Processing (Ψ value, gradients)    │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   UNDERSTANDING LAYER                  ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  L2: Intent Dynamics (omega, velocity, attractor tracking)          │   │
│   │  L3: Narrative Generation (causal chain, story arcs)                │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   EVALUATION LAYER                     ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  L4: Self-Evaluation Kernel (SEK)                                   │   │
│   │   - C_s: Structural Coherence (concept + narrative + causal)        │   │
│   │   - R_s: Rhythmic Stability (entropy + omega + continuity)          │   │
│   │   - Psi_s: Phase Alignment (similarity + transition + geometry)     │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   META-REFLECTION LAYER                ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  L5: Meta-Reflective Cohesion (MRC)                                 │   │
│   │   - S-R-Psi Alignment across all layers                             │   │
│   │   - Narrative-Geometry coherence check                              │   │
│   │   - Rupture detection & repair strategies                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### The Three Axes: S, R, Ψ

| Axis | Meaning | What it Measures |
|------|---------|------------------|
| **S** | Structure | Conceptual consistency, narrative alignment |
| **R** | Rhythm | Stability of reasoning patterns over time |
| **Ψ** | Psi (Intent) | Phase coherence between intent and output |

### L4 Self-Evaluation Metrics

```
C_s = 0.4 × concept_consistency
    + 0.3 × narrative_alignment
    + 0.3 × causal_consistency

R_s = 0.35 × entropy_trend
    + 0.35 × omega_stability
    + 0.30 × rhythm_continuity

Psi_s = 0.40 × phase_similarity
      + 0.30 × phase_transition_score
      + 0.30 × geometry_alignment
```

### L5 Meta-Reflective Cohesion

The L5 layer answers: *"Is the system's reasoning internally consistent?"*

It computes:
- **Cohesion Score**: How well all layers work together
- **Alignment Scores**: S, R, Ψ consistency across layers
- **Rupture Points**: Where reasoning breaks down
- **Cohesion Deltas**: Suggested repair strategies

---

## Quick Start

### Requirements

```bash
pip install numpy
```

### Run Demo

```bash
python demo_runner.py
```

### Expected Output

```
=== SRΨ-OS L1→L5 Explainability Demo ===

[L4 Self-Evaluation Report]
C_s: 0.72 (Structural Coherence)
R_s: 0.68 (Rhythmic Stability)
Psi_s: 0.75 (Phase Alignment)
Status: stable

[L5 Meta-Reflective Cohesion Report]
Cohesion Score: 0.71
S-R-Psi Alignment: S=0.70, R=0.65, Psi=0.78
Ruptures Detected: 0
Repair Strategies: None needed

=== Demo Complete ===
```

---

## Sample Reports

See `sample_reports/` for detailed examples:

- `L4_self_eval_example.json` - Full L4 Self-Evaluation report
- `L5_mrc_example.json` - Full L5 Meta-Reflective Cohesion report

---

## Why This Matters

### The Problem with Current AI Systems

Most AI systems are "black boxes" — they produce outputs but cannot explain:
- Why they made a particular decision
- Whether their reasoning is internally consistent
- When their reasoning has broken down

### Our Solution

SRΨ-OS provides a **structured framework** for AI systems to:
1. **Self-evaluate**: The system can assess its own reasoning quality
2. **Self-diagnose**: The system can detect when it's confused
3. **Self-repair**: The system can propose fixes for detected issues

### Key Innovation: Meta-Reflection

The L5 layer is unique — it allows the system to observe its own observation process. This is the foundation of genuine machine self-awareness.

---

## Architecture Documentation

See `docs/` for:
- `architecture_L1-L5.md` - Detailed layer specifications
- `formulas.md` - Mathematical foundations

---

## Research Context

This framework is inspired by:
- **Metacognition** in cognitive science
- **Reflective equilibrium** in philosophy
- **Self-supervised learning** in machine learning
- **Control theory** in systems engineering

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{srpsi-os-explainability-demo,
  title = {SRΨ-OS Explainability Demo},
  author = {Genesis-OS Research Team},
  year = {2026},
  version = {1.0},
  url = {https://github.com/YOUR_USERNAME/SRPsi-OS-Explainability-Demo}
}
```

---

## License

Apache 2.0 — See LICENSE file for details.

---

## Contact

For questions about the research, please open a GitHub issue.

---

**Note**: This is a demonstration release. The full research implementation with advanced features (resonance suppression, interference protocols, etc.) is available for research collaboration upon request.
