# SRΨ-OS L1→L5 Architecture

## Layer Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SRΨ-OS Five-Layer Explainability                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   L1: SIGNAL PROCESSING                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Input: Raw system signals                                          │   │
│   │  Output: Ψ value, gradients, convergence indicators                 │   │
│   │  Purpose: Convert raw data into structured intent signals           │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   L2: INTENT DYNAMICS                  ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Input: L1 signals                                                  │   │
│   │  Output: omega (uncertainty), velocity, attractor tracking          │   │
│   │  Purpose: Model how intent evolves over time                        │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   L3: NARRATIVE GENERATION             ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Input: L2 dynamics                                                 │   │
│   │  Output: Causal chain, story arcs, narrative geometry               │   │
│   │  Purpose: Generate human-readable explanation of reasoning          │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   L4: SELF-EVALUATION (SEK)            ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Input: L1-L3 outputs                                               │   │
│   │  Output: C_s, R_s, Psi_s, status, suggestions                       │   │
│   │  Purpose: Evaluate system's own reasoning quality                   │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│   L5: META-REFLECTIVE COHESION (MRC)   ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Input: L4 report + L1-L3 outputs                                   │   │
│   │  Output: Cohesion score, alignments, ruptures, repair strategies    │   │
│   │  Purpose: Meta-level coherence check across all layers              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### L1: Signal Processing

**Purpose**: Transform raw system output into structured intent signals.

**Key Outputs**:
- Ψ (Psi) value: Intent strength/intensity
- Gradients: Direction of intent change
- Convergence indicators: Is the system settling?

---

### L2: Intent Dynamics

**Purpose**: Model how intent evolves over time.

**Key Outputs**:
- Omega (ω): Uncertainty measure
- Velocity: Rate of intent change
- Attractor tracking: What stable states is the system moving toward?

---

### L3: Narrative Generation

**Purpose**: Create human-readable explanations of reasoning.

**Key Outputs**:
- Causal chain: What caused what
- Story arcs: Narrative structure of reasoning
- Narrative geometry: Spatial/temporal structure of the explanation

---

### L4: Self-Evaluation Kernel (SEK)

**Purpose**: Evaluate the system's own reasoning quality.

**Key Metrics**:

| Metric | Meaning | Components |
|--------|---------|------------|
| **C_s** | Structural Coherence | Concept consistency + Narrative alignment + Causal consistency |
| **R_s** | Rhythmic Stability | Entropy trend + Omega stability + Rhythm continuity |
| **Psi_s** | Phase Alignment | Phase similarity + Transition score + Geometry alignment |

**Status Rules**:
- `improving`: C_s > 0.75 AND R_s > 0.75 AND Psi_s > 0.75
- `drifting`: C_s < 0.40 OR R_s < 0.40 OR Psi_s < 0.40
- `stable`: Otherwise

---

### L5: Meta-Reflective Cohesion (MRC)

**Purpose**: Check if all layers are working together coherently.

**Key Outputs**:
- **Cohesion Score**: Overall cross-layer coherence
- **S-R-Psi Alignment**: Consistency across the three axes
- **Rupture Points**: Where reasoning breaks down
- **Cohesion Deltas**: Suggested repair strategies

---

## Data Flow

```
Raw Input
    │
    ▼
┌───────┐
│  L1   │ → Ψ, gradients, convergence
└───┬───┘
    │
    ▼
┌───────┐
│  L2   │ → omega, velocity, attractors
└───┬───┘
    │
    ▼
┌───────┐
│  L3   │ → narrative, causal chain, geometry
└───┬───┘
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
┌───────┐         ┌───────┐
│  L4   │         │ L1-L3 │
│ (SEK) │         │ data  │
└───┬───┘         └───┬───┘
    │                 │
    └────────┬────────┘
             │
             ▼
         ┌───────┐
         │  L5   │
         │ (MRC) │
         └───┬───┘
             │
             ▼
    Meta-Reflective Report
```

---

## Key Formulas

### L4 C_s (Structural Coherence)

```
C_s = 0.4 × concept_consistency
    + 0.3 × narrative_alignment
    + 0.3 × causal_consistency
```

### L4 R_s (Rhythmic Stability)

```
R_s = 0.35 × entropy_trend
    + 0.35 × omega_stability
    + 0.30 × rhythm_continuity
```

### L4 Psi_s (Phase Alignment)

```
Psi_s = 0.40 × phase_similarity
      + 0.30 × phase_transition_score
      + 0.30 × geometry_alignment
```

### L5 Cohesion Score

```
cohesion = 0.35 × s_r_psi_alignment_avg
         + 0.25 × narrative_geometry_alignment
         + 0.25 × evaluation_feedback_alignment
         + 0.15 × (1 - rupture_intensity)
```

---

## The Three Axes: S, R, Ψ

| Axis | Name | Meaning | High Value = |
|------|------|---------|--------------|
| **S** | Structure | Conceptual consistency | Clear, coherent reasoning |
| **R** | Rhythm | Temporal stability | Stable, predictable patterns |
| **Ψ** | Psi (Intent) | Phase coherence | Intent well-aligned with output |

---

## Rupture Types

When the L5 layer detects inconsistencies, it categorizes them:

| Rupture Type | Description | Severity |
|--------------|-------------|----------|
| `narrative_geometry_mismatch` | Narrative says X but geometry says Y | Medium |
| `phase_eval_mismatch` | Intent trend contradicts evaluation status | Medium-High |
| `basin_mismatch` | Referenced basin doesn't exist in surface | Low |
| `oscillation_detected` | System cohesion oscillating | Variable |

---

## Cohesion Delta Types (Repair Strategies)

| Delta Type | Target | Purpose |
|------------|--------|---------|
| `delta_weight_adjust` | Alignment weights | Rebalance weights for better alignment |
| `delta_threshold` | Detection thresholds | Adjust sensitivity |
| `delta_sync` | Cross-layer sync | Force re-synchronization |
| `delta_narrative_patch` | Narrative generator | Regenerate narrative |
| `delta_geometry_realign` | Basin tracking | Refresh geometry state |

---

## Design Principles

1. **Layered Reflection**: Each layer reflects on the layer below
2. **Self-Evaluation**: L4 evaluates L1-L3, L5 evaluates L4
3. **Human-Readable Output**: L3 generates natural language explanations
4. **Self-Repair**: L5 proposes repair strategies when ruptures are detected
5. **Principled Defaults**: Status thresholds are transparent and configurable
