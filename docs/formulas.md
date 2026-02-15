# SRΨ-OS Mathematical Foundations

## Core Concepts

### The Three Axes

The SRΨ-OS framework is built on three orthogonal axes that capture different aspects of reasoning quality:

| Axis | Symbol | Domain | Meaning |
|------|--------|--------|---------|
| Structure | S | [0, 1] | Conceptual consistency and coherence |
| Rhythm | R | [0, 1] | Temporal stability and continuity |
| Psi (Intent) | Ψ | [0, 1] | Phase coherence between intent and output |

---

## L4 Self-Evaluation Formulas

### C_s: Structural Coherence

Measures how well the system's reasoning holds together structurally.

```
C_s = w₁ × concept_consistency + w₂ × narrative_alignment + w₃ × causal_consistency

where:
  w₁ = 0.4  (concept weight)
  w₂ = 0.3  (narrative weight)
  w₃ = 0.3  (causal weight)
  w₁ + w₂ + w₃ = 1
```

**Components**:

- **concept_consistency**: How coherent are the concepts being used?
- **narrative_alignment**: How well do narrative sections align with each other?
- **causal_consistency**: How coherent is the causal chain?

---

### R_s: Rhythmic Stability

Measures how stable the system's reasoning patterns are over time.

```
R_s = w₄ × entropy_trend + w₅ × omega_stability + w₆ × rhythm_continuity

where:
  w₄ = 0.35 (entropy weight)
  w₅ = 0.35 (omega weight)
  w₆ = 0.30 (rhythm weight)
  w₄ + w₅ + w₆ = 1
```

**Components**:

- **entropy_trend**: Is the system becoming more or less ordered?
- **omega_stability**: How stable is the uncertainty measure?
- **rhythm_continuity**: Are there sudden breaks in the reasoning flow?

---

### Psi_s: Phase Alignment

Measures how well the system's output aligns with its intent.

```
Psi_s = w₇ × phase_similarity + w₈ × phase_transition_score + w₉ × geometry_alignment

where:
  w₇ = 0.40 (similarity weight)
  w₈ = 0.30 (transition weight)
  w₉ = 0.30 (geometry weight)
  w₇ + w₈ + w₉ = 1
```

**Components**:

- **phase_similarity**: How similar is current phase to expected phase?
- **phase_transition_score**: Is the system in a stable phase or transitioning?
- **geometry_alignment**: Does narrative geometry match intent geometry?

---

### Status Determination

```
status = "improving"  if C_s > τ₁ AND R_s > τ₁ AND Psi_s > τ₁
status = "drifting"   if C_s < τ₂ OR R_s < τ₂ OR Psi_s < τ₂
status = "stable"     otherwise

where:
  τ₁ = 0.75 (improving threshold)
  τ₂ = 0.40 (drifting threshold)
```

---

## L5 Meta-Reflective Formulas

### Cohesion Score

The overall cohesion score measures how well all layers work together:

```
cohesion = α × s_r_psi_alignment
         + β × narrative_geometry_alignment
         + γ × evaluation_feedback_alignment
         + δ × (1 - rupture_intensity)

where:
  α = 0.35  (S-R-Psi alignment weight)
  β = 0.25  (narrative-geometry weight)
  γ = 0.25  (evaluation-feedback weight)
  δ = 0.15  (rupture penalty weight)
  α + β + γ + δ = 1
```

---

### S-R-Psi Alignment

```
s_r_psi_alignment = {
  "S": concept_consistency from L4,
  "R": rhythm_continuity from L4,
  "Psi": phase_similarity from L4
}

s_r_psi_alignment_avg = mean(S, R, Psi)
```

---

### Rupture Intensity

```
rupture_intensity = (avg_severity + count_penalty) / 2

where:
  avg_severity = mean(rupture.severity for rupture in ruptures)
  count_penalty = min(1.0, len(ruptures) × 0.15)
```

---

## Alignment Thresholds

| Level | Threshold | Interpretation |
|-------|-----------|----------------|
| STRONG | > 0.75 | High alignment, no concerns |
| MODERATE | 0.40 - 0.75 | Acceptable, may need attention |
| WEAK | < 0.40 | Low alignment, intervention needed |

---

## Rupture Severity

| Severity | Range | Action |
|----------|-------|--------|
| LOW | < 0.3 | Monitor only |
| MEDIUM | 0.3 - 0.6 | Suggest repair |
| HIGH | > 0.6 | Immediate attention |

---

## Summary

The SRΨ-OS framework uses weighted combinations of interpretable metrics to:

1. **Self-evaluate** reasoning quality (L4)
2. **Detect inconsistencies** across layers (L5)
3. **Propose repairs** when needed (L5 deltas)

All formulas are designed to be:
- **Transparent**: Weights and thresholds are explicit
- **Interpretable**: Each component has clear semantic meaning
- **Configurable**: Parameters can be adjusted for different use cases
