#!/usr/bin/env python3
"""
SRPsi-OS Explainability Demo Runner v1.0

A minimal demonstration of the L1→L5 explainability pipeline.
This is a simplified version for public release.

For the full research implementation, contact the research team.

Usage:
    python demo_runner.py
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


# =============================================================================
# SECTION 1: SIMULATED L1-L3 OUTPUT (Demo Only)
# =============================================================================

@dataclass
class SimulatedIntentDynamics:
    """Simulated output from L2 (Intent Dynamics)"""
    psi_value: float = 0.35
    omega_value: float = 0.22
    trend: str = "converging"
    velocity: float = 0.08


@dataclass
class SimulatedNarrative:
    """Simulated output from L3 (Narrative Generation)"""
    headline: str = "System converging toward stable attractor"
    sections: List[Dict] = field(default_factory=lambda: [
        {"title": "Current State", "content": "Intent dynamics show converging trend"},
        {"title": "Trajectory", "content": "Low omega indicates high certainty"},
        {"title": "Projection", "content": "Expected to reach basin within 5 cycles"}
    ])


@dataclass
class SimulatedCausalChain:
    """Simulated output from L3 (Causal Chain)"""
    edges: List[Dict] = field(default_factory=lambda: [
        {"from": "intent", "to": "action", "strength": 0.85, "causal_type": "direct"},
        {"from": "action", "to": "outcome", "strength": 0.72, "causal_type": "probabilistic"}
    ])


# =============================================================================
# SECTION 2: L4 SELF-EVALUATION KERNEL (Demo Version)
# =============================================================================

@dataclass
class SelfEvalReport:
    """L4 Self-Evaluation Report"""
    C_s: float  # Structural Coherence
    R_s: float  # Rhythmic Stability
    Psi_s: float  # Phase Alignment
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = ""
    evaluation_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if not self.evaluation_id:
            self.evaluation_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict:
        return asdict(self)


class SelfEvaluationKernel:
    """
    L4 Self-Evaluation Kernel (Demo Version)

    Computes internal quality metrics:
    - C_s: Structural Coherence (concept + narrative + causal consistency)
    - R_s: Rhythmic Stability (entropy trend + omega stability + rhythm continuity)
    - Psi_s: Phase Alignment (phase similarity + geometry alignment)

    NOTE: This is a simplified demo version. The full research implementation
    includes additional components for temporal phase mapping, drift detection,
    and information gain estimation.
    """

    # Weight constants (publicly documented)
    CONCEPT_WEIGHT = 0.4
    NARRATIVE_WEIGHT = 0.3
    CAUSAL_WEIGHT = 0.3

    ENTROPY_WEIGHT = 0.35
    OMEGA_WEIGHT = 0.35
    RHYTHM_WEIGHT = 0.30

    SIMILARITY_WEIGHT = 0.40
    PHASE_SHIFT_WEIGHT = 0.30
    GEOMETRY_WEIGHT = 0.30

    # Thresholds
    IMPROVING_THRESHOLD = 0.75
    DRIFTING_THRESHOLD = 0.40

    def evaluate(
        self,
        intent_dynamics: Optional[SimulatedIntentDynamics] = None,
        narrative: Optional[SimulatedNarrative] = None,
        causal_chain: Optional[SimulatedCausalChain] = None,
    ) -> SelfEvalReport:
        """
        Evaluate system quality from L2-L3 outputs.

        Returns SelfEvalReport with C_s, R_s, Psi_s, status, and suggestions.
        """
        # Use defaults if not provided
        intent = intent_dynamics or SimulatedIntentDynamics()
        narr = narrative or SimulatedNarrative()
        causal = causal_chain or SimulatedCausalChain()

        # Compute C_s: Structural Coherence
        concept_consistency = self._compute_concept_consistency(intent)
        narrative_alignment = self._compute_narrative_alignment(narr)
        causal_consistency = self._compute_causal_consistency(causal)

        C_s = (
            self.CONCEPT_WEIGHT * concept_consistency +
            self.NARRATIVE_WEIGHT * narrative_alignment +
            self.CAUSAL_WEIGHT * causal_consistency
        )

        # Compute R_s: Rhythmic Stability
        entropy_trend = self._compute_entropy_trend(intent)
        omega_stability = self._compute_omega_stability(intent)
        rhythm_continuity = self._compute_rhythm_continuity(intent)

        R_s = (
            self.ENTROPY_WEIGHT * entropy_trend +
            self.OMEGA_WEIGHT * omega_stability +
            self.RHYTHM_WEIGHT * rhythm_continuity
        )

        # Compute Psi_s: Phase Alignment
        phase_similarity = self._compute_phase_similarity(intent, narr)
        phase_transition_score = self._compute_phase_transition(intent)
        geometry_alignment = self._compute_geometry_alignment(narr)

        Psi_s = (
            self.SIMILARITY_WEIGHT * phase_similarity +
            self.PHASE_SHIFT_WEIGHT * phase_transition_score +
            self.GEOMETRY_WEIGHT * geometry_alignment
        )

        # Clamp values
        C_s = max(0.0, min(1.0, C_s))
        R_s = max(0.0, min(1.0, R_s))
        Psi_s = max(0.0, min(1.0, Psi_s))

        # Determine status
        status = self._determine_status(C_s, R_s, Psi_s)

        # Build diagnostics
        diagnostics = {
            "C_s_components": {
                "concept_consistency": round(concept_consistency, 4),
                "narrative_alignment": round(narrative_alignment, 4),
                "causal_consistency": round(causal_consistency, 4),
            },
            "R_s_components": {
                "entropy_trend": round(entropy_trend, 4),
                "omega_stability": round(omega_stability, 4),
                "rhythm_continuity": round(rhythm_continuity, 4),
            },
            "Psi_s_components": {
                "phase_similarity": round(phase_similarity, 4),
                "phase_transition_score": round(phase_transition_score, 4),
                "geometry_alignment": round(geometry_alignment, 4),
            },
        }

        # Generate suggestions
        suggestions = self._generate_suggestions(C_s, R_s, Psi_s, diagnostics)

        return SelfEvalReport(
            C_s=C_s,
            R_s=R_s,
            Psi_s=Psi_s,
            status=status,
            diagnostics=diagnostics,
            suggestions=suggestions,
        )

    # --- C_s Component Methods ---

    def _compute_concept_consistency(self, intent: SimulatedIntentDynamics) -> float:
        """Measure conceptual coherence from intent dynamics."""
        score = 0.5
        if intent.omega_value < 0.25 and intent.trend == "converging":
            score += 0.35
        elif intent.omega_value < 0.35:
            score += 0.20
        if intent.psi_value < 0.4:
            score += 0.15
        return max(0.0, min(1.0, score))

    def _compute_narrative_alignment(self, narrative: SimulatedNarrative) -> float:
        """Measure how well narrative sections align."""
        score = 0.5
        score += min(0.25, len(narrative.sections) * 0.08)
        if narrative.headline:
            score += 0.10
        return max(0.0, min(1.0, score))

    def _compute_causal_consistency(self, causal: SimulatedCausalChain) -> float:
        """Measure causal chain coherence."""
        score = 0.5
        if causal.edges:
            strengths = [e.get("strength", 0.5) for e in causal.edges]
            avg_strength = sum(strengths) / len(strengths)
            score += (avg_strength - 0.5) * 0.4
        return max(0.0, min(1.0, score))

    # --- R_s Component Methods ---

    def _compute_entropy_trend(self, intent: SimulatedIntentDynamics) -> float:
        """Measure stability from intent trend."""
        if intent.trend == "stable":
            return 0.9
        elif intent.trend == "converging":
            return 0.8
        elif intent.trend == "oscillating":
            return 0.5
        else:
            return 0.4

    def _compute_omega_stability(self, intent: SimulatedIntentDynamics) -> float:
        """Lower omega = higher stability."""
        return 1.0 - min(1.0, intent.omega_value * 2)

    def _compute_rhythm_continuity(self, intent: SimulatedIntentDynamics) -> float:
        """Measure rhythm continuity from velocity."""
        if intent.velocity < 0.1:
            return 0.85
        elif intent.velocity < 0.2:
            return 0.70
        else:
            return 0.55

    # --- Psi_s Component Methods ---

    def _compute_phase_similarity(self, intent: SimulatedIntentDynamics, narr: SimulatedNarrative) -> float:
        """Measure phase alignment between intent and narrative."""
        score = 0.5
        if intent.psi_value < 0.4:
            if any(kw in narr.headline.lower() for kw in ["converging", "stable", "good"]):
                score += 0.30
        return max(0.0, min(1.0, score))

    def _compute_phase_transition(self, intent: SimulatedIntentDynamics) -> float:
        """Measure phase stability."""
        if intent.trend == "stable" and intent.omega_value < 0.25:
            return 0.9
        elif intent.trend == "converging":
            return 0.8
        else:
            return 0.6

    def _compute_geometry_alignment(self, narr: SimulatedNarrative) -> float:
        """Measure narrative-geometry alignment."""
        score = 0.5
        score += min(0.25, len(narr.sections) * 0.08)
        return max(0.0, min(1.0, score))

    # --- Status and Suggestions ---

    def _determine_status(self, C_s: float, R_s: float, Psi_s: float) -> str:
        """Determine overall system status."""
        if C_s > self.IMPROVING_THRESHOLD and R_s > self.IMPROVING_THRESHOLD and Psi_s > self.IMPROVING_THRESHOLD:
            return "improving"
        if C_s < self.DRIFTING_THRESHOLD or R_s < self.DRIFTING_THRESHOLD or Psi_s < self.DRIFTING_THRESHOLD:
            return "drifting"
        return "stable"

    def _generate_suggestions(self, C_s: float, R_s: float, Psi_s: float, diag: Dict) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        if C_s < 0.6:
            suggestions.append("Review concept consistency in intent dynamics")
        if R_s < 0.6:
            suggestions.append("Monitor rhythmic patterns for stability")
        if Psi_s < 0.6:
            suggestions.append("Re-align narrative with current intent state")
        if C_s > 0.8 and R_s > 0.8 and Psi_s > 0.8:
            suggestions.append("System in optimal state - maintain trajectory")
        return suggestions[:4]


# =============================================================================
# SECTION 3: L5 META-REFLECTIVE COHESION (Demo Version)
# =============================================================================

@dataclass
class RupturePoint:
    """Represents a detected rupture in coherence."""
    rupture_type: str
    severity: float
    description: str
    tick: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CohesionDelta:
    """Represents a repair strategy."""
    delta_type: str
    target: str
    reason: str
    priority: int = 3

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MetaReflectiveCohesionReport:
    """L5 Meta-Reflective Cohesion Report"""
    cohesion_score: float
    s_r_psi_alignment: Dict[str, float]
    narrative_geometry_alignment: float
    evaluation_feedback_alignment: float
    rupture_points: List[Dict]
    cohesion_deltas: List[Dict]
    unified_explanation: str
    tick: int = 0
    timestamp: str = ""
    report_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if not self.report_id:
            self.report_id = f"mrc-{str(uuid.uuid4())[:8]}"

    def to_dict(self) -> Dict:
        return asdict(self)


class MetaReflectiveCohesionEngine:
    """
    L5 Meta-Reflective Cohesion Engine (Demo Version)

    Unifies all previous layers (L1-L4) into a single self-consistent
    meta-structure, detecting ruptures and generating repair strategies.

    NOTE: This is a simplified demo version. The full research implementation
    includes resonance suppression, interference protocols, and global
    coherence supervision.
    """

    ALIGNMENT_STRONG = 0.75
    ALIGNMENT_WEAK = 0.40

    def build_report(
        self,
        self_eval_report: SelfEvalReport,
        tick: int = 0,
    ) -> MetaReflectiveCohesionReport:
        """
        Build Meta-Reflective Cohesion Report from L4 output.
        """
        # Compute S-R-Psi alignment from self-evaluation
        s_r_psi_alignment = self._compute_s_r_psi_alignment(self_eval_report)

        # Compute cross-layer alignments (simplified for demo)
        narrative_geometry_alignment = 0.5 + (self_eval_report.Psi_s - 0.5) * 0.6
        eval_alignment = self._compute_eval_alignment(self_eval_report)

        # Detect ruptures
        rupture_points = self._detect_ruptures(self_eval_report, tick)

        # Compute cohesion score
        cohesion_score = self._compute_cohesion_score(
            s_r_psi_alignment,
            narrative_geometry_alignment,
            eval_alignment,
            rupture_points,
        )

        # Generate repair strategies
        cohesion_deltas = self._generate_deltas(rupture_points, s_r_psi_alignment)

        # Generate explanation
        unified_explanation = self._generate_explanation(
            cohesion_score,
            s_r_psi_alignment,
            narrative_geometry_alignment,
            eval_alignment,
            rupture_points,
            cohesion_deltas,
        )

        return MetaReflectiveCohesionReport(
            cohesion_score=cohesion_score,
            s_r_psi_alignment=s_r_psi_alignment,
            narrative_geometry_alignment=narrative_geometry_alignment,
            evaluation_feedback_alignment=eval_alignment,
            rupture_points=[rp.to_dict() if isinstance(rp, RupturePoint) else rp for rp in rupture_points],
            cohesion_deltas=[cd.to_dict() if isinstance(cd, CohesionDelta) else cd for cd in cohesion_deltas],
            unified_explanation=unified_explanation,
            tick=tick,
        )

    def _compute_s_r_psi_alignment(self, report: SelfEvalReport) -> Dict[str, float]:
        """Extract S, R, Psi alignment from self-evaluation."""
        diag = report.diagnostics
        return {
            "S": diag.get("C_s_components", {}).get("concept_consistency", 0.5),
            "R": diag.get("R_s_components", {}).get("rhythm_continuity", 0.5),
            "Psi": diag.get("Psi_s_components", {}).get("phase_similarity", 0.5),
        }

    def _compute_eval_alignment(self, report: SelfEvalReport) -> float:
        """Compute alignment between evaluation and actual state."""
        alignment = 0.5
        if report.C_s > 0.5:
            alignment += 0.15
        if report.R_s > 0.5:
            alignment += 0.15
        if report.Psi_s > 0.5:
            alignment += 0.15
        return max(0.0, min(1.0, alignment))

    def _detect_ruptures(self, report: SelfEvalReport, tick: int) -> List[RupturePoint]:
        """Detect coherence ruptures."""
        ruptures = []

        # Check for status-score mismatch
        if report.status == "improving":
            if report.C_s < 0.75 or report.R_s < 0.75 or report.Psi_s < 0.75:
                ruptures.append(RupturePoint(
                    rupture_type="status_score_mismatch",
                    severity=0.4,
                    description="Status shows 'improving' but scores below threshold",
                    tick=tick,
                ))

        # Check for low coherence
        if report.C_s < 0.4:
            ruptures.append(RupturePoint(
                rupture_type="low_structural_coherence",
                severity=0.6,
                description=f"C_s below threshold: {report.C_s:.2f}",
                tick=tick,
            ))

        return ruptures

    def _compute_cohesion_score(
        self,
        s_r_psi: Dict[str, float],
        narr_geom: float,
        eval_align: float,
        ruptures: List[RupturePoint],
    ) -> float:
        """Compute overall cohesion score."""
        srpsi_avg = sum(s_r_psi.values()) / len(s_r_psi)
        rupture_penalty = min(0.3, len(ruptures) * 0.1)

        cohesion = 0.35 * srpsi_avg + 0.25 * narr_geom + 0.25 * eval_align + 0.15 * (1.0 - rupture_penalty)
        return max(0.0, min(1.0, cohesion))

    def _generate_deltas(
        self,
        ruptures: List[RupturePoint],
        alignment: Dict[str, float],
    ) -> List[CohesionDelta]:
        """Generate repair strategies."""
        deltas = []

        for rp in ruptures:
            if rp.rupture_type == "status_score_mismatch":
                deltas.append(CohesionDelta(
                    delta_type="threshold_adjustment",
                    target="status_determination",
                    reason="Re-evaluate status thresholds",
                    priority=1,
                ))
            elif rp.rupture_type == "low_structural_coherence":
                deltas.append(CohesionDelta(
                    delta_type="weight_rebalance",
                    target="C_s_weights",
                    reason="Structural coherence below acceptable level",
                    priority=0,
                ))

        # Check for weak alignment
        min_align = min(alignment.values())
        if min_align < self.ALIGNMENT_WEAK:
            deltas.append(CohesionDelta(
                delta_type="alignment_check",
                target="cross_layer_sync",
                reason=f"Weak alignment detected: {min_align:.2f}",
                priority=1,
            ))

        return deltas[:4]

    def _generate_explanation(
        self,
        cohesion: float,
        s_r_psi: Dict[str, float],
        narr_geom: float,
        eval_align: float,
        ruptures: List[RupturePoint],
        deltas: List[CohesionDelta],
    ) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"[L5 Meta-Reflection] Cohesion Score: {cohesion:.2%}",
            "",
            "S-R-Psi Alignment:",
        ]

        for axis, score in s_r_psi.items():
            status = "STRONG" if score > self.ALIGNMENT_STRONG else "WEAK" if score < self.ALIGNMENT_WEAK else "MODERATE"
            lines.append(f"  - {axis}: {score:.2%} ({status})")

        lines.extend([
            "",
            "Cross-Layer Alignment:",
            f"  - Narrative-Geometry: {narr_geom:.2%}",
            f"  - Evaluation-Feedback: {eval_align:.2%}",
            "",
            f"Ruptures Detected: {len(ruptures)}",
            f"Repair Strategies: {len(deltas)}",
        ])

        return "\n".join(lines)


# =============================================================================
# SECTION 4: DEMO RUNNER
# =============================================================================

def run_demo():
    """Run the complete L1→L5 explainability demo."""
    print("=" * 60)
    print("    SRΨ-OS L1→L5 Explainability Demo v1.0")
    print("=" * 60)
    print()

    # Step 1: Simulate L2-L3 outputs
    print("[Step 1] Generating simulated L2-L3 outputs...")
    intent_dynamics = SimulatedIntentDynamics()
    narrative = SimulatedNarrative()
    causal_chain = SimulatedCausalChain()
    print(f"  - Intent: psi={intent_dynamics.psi_value}, omega={intent_dynamics.omega_value}")
    print(f"  - Narrative: {narrative.headline[:40]}...")
    print()

    # Step 2: Run L4 Self-Evaluation
    print("[Step 2] Running L4 Self-Evaluation Kernel (SEK)...")
    sek = SelfEvaluationKernel()
    l4_report = sek.evaluate(intent_dynamics, narrative, causal_chain)

    print("-" * 50)
    print("L4 SELF-EVALUATION REPORT")
    print("-" * 50)
    print(f"  C_s (Structural Coherence):  {l4_report.C_s:.4f}")
    print(f"  R_s (Rhythmic Stability):    {l4_report.R_s:.4f}")
    print(f"  Psi_s (Phase Alignment):     {l4_report.Psi_s:.4f}")
    print(f"  Status:                      {l4_report.status}")
    print()

    if l4_report.suggestions:
        print("  Suggestions:")
        for s in l4_report.suggestions:
            print(f"    - {s}")
    print()

    # Step 3: Run L5 Meta-Reflective Cohesion
    print("[Step 3] Running L5 Meta-Reflective Cohesion (MRC)...")
    mrc = MetaReflectiveCohesionEngine()
    l5_report = mrc.build_report(l4_report, tick=0)

    print("-" * 50)
    print("L5 META-REFLECTIVE COHESION REPORT")
    print("-" * 50)
    print(f"  Cohesion Score:          {l5_report.cohesion_score:.4f}")
    print(f"  S-R-Psi Alignment:")
    print(f"    S:   {l5_report.s_r_psi_alignment['S']:.4f}")
    print(f"    R:   {l5_report.s_r_psi_alignment['R']:.4f}")
    print(f"    Psi: {l5_report.s_r_psi_alignment['Psi']:.4f}")
    print(f"  Narrative-Geometry:      {l5_report.narrative_geometry_alignment:.4f}")
    print(f"  Evaluation-Feedback:     {l5_report.evaluation_feedback_alignment:.4f}")
    print(f"  Ruptures Detected:       {len(l5_report.rupture_points)}")
    print(f"  Repair Strategies:       {len(l5_report.cohesion_deltas)}")
    print()

    # Step 4: Print unified explanation
    print("-" * 50)
    print("UNIFIED EXPLANATION")
    print("-" * 50)
    print(l5_report.unified_explanation)
    print()

    # Step 5: Save reports to JSON
    print("[Step 4] Saving reports to sample_reports/...")
    import os
    os.makedirs("sample_reports", exist_ok=True)

    with open("sample_reports/L4_self_eval_demo.json", "w") as f:
        json.dump(l4_report.to_dict(), f, indent=2)
    print("  - sample_reports/L4_self_eval_demo.json")

    with open("sample_reports/L5_mrc_demo.json", "w") as f:
        json.dump(l5_report.to_dict(), f, indent=2)
    print("  - sample_reports/L5_mrc_demo.json")
    print()

    print("=" * 60)
    print("    Demo Complete!")
    print("=" * 60)

    return l4_report, l5_report


if __name__ == "__main__":
    run_demo()
