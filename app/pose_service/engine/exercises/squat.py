from typing import List, Dict, Tuple
from ..types import ExerciseBase, AnalyzeResult, Issue
from ..utils import angle, torso_angle

KNEE_OVER_TOE_THRESH = 0.25
TORSO_LEAN_THRESH    = 45.0

class SquatExercise(ExerciseBase):
    name = "squat"
    required_view = "side"

    def detect_score(self, series: List[Dict[str, Tuple[float,float]]]) -> float:
        knees, torsos = [], []
        for fr in series:
            hip = fr.get("LEFT_HIP") or fr.get("RIGHT_HIP")
            knee = fr.get("LEFT_KNEE") or fr.get("RIGHT_KNEE")
            ankle = fr.get("LEFT_ANKLE") or fr.get("RIGHT_ANKLE")
            shoulder = fr.get("LEFT_SHOULDER") or fr.get("RIGHT_SHOULDER")
            if not (hip and knee and ankle and shoulder):
                continue
            knees.append(angle(hip, knee, ankle))
            torsos.append(torso_angle(hip, shoulder))
        if not knees:
            return 0.0
        k_range = max(knees) - min(knees)
        t_max = max(torsos) if torsos else 0.0
        score = 0.0
        if k_range > 40: score += 0.6
        if 10 <= t_max <= 80: score += 0.4
        return score

    def analyze(self, series: List[Dict[str, Tuple[float,float]]]) -> AnalyzeResult:
        knees, torsos = [], []
        hip_minus_knee_at_bottom = None
        max_hip_y = -1.0
        issues: List[Issue] = []

        # (1) Flag einmalig setzen
        knee_over_toe_flag = False

        for fr in series:
            hip = fr.get("LEFT_HIP") or fr.get("RIGHT_HIP")
            knee = fr.get("LEFT_KNEE") or fr.get("RIGHT_KNEE")
            ankle = fr.get("LEFT_ANKLE") or fr.get("RIGHT_ANKLE")
            shoulder = fr.get("LEFT_SHOULDER") or fr.get("RIGHT_SHOULDER")
            heel = fr.get("LEFT_HEEL") or fr.get("RIGHT_HEEL")
            toe = fr.get("LEFT_FOOT_INDEX") or fr.get("RIGHT_FOOT_INDEX")
            if not (hip and knee and ankle and shoulder and heel and toe):
                continue

            knees.append(angle(hip, knee, ankle))
            torsos.append(torso_angle(hip, shoulder))

            # Tiefster Punkt
            if hip[1] > max_hip_y:
                max_hip_y = hip[1]
                hip_minus_knee_at_bottom = hip[1] - knee[1]

            # Knie über Zehen relativ zur Fußlänge
            foot_len = abs(toe[0] - heel[0])
            fdir = 1 if toe[0] >= heel[0] else -1
            knee_forward = (knee[0] - toe[0]) * fdir

            # (2) Nur Flag setzen, NICHT schon Issue anhängen
            if foot_len > 0 and knee_forward > KNEE_OVER_TOE_THRESH * foot_len:
                knee_over_toe_flag = True

        kmin = min(knees) if knees else 60.0
        kmax = max(knees) if knees else 120.0
        tmax = max(torsos) if torsos else 35.0

        # Einmalige Issues nach der Schleife:
        if knee_over_toe_flag:
            issues.append(Issue(
                code="knee_over_toe_high",
                msg="Knie deutlich vor Zehen → mehr Gewicht auf Fersen, Hüfte zuerst zurück.",
                severity="warn"
            ))

        if hip_minus_knee_at_bottom is not None and hip_minus_knee_at_bottom <= 0.0:
            issues.append(Issue(code="low_depth", msg="Tiefe vermutlich nicht unter parallel.", severity="info"))

        if tmax >= TORSO_LEAN_THRESH:
            issues.append(Issue(code="torso_lean_high", msg="Starke Oberkörpervorlage.", severity="info"))

        return AnalyzeResult(
            exercise=self.name,
            reps=1,
            issues=issues,
            metrics={"knee_min": round(kmin,1), "knee_max": round(kmax,1), "torso_max": round(tmax,1)},
            keyframes=[],
        )
