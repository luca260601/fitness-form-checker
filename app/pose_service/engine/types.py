from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Frame = Dict[str, Tuple[float, float]]  # {"LEFT_HIP": (x,y), ...} in Normalized-Koordinaten

@dataclass
class Issue:
    code: str
    msg: str
    severity: str = "info"      # info | warn | high
    rep_index: Optional[int] = None

@dataclass
class AnalyzeResult:
    exercise: str
    reps: int
    issues: List[Issue]
    metrics: Dict[str, float]   # z.B. {"knee_min": 60.0}
    keyframes: List[bytes]      # PNG-Bytes (Step 2)

class ExerciseBase:
    name: str = "base"
    required_view: str = "side"  # side | front | any

    def detect_score(self, series: List[Frame]) -> float:
        return 0.0

    def analyze(self, series: List[Frame]) -> AnalyzeResult:
        return AnalyzeResult(exercise=self.name, reps=0, issues=[], metrics={}, keyframes=[])
