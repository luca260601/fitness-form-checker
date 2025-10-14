from typing import List, Type
from .types import ExerciseBase
from .exercises.squat import SquatExercise

ALL: List[Type[ExerciseBase]] = [
    SquatExercise,
]
