from dataclasses import dataclass, field
from typing import Literal, List

@dataclass(frozen=True)
class ScoreBreakdown:
    cf: float
    content: float
    popularity: float

@dataclass(frozen=True)
class Evidence:
    source_product_ids: List[int]
    source_names: List[str]
    kind: Literal["cf", "content", "popular", "none"]

@dataclass(frozen=True)
class Recommendation:
    product_id: int
    product_name: str
    category: str
    score: float
    breakdown: ScoreBreakdown
    evidence: Evidence
    reason: str
