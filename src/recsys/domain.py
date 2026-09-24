from dataclasses import dataclass, field
from typing import Literal, List

@dataclass
class ScoreBreakdown:
    cf: float
    content: float
    popularity: float

@dataclass
class Evidence:
    source_product_ids: List[int]
    source_names: List[str]
    kind: Literal["cf", "content", "popular", "none"]

@dataclass
class Recommendation:
    product_id: int
    product_name: str
    category: str
    score: float
    breakdown: ScoreBreakdown
    evidence: Evidence
    reason: str
