import pandas as pd
import numpy as np
from src.recsys.models.base import BaseRecommender
from src.recsys.preprocessing.matrix import InteractionMatrix
from src.recsys.domain import Evidence

class PopularityRecommender(BaseRecommender):
    def __init__(self, name="popularity"):
        super().__init__(name)
        self.scores = None
        self.products = None
        
    def fit(self, matrix: InteractionMatrix, products: pd.DataFrame):
        if matrix.csr.shape[0] == 0:
            item_scores = np.zeros(matrix.csr.shape[1])
        else:
            col_sums = np.array(matrix.csr.sum(axis=0)).flatten()
            max_sum = col_sums.max()
            item_scores = col_sums / max_sum if max_sum > 0 else col_sums
            
        self.scores = pd.Series(item_scores, index=[matrix.idx_to_item[i] for i in range(matrix.csr.shape[1])])
        self.products = products
        self.is_fitted = True
        
    def score_user(self, user_id: int) -> pd.Series:
        return self.scores.copy()
        
    def similar_items(self, product_id: int, n: int) -> list[int]:
        if product_id in self.scores:
            return self.scores.drop(product_id).sort_values(ascending=False).head(n).index.tolist()
        return self.scores.sort_values(ascending=False).head(n).index.tolist()
        
    def explain(self, user_id: int, product_id: int) -> Evidence:
        return Evidence(
            source_product_ids=[],
            source_names=[],
            kind="popular"
        )
