import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.recsys.models.base import BaseRecommender
from src.recsys.preprocessing.matrix import InteractionMatrix
from src.recsys.domain import Evidence

class ItemCFRecommender(BaseRecommender):
    def __init__(self, name="item_cf", top_k_neighbors=None):
        super().__init__(name)
        self.top_k_neighbors = top_k_neighbors
        self.similarity = None
        self.matrix = None
        self.products = None
        
    def fit(self, matrix: InteractionMatrix, products: pd.DataFrame):
        self.matrix = matrix
        self.products = products
        
        if matrix.csr.shape[1] > 0:
            sim = cosine_similarity(matrix.csr.T, dense_output=True)
            np.fill_diagonal(sim, 0.0)
            
            if self.top_k_neighbors is not None:
                for i in range(sim.shape[0]):
                    row = sim[i]
                    if len(row) > self.top_k_neighbors:
                        cutoff = np.partition(row, -self.top_k_neighbors)[-self.top_k_neighbors]
                        row[row < cutoff] = 0.0
            
            self.similarity = sim
        else:
            self.similarity = np.zeros((0, 0))
            
        self.is_fitted = True
        
    def score_user(self, user_id: int) -> pd.Series:
        if not self.matrix.has_user(user_id):
            return pd.Series(0.0, index=[self.matrix.idx_to_item[i] for i in range(self.matrix.csr.shape[1])])
            
        user_vec = self.matrix.user_vector(user_id)
        scores = user_vec @ self.similarity
        
        return pd.Series(scores, index=[self.matrix.idx_to_item[i] for i in range(self.matrix.csr.shape[1])])
        
    def similar_items(self, product_id: int, n: int) -> list[int]:
        if product_id not in self.matrix.item_to_idx:
            return []
            
        idx = self.matrix.item_to_idx[product_id]
        row = self.similarity[idx]
        
        top_indices = np.argsort(row)[::-1][:n]
        return [self.matrix.idx_to_item[i] for i in top_indices if row[i] > 0]
        
    def explain(self, user_id: int, product_id: int) -> Evidence:
        if not self.matrix.has_user(user_id) or product_id not in self.matrix.item_to_idx:
            return Evidence([], [], "cf")
            
        user_idx = self.matrix.user_to_idx[user_id]
        target_idx = self.matrix.item_to_idx[product_id]
        
        user_row = self.matrix.csr[user_idx].toarray().flatten()
        sim_col = self.similarity[:, target_idx]
        
        contributions = user_row * sim_col
        if contributions.sum() == 0:
            return Evidence([], [], "cf")
            
        top_idx = np.argsort(contributions)[::-1][:2]
        top_idx = [i for i in top_idx if contributions[i] > 0]
        
        source_ids = [self.matrix.idx_to_item[i] for i in top_idx]
        source_names = []
        for pid in source_ids:
            name_series = self.products.loc[self.products["product_id"] == pid, "name"]
            source_names.append(name_series.iloc[0] if not name_series.empty else str(pid))
            
        return Evidence(source_product_ids=source_ids, source_names=source_names, kind="cf")
