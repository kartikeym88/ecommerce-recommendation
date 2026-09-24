import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class InteractionMatrix:
    def __init__(self, csr: csr_matrix, user_to_idx: dict, idx_to_user: dict, item_to_idx: dict, idx_to_item: dict):
        self.csr = csr
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        
    def user_vector(self, user_id: int) -> np.ndarray:
        if user_id not in self.user_to_idx:
            return np.zeros(self.csr.shape[1])
        idx = self.user_to_idx[user_id]
        return self.csr[idx].toarray().flatten()
        
    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_to_idx
        
    def items_of_user(self, user_id: int) -> list[int]:
        if not self.has_user(user_id):
            return []
        idx = self.user_to_idx[user_id]
        row = self.csr[idx]
        item_indices = row.indices
        return [self.idx_to_item[i] for i in item_indices]
        
    @property
    def sparsity(self) -> float:
        total = self.csr.shape[0] * self.csr.shape[1]
        if total == 0: return 0.0
        return 1.0 - (self.csr.nnz / total)

class InteractionMatrixBuilder:
    def build(self, strength_df: pd.DataFrame) -> InteractionMatrix:
        users = strength_df["user_id"].unique()
        items = strength_df["product_id"].unique()
        
        user_to_idx = {u: i for i, u in enumerate(users)}
        idx_to_user = {i: u for i, u in enumerate(users)}
        item_to_idx = {item: i for i, item in enumerate(items)}
        idx_to_item = {i: item for i, item in enumerate(items)}
        
        row_indices = strength_df["user_id"].map(user_to_idx).values
        col_indices = strength_df["product_id"].map(item_to_idx).values
        data = strength_df["strength"].values
        
        csr = csr_matrix((data, (row_indices, col_indices)), shape=(len(users), len(items)))
        
        return InteractionMatrix(csr, user_to_idx, idx_to_user, item_to_idx, idx_to_item)
