import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.recsys.models.base import BaseRecommender
from src.recsys.preprocessing.matrix import InteractionMatrix
from src.recsys.features.content import ContentFeatureBuilder
from src.recsys.domain import Evidence

class ContentBasedRecommender(BaseRecommender):
    def __init__(self, name="content_based"):
        super().__init__(name)
        self.features = None
        self.matrix = None
        
    def fit(self, matrix: InteractionMatrix, products: pd.DataFrame, features: ContentFeatureBuilder = None):
        self.matrix = matrix
        self.features = features
        self.is_fitted = True
        
    def build_user_profile(self, user_id: int):
        if not self.matrix.has_user(user_id):
            return None
        user_vec = self.matrix.user_vector(user_id)
        if user_vec.sum() == 0:
            return None
            
        profile = (user_vec @ self.features.product_matrix) / user_vec.sum()
        return profile
        
    def score_user(self, user_id: int) -> pd.Series:
        profile = self.build_user_profile(user_id)
        if profile is None:
            return pd.Series(0.0, index=self.features.product_ids)
            
        profile_2d = np.asarray(profile).reshape(1, -1)
        scores = cosine_similarity(profile_2d, self.features.product_matrix).flatten()
        return pd.Series(scores, index=self.features.product_ids)
        
    def similar_items(self, product_id: int, n: int) -> list[int]:
        vec = self.features.get_vector(product_id)
        if vec is None: return []
        scores = cosine_similarity(vec, self.features.product_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:n+1]
        return [self.features.product_ids[i] for i in top_idx if self.features.product_ids[i] != product_id][:n]
        
    def explain(self, user_id: int, product_id: int) -> Evidence:
        return Evidence([], [], "content")
