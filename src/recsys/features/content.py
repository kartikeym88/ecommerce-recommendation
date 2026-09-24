import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.recsys.config import settings

class ContentFeatureBuilder:
    def __init__(self, tfidf_params=None):
        params = tfidf_params or settings.tfidf_params.model_dump()
        if "ngram_range" in params and isinstance(params["ngram_range"], list):
            params["ngram_range"] = tuple(params["ngram_range"])
        self.vectorizer = TfidfVectorizer(**params)
        self.product_matrix = None
        self.product_ids = None
        self._product_to_idx = {}
        
    def _build_text(self, row: pd.Series) -> str:
        parts = [
            str(row.get("name", "")),
            str(row.get("category", "")),
            str(row.get("subcategory", "")),
            str(row.get("brand", "")),
            str(row.get("tags", "")).replace("|", " "),
            str(row.get("description", ""))
        ]
        return " ".join(parts)
        
    def fit(self, products_df: pd.DataFrame):
        self.product_ids = products_df["product_id"].values
        self._product_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        corpus = products_df.apply(self._build_text, axis=1).tolist()
        self.product_matrix = self.vectorizer.fit_transform(corpus)
        
    def transform_new(self, product_row: pd.Series) -> np.ndarray:
        text = self._build_text(product_row)
        return self.vectorizer.transform([text])
        
    def get_vector(self, product_id: int):
        idx = self._product_to_idx.get(product_id)
        if idx is None:
            return None
        return self.product_matrix[idx]
