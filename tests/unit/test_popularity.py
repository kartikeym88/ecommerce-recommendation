import pandas as pd
import numpy as np
from src.recsys.preprocessing.matrix import InteractionMatrix
from scipy.sparse import csr_matrix
from src.recsys.models.popularity import PopularityRecommender

def test_popularity():
    csr = csr_matrix(np.array([
        [1.0, 0.0, 5.0],
        [0.0, 2.0, 5.0]
    ]))
    idx_to_item = {0: 10, 1: 20, 2: 30}
    item_to_idx = {10: 0, 20: 1, 30: 2}
    mat = InteractionMatrix(csr, {1:0, 2:1}, {0:1, 1:2}, item_to_idx, idx_to_item)
    
    products = pd.DataFrame([{"product_id": 10}, {"product_id": 20}, {"product_id": 30}])
    
    rec = PopularityRecommender()
    rec.fit(mat, products)
    
    scores = rec.score_user(1)
    assert scores[30] == 1.0
    assert scores[20] == 0.2
    assert scores[10] == 0.1
