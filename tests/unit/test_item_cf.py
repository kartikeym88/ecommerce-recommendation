import pandas as pd
import numpy as np
from src.recsys.preprocessing.matrix import InteractionMatrix
from scipy.sparse import csr_matrix
from src.recsys.models.item_cf import ItemCFRecommender

def test_item_cf():
    csr = csr_matrix(np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0]
    ]))
    idx_to_item = {0: 10, 1: 20, 2: 30}
    item_to_idx = {10: 0, 20: 1, 30: 2}
    mat = InteractionMatrix(csr, {1:0, 2:1, 3:2}, {0:1, 1:2, 2:3}, item_to_idx, idx_to_item)
    
    products = pd.DataFrame([
        {"product_id": 10, "name": "A"}, 
        {"product_id": 20, "name": "B"}, 
        {"product_id": 30, "name": "C"}
    ])
    
    rec = ItemCFRecommender()
    rec.fit(mat, products)
    
    assert rec.similarity[0, 1] > 0
    assert rec.similarity[0, 2] == 0
    
    scores = rec.score_user(3)
    assert scores[20] > 0
    assert scores[10] == 0
    
    ev = rec.explain(3, 20)
    assert ev.kind == "cf"
    assert ev.source_product_ids == [30]
    assert ev.source_names == ["C"]
