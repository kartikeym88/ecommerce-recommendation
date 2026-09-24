import pandas as pd
from src.recsys.preprocessing.matrix import InteractionMatrixBuilder

def test_matrix_builder():
    df = pd.DataFrame([
        {"user_id": 10, "product_id": 100, "strength": 1.5},
        {"user_id": 20, "product_id": 200, "strength": 2.5}
    ])
    
    builder = InteractionMatrixBuilder()
    mat = builder.build(df)
    
    assert mat.csr.shape == (2, 2)
    assert mat.has_user(10)
    assert mat.has_user(20)
    assert not mat.has_user(30)
    
    assert 100 in mat.items_of_user(10)
    assert mat.sparsity == 0.5
