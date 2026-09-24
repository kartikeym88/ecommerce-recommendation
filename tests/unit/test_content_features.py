import pandas as pd
from src.recsys.features.content import ContentFeatureBuilder
from sklearn.metrics.pairwise import cosine_similarity

def test_content_feature_builder():
    products = pd.DataFrame([
        {"product_id": 1, "name": "Gaming Mouse", "category": "gamer", "subcategory": "Mouse", "brand": "BrandA", "tags": "gamer|mouse", "description": "Good mouse"},
        {"product_id": 2, "name": "Gaming Keyboard", "category": "gamer", "subcategory": "Keyboard", "brand": "BrandA", "tags": "gamer|keyboard", "description": "Good keyboard"},
        {"product_id": 3, "name": "Smart Bulb", "category": "smart_home", "subcategory": "Bulb", "brand": "BrandB", "tags": "smart|bulb", "description": "Bright bulb"}
    ])
    
    builder = ContentFeatureBuilder(tfidf_params={"stop_words": "english"})
    builder.fit(products)
    
    assert builder.product_matrix.shape[0] == 3
    
    from scipy.sparse.linalg import norm
    norms = norm(builder.product_matrix, axis=1)
    for n in norms:
        assert abs(n - 1.0) < 1e-5
        
    v1 = builder.get_vector(1)
    v2 = builder.get_vector(2)
    v3 = builder.get_vector(3)
    
    sim12 = cosine_similarity(v1, v2)[0][0]
    sim13 = cosine_similarity(v1, v3)[0][0]
    
    assert sim12 > sim13
    
    new_prod = pd.Series({"name": "Gaming Headset", "category": "gamer"})
    new_v = builder.transform_new(new_prod)
    assert new_v.shape[1] == builder.product_matrix.shape[1]
