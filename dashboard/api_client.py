import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.recsys.config import settings
from src.recsys.data.repository import CsvRepository
from src.recsys.preprocessing.processor import DataProcessor
from src.recsys.preprocessing.matrix import InteractionMatrixBuilder
from src.recsys.features.content import ContentFeatureBuilder
from src.recsys.models.popularity import PopularityRecommender
from src.recsys.models.item_cf import ItemCFRecommender
from src.recsys.models.content_based import ContentBasedRecommender
from src.recsys.models.hybrid import HybridRecommender
from src.recsys.ranking.ranker import Ranker

import streamlit as st

@st.cache_resource
def load_system():
    repo = CsvRepository(settings.paths.data_dir)
    users = repo.load_users()
    products = repo.load_products()
    interactions = repo.load_interactions()
    
    processor = DataProcessor()
    processed = processor.run(users, products, interactions)
    matrix = InteractionMatrixBuilder().build(processed.strength_df)
    
    features = ContentFeatureBuilder()
    features.fit(products)
    
    pop = PopularityRecommender()
    cf = ItemCFRecommender(top_k_neighbors=20)
    content = ContentBasedRecommender()
    
    hybrid = HybridRecommender(cf, content, pop)
    hybrid.fit(matrix, products, features)
    
    return users, products, interactions, hybrid, Ranker()

def get_users():
    users, _, _, _, _ = load_system()
    return users.to_dict(orient="records")

def get_activity(user_id: int):
    _, products, interactions, _, _ = load_system()
    user_acts = interactions[interactions["user_id"] == user_id].copy()
    user_acts = user_acts.sort_values("timestamp", ascending=False).head(10)
    user_acts = user_acts.merge(products[["product_id", "name"]], on="product_id", how="left")
    return user_acts.to_dict(orient="records")

def get_recommendations(user_id: int, n: int):
    _, products, interactions, hybrid, ranker = load_system()
    user_history = interactions[interactions["user_id"] == user_id]["product_id"].unique()
    num_interactions = len(user_history)
    
    scores_dict, breakdowns, strategy = hybrid.score_user(user_id, num_interactions)
    recs, _ = ranker.rank(scores_dict, breakdowns, strategy, user_history, products, n=n)
    return recs, strategy
