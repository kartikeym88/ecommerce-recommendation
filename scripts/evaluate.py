import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.recsys.config import settings
from src.recsys.data.repository import CsvRepository
from src.recsys.models.hybrid import HybridRecommender
from src.recsys.ranking.ranker import Ranker

def compute_metrics(recs, ground_truth, k=10):
    recs_k = recs[:k]
    hits = 0
    dcg = 0.0
    
    for i, rec in enumerate(recs_k):
        if rec.product_id in ground_truth:
            hits += 1
            dcg += 1.0 / np.log2(i + 2)
            
    # IDCG (Ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    
    recall = hits / len(ground_truth) if ground_truth else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return recall, ndcg

def main():
    print("Loading data for evaluation...")
    repo = CsvRepository(settings.paths.data_dir)
    interactions = repo.load_interactions()
    products = repo.load_products()
    
    print("Creating train/test temporal split...")
    # 80/20 temporal split
    interactions = interactions.sort_values("timestamp")
    split_idx = int(len(interactions) * 0.8)
    train_df = interactions.iloc[:split_idx]
    test_df = interactions.iloc[split_idx:]
    
    # Get ground truth: items users interacted with in the test set
    print("Extracting ground truth...")
    ground_truth = test_df.groupby("user_id")["product_id"].apply(set).to_dict()
    
    # Only evaluate on users who exist in both train and test
    train_users = set(train_df["user_id"].unique())
    eval_users = [u for u in ground_truth.keys() if u in train_users]
    
    print(f"Training Hybrid Recommender on {len(train_df)} interactions...")
    print(f"Evaluating {len(eval_users)} users (K=10)...")
    
    from src.recsys.preprocessing.processor import DataProcessor
    from src.recsys.preprocessing.matrix import InteractionMatrixBuilder
    from src.recsys.features.content import ContentFeatureBuilder
    from src.recsys.models.popularity import PopularityRecommender
    from src.recsys.models.item_cf import ItemCFRecommender
    from src.recsys.models.content_based import ContentBasedRecommender
    
    processor = DataProcessor()
    processed = processor.run(repo.load_users(), products, train_df)
    matrix = InteractionMatrixBuilder().build(processed.strength_df)
    
    features = ContentFeatureBuilder()
    features.fit(products)
    
    pop = PopularityRecommender()
    cf = ItemCFRecommender(top_k_neighbors=20)
    content = ContentBasedRecommender()
    
    recommender = HybridRecommender(cf, content, pop)
    recommender.fit(matrix, products, features)
    
    ranker = Ranker()
    
    total_recall = 0.0
    total_ndcg = 0.0
    valid_users = 0
    
    print(f"Evaluating {len(eval_users)} users (K=10)...")
    for user_id in tqdm(eval_users):
        history = train_df[train_df["user_id"] == user_id]["product_id"].tolist()
        if not history:
            continue
            
        scores, breakdowns, strategy = recommender.score_user(user_id, len(history))
        
        recs, _ = ranker.rank(scores, breakdowns, strategy, history, products, n=10, user_id=user_id)
        
        recall, ndcg = compute_metrics(recs, ground_truth[user_id], k=10)
        total_recall += recall
        total_ndcg += ndcg
        valid_users += 1
        
    final_recall = total_recall / valid_users if valid_users > 0 else 0
    final_ndcg = total_ndcg / valid_users if valid_users > 0 else 0
    
    print("\n" + "="*50)
    print("OFFLINE EVALUATION METRICS")
    print("="*50)
    print(f"Algorithm Strategy: HYBRID (CF + Content + Popularity)")
    print(f"Recall@10: {final_recall:.4f}")
    print(f"NDCG@10:   {final_ndcg:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
