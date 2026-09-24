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

def main():
    print("Loading data...")
    repo = CsvRepository(settings.paths.data_dir)
    try:
        users = repo.load_users()
        products = repo.load_products()
        interactions = repo.load_interactions()
    except Exception as e:
        print("Data not found or invalid. Please run 'make data' first.")
        return
        
    print("Processing data...")
    processor = DataProcessor()
    processed = processor.run(users, products, interactions)
    matrix = InteractionMatrixBuilder().build(processed.strength_df)
    
    print("Building features...")
    features = ContentFeatureBuilder()
    features.fit(products)
    
    print("Training models...")
    pop = PopularityRecommender()
    cf = ItemCFRecommender(top_k_neighbors=20)
    content = ContentBasedRecommender()
    
    hybrid = HybridRecommender(cf, content, pop)
    hybrid.fit(matrix, products, features)
    
    ranker = Ranker()
    
    user_id = 1
    user_history = interactions[interactions["user_id"] == user_id]["product_id"].unique()
    num_interactions = len(user_history)
    
    print(f"\nGenerating recommendations for User {user_id} (History size: {num_interactions})...")
    scores_dict, breakdowns, strategy = hybrid.score_user(user_id, num_interactions)
    
    recs, _ = ranker.rank(scores_dict, breakdowns, strategy, user_history, products, n=5)
    
    print(f"\nStrategy used: {strategy.upper()}")
    print("-" * 50)
    for i, rec in enumerate(recs, 1):
        print(f"#{i}: {rec.product_name} ({rec.category}) - Score: {rec.score:.4f}")
        bd = rec.breakdown
        print(f"    [CF: {bd.cf:.2f}, Content: {bd.content:.2f}, Pop: {bd.popularity:.2f}]")
    print("-" * 50)

if __name__ == "__main__":
    main()
