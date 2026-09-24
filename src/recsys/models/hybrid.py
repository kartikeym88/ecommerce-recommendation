import pandas as pd
from src.recsys.domain import ScoreBreakdown
from src.recsys.config import settings

class HybridRecommender:
    def __init__(self, cf, content, popularity):
        self.cf = cf
        self.content = content
        self.popularity = popularity
        self.weights = settings.hybrid_weights.model_dump()
        self.cold_start_threshold = settings.cold_start_threshold
        
    def fit(self, matrix, products, features):
        self.cf.fit(matrix, products)
        self.content.fit(matrix, products, features)
        self.popularity.fit(matrix, products)
        
    def strategy_for(self, user_id: int, num_interactions: int) -> str:
        if num_interactions == 0:
            return "popularity_fallback"
        elif num_interactions < self.cold_start_threshold:
            return "content_popularity"
        return "hybrid"
        
    def candidates(self, user_id: int, pool_size: int = 100):
        return self.popularity.products["product_id"].tolist()
        
    def score_user(self, user_id: int, num_interactions: int):
        strategy = self.strategy_for(user_id, num_interactions)
        
        pop_scores = self.popularity.score_user(user_id)
        if strategy == "popularity_fallback":
            breakdowns = {item: ScoreBreakdown(0.0, 0.0, 1.0) for item in pop_scores.index}
            return pop_scores.to_dict(), breakdowns, strategy
            
        content_scores = self.content.score_user(user_id)
        cf_scores = self.cf.score_user(user_id) if strategy == "hybrid" else pd.Series(0.0, index=pop_scores.index)
        
        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9) if s.max() > s.min() else s
        
        pop_norm = norm(pop_scores)
        content_norm = norm(content_scores)
        cf_norm = norm(cf_scores)
        
        if strategy == "hybrid":
            w_cf, w_cont, w_pop = self.weights["cf"], self.weights["content"], self.weights["popularity"]
        else:
            w_cf, w_cont, w_pop = 0.0, 0.6, 0.4
            
        final_scores = w_cf * cf_norm + w_cont * content_norm + w_pop * pop_norm
        
        breakdowns = {
            item: ScoreBreakdown(cf=cf_norm.get(item,0), content=content_norm.get(item,0), popularity=pop_norm.get(item,0))
            for item in final_scores.index
        }
        
        return final_scores.to_dict(), breakdowns, strategy
