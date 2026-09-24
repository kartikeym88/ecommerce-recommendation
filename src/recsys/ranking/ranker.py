from src.recsys.domain import Recommendation
from src.recsys.explain.llm import LLMExplainer

class Ranker:
    def __init__(self):
        self.explainer = LLMExplainer()
        
    def rank(self, scores_dict, breakdowns, strategy, user_history, products_df, n, explainer=None, user_id=None):
        purchased = set(user_history)
        
        candidates = []
        for pid, score in scores_dict.items():
            if pid in purchased: continue
            candidates.append((pid, score))
            
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        
        history_names = []
        for pid in user_history:
            name_series = products_df.loc[products_df["product_id"] == pid, "name"]
            if not name_series.empty:
                history_names.append(name_series.iloc[0])
        
        recs = []
        cat_counts = {}
        for pid, score in candidates:
            if len(recs) >= n: break
            
            prod_row = products_df[products_df["product_id"] == pid].iloc[0]
            cat = prod_row["category"]
            name = prod_row["name"]
            
            if cat_counts.get(cat, 0) >= 3: continue
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
            rec = Recommendation(
                product_id=pid,
                product_name=name,
                category=cat,
                score=round(float(score), 4),
                breakdown=breakdowns[pid] if breakdowns else None,
                evidence=None,
                reason=""
            )
            rec.reason = self.explainer.generate_explanation(history_names, rec)
            recs.append(rec)
            
        return recs, strategy
