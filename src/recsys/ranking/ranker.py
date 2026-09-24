from src.recsys.domain import Recommendation

class Ranker:
    def rank(self, scores_dict, breakdowns, strategy, user_history, products_df, n, explainer=None, user_id=None):
        purchased = set(user_history)
        
        candidates = []
        for pid, score in scores_dict.items():
            if pid in purchased: continue
            candidates.append((pid, score))
            
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        
        recs = []
        cat_counts = {}
        for pid, score in candidates:
            if len(recs) >= n: break
            
            prod_row = products_df[products_df["product_id"] == pid].iloc[0]
            cat = prod_row["category"]
            name = prod_row["name"]
            
            if cat_counts.get(cat, 0) >= 3: continue
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
            recs.append(Recommendation(
                product_id=pid,
                product_name=name,
                category=cat,
                score=round(float(score), 4),
                breakdown=breakdowns[pid] if breakdowns else None,
                evidence=None,
                reason="Generated reason"
            ))
            
        return recs, strategy
