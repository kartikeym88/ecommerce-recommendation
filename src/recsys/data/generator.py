import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

class SyntheticDataGenerator:
    def __init__(self, n_users: int = 1000, n_products: int = 200, n_days: int = 180, seed: int = 42):
        self.n_users = n_users
        self.n_products = n_products
        self.n_days = n_days
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.segments = ["gamer", "remote_worker", "creator", "fitness", "smart_home"]
        
        self.categories = {
            "gamer": ["Gaming Headset", "Mechanical Keyboard", "Gaming Mouse", "GPU"],
            "remote_worker": ["Office Chair", "Webcam", "Monitor", "USB Hub", "Laptop Stand"],
            "creator": ["Microphone", "Drawing Tablet", "Lighting", "Storage"],
            "fitness": ["Smartwatch", "Earbuds", "Scale"],
            "smart_home": ["Smart Speaker", "Smart Bulb", "Security Camera"]
        }
        
    def generate_products(self) -> pd.DataFrame:
        products = []
        product_id = 1000
        
        for segment, subcategories in self.categories.items():
            for subcat in subcategories:
                for _ in range(self.rng.integers(3, 6)):
                    products.append({
                        "product_id": product_id,
                        "name": f"{subcat} {self.rng.choice(['Pro', 'Max', 'Lite', 'Plus', 'Ultra', 'Elite', 'Basic'])} {self.rng.integers(1, 100)}",
                        "category": segment,
                        "subcategory": subcat,
                        "brand": f"Brand{self.rng.integers(1, 10)}",
                        "price": round(self.rng.uniform(19.99, 499.99), 2),
                        "description": f"A high-quality {subcat.lower()} for your needs.",
                        "tags": f"{segment}|{subcat.replace(' ', '').lower()}",
                        "created_at": pd.Timestamp.now().date() - pd.Timedelta(days=int(self.rng.integers(0, 365)))
                    })
                    product_id += 1
                    
        return pd.DataFrame(products)
        
    def generate_users(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        users = []
        segments = []
        
        for i in range(1, self.n_users + 1):
            segment = self.rng.choice(self.segments)
            users.append({
                "user_id": i,
                "name": f"User {i}",
                "age_group": self.rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"]),
                "country": self.rng.choice(["US", "UK", "CA", "AU", "DE"]),
                "signup_date": pd.Timestamp.now().date() - pd.Timedelta(days=int(self.rng.integers(0, self.n_days)))
            })
            segments.append({
                "user_id": i,
                "segment": segment
            })
            
        return pd.DataFrame(users), pd.DataFrame(segments)
        
    def generate_interactions(self, users_df: pd.DataFrame, products_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
        interactions = []
        interaction_id = 1
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.n_days)
        
        products_by_segment = products_df.groupby("category")["product_id"].apply(list).to_dict()
        all_product_ids = products_df["product_id"].tolist()
        
        for _, user_row in users_df.iterrows():
            user_id = user_row["user_id"]
            segment = segments_df.loc[segments_df["user_id"] == user_id, "segment"].values[0]
            
            n_sessions = self.rng.integers(5, 51)
            
            for _ in range(n_sessions):
                session_time = start_time + timedelta(
                    days=int(self.rng.integers(0, self.n_days)),
                    hours=int(self.rng.integers(0, 24)),
                    minutes=int(self.rng.integers(0, 60))
                )
                
                if self.rng.random() < 0.75 and segment in products_by_segment:
                    pool = products_by_segment[segment]
                else:
                    pool = all_product_ids
                
                if not pool: continue
                
                n_views = self.rng.integers(1, 6)
                viewed_items = self.rng.choice(pool, size=n_views, replace=False) if len(pool) >= n_views else pool
                
                for item_id in viewed_items:
                    interactions.append({
                        "interaction_id": interaction_id,
                        "user_id": user_id,
                        "product_id": item_id,
                        "event_type": "view",
                        "rating": None,
                        "timestamp": session_time
                    })
                    interaction_id += 1
                    
                    if self.rng.random() < 0.4:
                        session_time += timedelta(minutes=int(self.rng.integers(1, 5)))
                        interactions.append({
                            "interaction_id": interaction_id,
                            "user_id": user_id,
                            "product_id": item_id,
                            "event_type": "click",
                            "rating": None,
                            "timestamp": session_time
                        })
                        interaction_id += 1
                        
                        p_action = 0.2 if item_id in products_by_segment.get(segment, []) else 0.05
                        
                        action_roll = self.rng.random()
                        if action_roll < p_action:
                            event_type = "purchase" if self.rng.random() < 0.3 else "like"
                            session_time += timedelta(minutes=int(self.rng.integers(1, 10)))
                            interactions.append({
                                "interaction_id": interaction_id,
                                "user_id": user_id,
                                "product_id": item_id,
                                "event_type": event_type,
                                "rating": None,
                                "timestamp": session_time
                            })
                            interaction_id += 1
                            
                            if event_type == "purchase" and self.rng.random() < 0.5:
                                session_time += timedelta(days=int(self.rng.integers(1, 7)))
                                if item_id in products_by_segment.get(segment, []):
                                    rating = float(self.rng.choice([4, 5, 5]))
                                else:
                                    rating = float(self.rng.choice([1, 2, 3, 4, 5]))
                                    
                                interactions.append({
                                    "interaction_id": interaction_id,
                                    "user_id": user_id,
                                    "product_id": item_id,
                                    "event_type": "rating",
                                    "rating": rating,
                                    "timestamp": session_time
                                })
                                interaction_id += 1
                                
        interactions_df = pd.DataFrame(interactions)
        if len(interactions_df) > 0:
            interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)
            interactions_df["interaction_id"] = range(1, len(interactions_df) + 1)
            
            # Ensure no future timestamps
            now = datetime.now()
            interactions_df.loc[interactions_df["timestamp"] > now, "timestamp"] = now

        return interactions_df

    def generate_all(self):
        products_df = self.generate_products()
        users_df, segments_df = self.generate_users()
        interactions_df = self.generate_interactions(users_df, products_df, segments_df)
        return users_df, products_df, interactions_df, segments_df

    def write(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        users_df, products_df, interactions_df, segments_df = self.generate_all()
        
        users_df.to_csv(raw_dir / "users.csv", index=False)
        products_df.to_csv(raw_dir / "products.csv", index=False)
        interactions_df.to_csv(raw_dir / "interactions.csv", index=False)
        segments_df.to_csv(raw_dir / "_ground_truth_segments.csv", index=False)
