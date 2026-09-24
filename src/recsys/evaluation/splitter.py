import pandas as pd

class TemporalSplitter:
    def split(self, interactions: pd.DataFrame, test_fraction: float = 0.2, min_train: int = 5):
        df = interactions.sort_values("timestamp").copy()
        
        train_dfs = []
        test_dfs = []
        
        for user_id, group in df.groupby("user_id"):
            n_events = len(group)
            n_test = int(n_events * test_fraction)
            
            if n_events - n_test >= min_train and n_test > 0:
                train_part = group.iloc[:-n_test]
                test_part = group.iloc[-n_test:]
                
                test_relevant = test_part[
                    (test_part["event_type"].isin(["purchase", "like"])) |
                    ((test_part["event_type"] == "rating") & (test_part["rating"] >= 4.0))
                ]
                
                if len(test_relevant) > 0:
                    train_dfs.append(train_part)
                    test_dfs.append(test_part)
                    continue
            
            train_dfs.append(group)
            
        train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame(columns=interactions.columns)
        test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame(columns=interactions.columns)
        
        return train_df, test_df
