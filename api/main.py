from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dashboard.api_client import get_recommendations, get_activity

app = FastAPI(title="RecSys API", description="AI-Powered Hybrid Recommendation Engine")

class Breakdown(BaseModel):
    cf: float
    content: float
    popularity: float

class RecommendationResponse(BaseModel):
    product_id: int
    product_name: str
    category: str
    score: float
    reason: Optional[str] = None
    breakdown: Optional[Breakdown] = None

class ActivityResponse(BaseModel):
    event_type: str
    product_id: int
    name: str
    timestamp: str

@app.get("/health")
def health():
    return {"status": "ok", "message": "RecSys Engine is running"}

@app.get("/users/{user_id}/recommendations", response_model=List[RecommendationResponse])
def get_user_recommendations(user_id: int, limit: int = 5):
    try:
        recs, strategy = get_recommendations(user_id, limit)
        output = []
        for r in recs:
            output.append({
                "product_id": r.product_id,
                "product_name": r.product_name,
                "category": r.category,
                "score": r.score,
                "reason": r.reason,
                "breakdown": {
                    "cf": r.breakdown.cf,
                    "content": r.breakdown.content,
                    "popularity": r.breakdown.popularity
                } if r.breakdown else None
            })
        return output
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found or error: {str(e)}")

@app.get("/users/{user_id}/activity", response_model=List[ActivityResponse])
def get_user_activity(user_id: int):
    try:
        return get_activity(user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail="User not found")
