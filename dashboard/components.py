import streamlit as st

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: #1a1a1a;
        }
        .badge {
            display: inline-block;
            padding: 2px 6px;
            font-size: 0.8em;
            font-weight: 600;
            border-radius: 4px;
            background-color: #ebeae6;
            color: #5a6b7d;
            border: 1px solid #d1d0cb;
        }
        .rec-row {
            padding: 8px 0;
            border-bottom: 1px solid #d1d0cb;
            display: flex;
            justify-content: space-between;
        }
        .rec-rank {
            font-weight: 600;
            margin-right: 12px;
            color: #5a6b7d;
        }
        .rec-name {
            font-weight: 600;
        }
        .rec-meta {
            font-size: 0.9em;
            color: #666;
        }
        .rec-score {
            font-family: monospace;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }
        th, td {
            text-align: left;
            padding: 8px 4px;
            border-bottom: 1px solid #d1d0cb;
        }
        th {
            font-weight: 600;
            color: #5a6b7d;
        }
    </style>
    """, unsafe_allow_html=True)

def render_activity_table(activities):
    html = "<table><tr><th>Event</th><th>Product</th><th>Date</th></tr>"
    for act in activities:
        event = str(act.get("event_type", "")).upper()
        name = act.get("name", "Unknown")
        date = str(act.get("timestamp", ""))[:16]
        html += f"<tr><td><span class='badge'>{event}</span></td><td>{name}</td><td>{date}</td></tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

def render_recommendation_row(rank, rec):
    cf = f"{rec.breakdown.cf:.2f}" if rec.breakdown else "0.00"
    cont = f"{rec.breakdown.content:.2f}" if rec.breakdown else "0.00"
    pop = f"{rec.breakdown.popularity:.2f}" if rec.breakdown else "0.00"
    
    html = f"""
    <div class="rec-row">
        <div>
            <span class="rec-rank">#{rank}</span>
            <span class="rec-name">{rec.product_name}</span> 
            <span class="rec-meta">({rec.category})</span>
            <div style="font-size: 0.85em; color: #5a6b7d; margin-top: 4px; font-style: italic;">
                {rec.reason}
            </div>
        </div>
        <div class="rec-score">
            Score: {rec.score:.4f} [CF:{cf} CT:{cont} P:{pop}]
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
