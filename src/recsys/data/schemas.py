import pandera as pa

UserSchema = pa.DataFrameSchema({
    "user_id": pa.Column(int, pa.Check.ge(1)),
    "name": pa.Column(str),
    "age_group": pa.Column(str),
    "country": pa.Column(str),
    "signup_date": pa.Column(str, coerce=True)
})

ProductSchema = pa.DataFrameSchema({
    "product_id": pa.Column(int, pa.Check.ge(1)),
    "name": pa.Column(str),
    "category": pa.Column(str),
    "subcategory": pa.Column(str),
    "brand": pa.Column(str),
    "price": pa.Column(float, pa.Check.ge(0.0)),
    "description": pa.Column(str),
    "tags": pa.Column(str),
    "created_at": pa.Column(str, coerce=True)
})

InteractionSchema = pa.DataFrameSchema({
    "interaction_id": pa.Column(int, pa.Check.ge(1), coerce=True),
    "user_id": pa.Column(int, pa.Check.ge(1), coerce=True),
    "product_id": pa.Column(int, pa.Check.ge(1), coerce=True),
    "event_type": pa.Column(str, pa.Check.isin(["view", "click", "like", "purchase", "rating"]), coerce=True),
    "rating": pa.Column(float, pa.Check.in_range(1.0, 5.0), nullable=True, coerce=True),
    "timestamp": pa.Column("datetime64[ns]", coerce=True)
})
