.PHONY: setup data train evaluate api dashboard test

setup:
	pip install -e .[dev,llm]

data:
	python scripts/generate_data.py

train:
	python src/recsys/pipelines/train.py

batch:
	python src/recsys/pipelines/batch.py

evaluate:
	python scripts/run_evaluation.py

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v
