build:
	poetry install -E xai -E lightgbm

test:
	poetry run pytest