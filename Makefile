run:
	uv run gradio src/qiita-editor-with-agent/main.py

lint:
	uv run isort src
	uv run black -l 79 src
	uv run flake8 src
