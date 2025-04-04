include .env
export OPENAI_API_KEY := $(OPENAI_API_KEY)
export OPENAI_AGENTS_DISABLE_TRACING := $(OPENAI_AGENTS_DISABLE_TRACING)
export QIITA_API_ACCESS_TOKEN := $(QIITA_API_ACCESS_TOKEN)

run:
	uv run gradio src/qiita-editor-with-agent/main.py

lint:
	uv run isort src
	uv run black -l 79 src
	uv run flake8 src
