FROM python:3.13-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src/inference-service
COPY ./ /src/inference-service

# Install the project from pyproject.toml (creates the start-server script)
# RUN pip install --no-cache-dir -e .
RUN uv sync --frozen --no-cache

# Debug: Check what was installed
# RUN uv run which start-server && cat $(uv run which start-server) | head -20

# Optional: but makes things clearer
EXPOSE 8000

CMD ["uv", "run", "start-server", "--host", "0.0.0.0", "--port", "8000", "--dev"]
# CMD "uv run start-server"
