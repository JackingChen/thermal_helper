
## Deployment Guide

This section describes how to deploy the app on a remote server or VM using the published Docker image.

### Prerequisites

- Docker installed on the target machine
- A running Ollama instance with the required model pulled.

### 1. Install Ollama

Run the official one-line installer on the target machine:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installation, Ollama starts automatically as a systemd service. Verify it is running:

```bash
ollama list
```

Pull the model used by the app:

```bash
ollama pull qwen3:14b
```

By default Ollama listens on `127.0.0.1:11434`. To make it reachable from Docker containers (which use the host's `172.17.0.1` bridge address), set the bind address before starting:

```bash
# Add to /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11435"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Confirm it is reachable:

```bash
curl http://127.0.0.1:11435/api/tags
```

### 2. Log in to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted. This is required to pull the image.

### 3. Create a `.env` file

The `.env` file holds secrets that **must not be committed to version control**. Create it manually on the target machine:

```bash
cat > .env << 'EOF'
OLLAMA_ENDPOINT=http://172.17.0.1:11435
OLLAMA_MODEL=qwen3:14b
EOF
```

Keep the `.env` file private — it is already listed in `.gitignore`.

> **Why not commit `.env`?**  
> While Ollama doesn't use billing keys, it is best practice to keep environment-specific configuration separate.

### 4. Pull and run the image

Create a `docker-compose.yml` on the target machine (no source code needed):

```yaml
services:
  thermal-helper:
    image: jackingchen120955/thermal_helper:latest
    container_name: thermal-helper
    ports:
      - "8601:8501"
    volumes:
      - .:/app
    env_file:
      - .env
    environment:
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
    restart: unless-stopped
```

Then start the container:

```bash
docker compose pull          # fetch the latest image
docker compose up -d         # start in detached mode
```

Access the app at `http://<host-ip>:8601`.

### 5. Updating to a newer image

```bash
docker compose pull
docker compose up -d         # recreates the container with the new image
```

### Alternative: plain `docker run`

If you prefer not to use Compose:

```bash
docker pull jackingchen120955/thermal_helper:latest

docker run -d \
  --name thermal-helper \
  -p 8601:8501 \
  --env-file .env \
  -e STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll \
  --restart unless-stopped \
  jackingchen120955/thermal_helper:latest
```
