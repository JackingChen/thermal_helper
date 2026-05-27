
## Deployment Guide

This section describes how to deploy the app on a remote server or VM using the published Docker image.

### Prerequisites

- Docker installed on the target machine
- A valid `.env` file with your Azure OpenAI credentials (see below)

### 1. Log in to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted. This is required to pull the image.

### 2. Create a `.env` file

The `.env` file holds secrets that **must not be committed to version control**. Create it manually on the target machine:

```bash
cat > .env << 'EOF'
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
EOF
```

Keep the `.env` file private — it is already listed in `.gitignore`.

> **Why not commit `.env`?**  
> Azure OpenAI keys grant billing access. Pushing them to a public (or even private) repository risks credential exposure via git history. Always inject secrets at runtime through environment variables or a secrets manager.

### 3. Pull and run the image

Create a `docker-compose.yml` on the target machine (no source code needed):

```yaml
services:
  thermal-helper:
    image: jackingchen120955/thermal_helper:latest
    container_name: thermal-helper
    ports:
      - "8601:8501"
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

### 4. Updating to a newer image

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
