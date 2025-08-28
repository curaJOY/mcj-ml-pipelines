# Prodigy — Local Docker Setup

Run the Prodigy labeling UI locally with custom recipe and dataset.

## Repo layout

├─ data/                     # Cleaned text dataset (tweets) used as input

│  └─ twitter_parsed_clean.txt

├─ recipes_pkg/              # Your Python package with Prodigy recipes

│  ├─ bullying_workflow.py

│  └─ pyproject.toml

├─ whl/                      # Wheel files to install inside the image

│  ├─ prodigy-.whl          # Prodigy wheel

│  └─ en_core_web_sm-.whl   # spaCy model wheel

├─ Dockerfile

├─ docker-compose.yml

└─ README.md

## What each file/folder does

- `data/`  
  Holds the text data you will annotate. `docker-compose.yml` mounts this folder into the container so Prodigy can read `twitter_parsed_clean.txt`.

- `recipes_pkg/`  
  Python package that defines your recipe(s). `bullying_workflow.py` implements the `bullying-workflow` recipe referenced by the compose command. `pyproject.toml` declares package metadata and build requirements.

- `whl/`  
  Local wheels installed during the Docker build, including Prodigy and `en_core_web_sm`. This keeps builds deterministic and avoids network pulls.

- `Dockerfile`  
  Builds a runnable image with Prodigy, spaCy model, and your `recipes_pkg`.

- `docker-compose.yml`  
  Defines the service, platform, ports, env vars, volumes, and the Prodigy run command. It exposes Prodigy on port 8080 and mounts `./data` to `/app/data` in the container. It runs:
  `prodigy bullying-workflow cyberbullying_dataset /app/data/twitter_parsed_clean.txt -l txt -m en_core_web_sm`

which starts your `bullying-workflow` recipe on the given dataset and file.

## Prerequisites

- Docker and Docker Compose installed
- A valid Prodigy license key (contact me or Caitlyn if needed)

## Configuration

Create a `.env` file in the repo root:

```bash
PRODIGY_KEY=(Prodigy key)
PRODIGY_AUTH_SECRET=$(openssl rand -hex 32) # or any other long random string
```

## Build and run

1) Build the image

```bash
docker compose build
```

2) Start the service

```bash
docker compose up
```

3) Open Prodgy

```bash
open http://localhost:8080
```

4) Stop the service

```bash
docker compose down
```

5) If editing recipes, restart the service to update

```bash
docker compose restart prodigy
```

## File Rundown

### Dockerfile

Builds a self-contained image that can run Prodigy with your custom recipe.
- Starts from python:3.10-bullseye, installs build tools, then installs your pinned pip tooling and numpy.
- Installs Prodigy from a local wheel: prodigy-1.11.7-cp310-cp310-linux_x86_64.whl.
- Installs spaCy 3.4.4 and the en_core_web_sm model from a local wheel, with sanity checks.
- Copies and installs your recipes_pkg so the bullying-workflow entry point is discoverable.
- Creates a non-root prodigy user, sets /app as the working directory, copies data/, exposes port 8080, and sets the container’s default command to run your recipe on the cleaned text file.

I am using an AMD64 Linux wheel for Prodigy and the spaCy model, so the container can run on Render’s Linux AMD64 environment without compiling native extensions during deploy. `docker-compose.yml` also pins platform: linux/amd64, which mirrors the target runtime.  ￼

### docker-compose.yml

Defines how to run the image locally.
- Sets platform: linux/amd64 to match the wheel architecture.
- Builds from the Dockerfile and passes the wheel name as a build arg.
- Publishes 8080:8080, sets required env vars (license key, auth secret, host, port), and mounts ./data into the container so the recipe can read your tweet file.
- Runs the recipe command: `prodigy bullying-workflow cyberbullying_dataset /app/data/twitter_parsed_clean.txt -l txt -m en_core_web_sm` which starts the UI on port 8080.

### bullying_workflow.py

Implements guided 4-step annotation flow.
-	CLI signature: bullying-workflow <dataset> <source> -l {txt|jsonl} -m <spacy_model>.
-	Loads a spaCy tokenizer once, streams examples from .txt or .jsonl, attaches token boundaries so spans_manual can highlight text correctly, and adds stable hashes.
-	Provides a compact HTML “wizard” for steps 1–3 (binary label, role, severity), then reveals spans_manual in step 4 for span labeling.
-	Validates answers server-side so incomplete submissions are rejected with clear messages.

### ersioning tip

Any time you change UI logic, labels, validation, or parameters:
- Bump the printed version string at the top of bullying_workflow.py (e.g., v0.0.30 → v0.0.31).
- Bump the version in recipes_pkg/pyproject.toml so Docker rebuilds install a new package version and Prodigy picks up the change cleanly.
