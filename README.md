
## Project Overview

USCIS Policy RAG is a Retrieval-Augmented Generation system that enables
semantic search and question answering over USCIS policy manuals using:

- Amazon Bedrock foundation models
- pgvector for semantic retrieval
- FastAPI for API serving
- LangChain for orchestration

## Local Development Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/uscis-policy-rag.git
cd uscis-policy-rag
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Run tests:
```bash
pytest
```

## Database Setup with Docker

Start PostgreSQL + pgvector:

```bash
docker compose up -d
```
Verify the container is running:
```bash
docker ps
```
Connect to PostgreSQL inside the container:
```bash
docker exec -it uscis-rag-postgres psql -U postgres -d uscis_rag
```
Enable the pgvector extension:
```SQL
CREATE EXTENSION IF NOT EXISTS vector;
```
Verify the extension is installed:
```SQL
\dx
```
Exit PostgreSQL:
```SQL
\q
```
 
## Verify Database Connection

 Run the database check script:
 ```bash
 python -m scripts.check_db
 ```

## Test Document Ingestion

Upload at least one USCIS PDF to the configured S3 bucket and prefix, then run:

```bash
python -m scripts.test_ingestion
```
