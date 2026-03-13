from sqlalchemy import text
from app.db import engine


def main():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]

        print("Connected to PostgreSQL successfully.")
        print(version)

        result = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        )

        if result.fetchone():
            print("pgvector extension is enabled.")
        else:
            print("pgvector extension is NOT enabled.")

        result = conn.execute(text("SELECT COUNT(*) FROM document_chunks;"))
        count = result.fetchone()[0]

        print(f"document_chunks row count: {count}")

    print("Database setup looks good.")


if __name__ == "__main__":
    main()