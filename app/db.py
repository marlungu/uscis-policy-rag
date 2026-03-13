from sqlalchemy import create_engine, text, event
from pgvector.psycopg import register_vector


from app.config import settings


engine = create_engine(settings.postgres_url, future=True)

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)

def check_database_connection() -> None:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        version = result.scalar()
        print("Connected to PostgreSQL successfully.")
        print(version)


def check_pgvector_extension() -> None:
    with engine.connect() as connection:
        result = connection.execute(
            text(
                """
                SELECT extname
                FROM pg_extension
                WHERE extname = 'vector';
                """
            )
        )
        extension = result.scalar()

        if extension == "vector":
            print("pgvector extension is enabled.")
        else:
            raise RuntimeError("pgvector extension is NOT enabled in this database.")