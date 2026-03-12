from app.db import check_database_connection, check_pgvector_extension


if __name__ == "__main__":
    check_database_connection()
    check_pgvector_extension()
    print("Database setup looks good.")