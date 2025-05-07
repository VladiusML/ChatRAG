import os
import time

import psycopg2


def wait_for_db():
    while True:
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            conn.close()
            print("Database connection successful!")
            break
        except Exception as e:
            print(f"Waiting for database connection... {e}")
            time.sleep(5)


if __name__ == "__main__":
    wait_for_db()
