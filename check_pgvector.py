

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_and_install_pgvector():
    """
    Connects to the PostgreSQL database, checks if the 'vector' extension is installed,
    and attempts to install it if it's not.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        conn.autocommit = True  # Set autocommit to True for CREATE EXTENSION
        cur = conn.cursor()

        # Check if the extension is installed
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        extension_exists = cur.fetchone()

        if extension_exists:
            print("✅ The 'pgvector' extension is already installed.")
        else:
            print("🟡 The 'pgvector' extension is not installed. Attempting to install it...")
            try:
                cur.execute("CREATE EXTENSION vector")
                print("✅ Successfully installed the 'pgvector' extension.")
            except Exception as e:
                print(f"❌ Failed to install the 'pgvector' extension.")
                print("Error:", e)
                print("\nPlease make sure you have superuser privileges or that the 'pgvector' extension is available for your PostgreSQL instance.")

        # Close the connection
        cur.close()
        conn.close()

    except psycopg2.OperationalError as e:
        print("❌ Could not connect to the PostgreSQL database.")
        print("Error:", e)
        print("Please check your database connection details in the .env file and ensure the database is running.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_and_install_pgvector()

