# PostgreSQL Setup Guide for MCP Server

This guide provides step-by-step instructions to set up your PostgreSQL database for the MCP server, including creating a new database, user, and enabling the `pgvector` extension.

**Prerequisites:**

- You have PostgreSQL installed and running on your system.
- You have access to a PostgreSQL superuser account (e.g., `postgres`) to create new databases and users.

---

### Step 1: Connect to PostgreSQL as a Superuser

Open your terminal and connect to your PostgreSQL server, typically as the `postgres` user:

```bash
psql -h 10.1.1.102 -p 5433 -U webui_user
```

You will be prompted for the `postgres` user's password.

---

### Step 2: Create a New Database

Once connected to `psql`, create a new database for your MCP server. Replace `your_db_name` with your desired database name.

```sql
CREATE DATABASE your_db_name;
```

---

### Step 3: Create a New User

Next, create a new user that the MCP server will use to connect to the database. Replace `your_db_user` and `your_db_password` with your desired username and a strong password.

```sql
CREATE USER your_db_user WITH PASSWORD 'your_db_password';
```

---

### Step 4: Grant Privileges to the New User

Grant all necessary privileges on your new database to the new user:

```sql
GRANT ALL PRIVILEGES ON DATABASE your_db_name TO your_db_user;
```

---

### Step 5: Connect to Your New Database and Install `pgvector`

Now, disconnect from the `postgres` database (type `\q` and press Enter) and reconnect to your newly created database as the new user:

```bash
psql -U your_db_user -d your_db_name
```

Once connected to `your_db_name`, enable the `pgvector` extension:

```sql
CREATE EXTENSION vector;
```

---

### Step 6: Create Database Schema

Exit `psql` again (type `\q` and press Enter). Now, you need to execute the SQL commands from the `crawled_pages.sql` file against your new database. This will create the tables and functions required by the MCP server.

Make sure you are in the root directory of your `mcp-crawl4ai-rag` project (where `crawled_pages.sql` is located).

```bash
psql -h localhost -p 5432 -U your_db_user -d your_db_name -f crawled_pages.sql
```

- Replace `localhost` and `5432` if your PostgreSQL server is on a different host or port.
- Replace `your_db_user` and `your_db_name` with the credentials you created.

---

### Step 7: Update Your `.env` File

Finally, update your `.env` file (or create one from `.env.example`) with the database connection details you just set up:

```
# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

Replace the placeholder values with your actual database host, port, name, user, and password.

---

After completing these steps, your PostgreSQL database should be ready for the MCP server.
