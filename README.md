## Quick start

```
# 1. Copy and fill in your credentials
cp .env.example .env
nano .env

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Initialize the database (creates all tables + indexes)
python -m database.init_db

# 4. Run the app
python main.py
```

To wipe and recreate the schema:

```
python -m database.init_db --drop
```
