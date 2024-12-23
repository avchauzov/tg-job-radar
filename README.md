# Telegram Job Radar

An automated system for collecting and analyzing job postings from Telegram channels using Airflow.

## Project Structure
```
./
├── poetry.lock
├── _production
│   ├── airflow
│   │   ├── dags
│   │   │   └── main_dag.py
│   │   └── plugins
│   │       ├── production
│   │       │   └── email_notifications.py
│   │       ├── raw
│   │       │   └── data_collection.py
│   │       └── staging
│   │           └── data_cleaning.py
│   ├── config
│   │   ├── config_db.py
│   │   ├── config.json
│   │   └── config.py
│   ├── __init__.py
│   └── utils
│       ├── common.py
│       ├── email.py
│       ├── exceptions.py
│       ├── llm.py
│       ├── prompts.py
│       ├── sql.py
│       ├── text.py
│       └── tg.py
├── pyproject.toml
└── README.md
```

## Features
- Automated data collection from Telegram channels
- Text processing and data cleaning
- LLM-based analysis and classification
- SQL database integration
- Email notifications system
- Comprehensive test coverage

## Components
- **Airflow DAGs**: Orchestration of data pipeline (`main_dag.py`)
- **Airflow Plugins**:
  - `raw/`: Data collection from Telegram
  - `staging/`: Data cleaning and preprocessing
  - `production/`: Email notification system
- **Utils**:
  - `common.py`: Shared utility functions
  - `email.py`: Email handling
  - `llm.py`: LLM integration
  - `sql.py`: Database operations
  - `text.py`: Text processing
  - `tg.py`: Telegram API interactions

## Setup
1. Install dependencies using Poetry:
```bash
poetry install
```

2. Configure the application:
- Copy `.env.example` to `.env`

- Update configuration with your credentials:
  - Telegram API credentials
  - Database connection details
  - Email settings
  - LLM API keys
  - Other relevant settings
- Copy `production/config/config.json.example` to `production/config/config.json`
- Update `production/config/config.json` with your Telegram channel names and other settings

3. Start Airflow:
```bash
airflow standalone
```



## Development
- Python 3.12+
- Poetry for dependency management
- Follow PEP 8 style guide

## Configuration
Key configuration files:
- `config/config.py`: Base configuration setup
- `config/config_db.py`: Database configuration
- `config/config.json`: Runtime configuration (not tracked in git)
