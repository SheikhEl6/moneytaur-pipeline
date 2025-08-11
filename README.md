# MoneyTaur Pipeline

Template repository for MoneyTaur data pipeline with structured folders for ingestion, ETL, enrichment, API, and notebooks.

## Overview

This repository provides a structured template for building data pipelines for MoneyTaur, a financial data processing platform. The pipeline is organized into modular components for data ingestion, transformation, enrichment, and serving.

## Project Structure

```
moneytaur-pipeline/
├── api/                    # API endpoints and web services
│   └── README.md          # API documentation
├── enrich/                # Data enrichment modules
│   └── README.md          # Enrichment process documentation
├── etl/                   # Extract, Transform, Load processes
│   └── README.md          # ETL pipeline documentation
├── ingest/                # Data ingestion modules
│   └── README.md          # Ingestion process documentation
├── notebooks/             # Jupyter notebooks for analysis
│   └── README.md          # Notebook documentation
├── LICENSE                # MIT License
├── README.md              # This file
└── Makefile              # Build and deployment automation
```

## Environment Setup

### Prerequisites

- Python 3.8+
- pip or conda for package management
- OpenAI API access (for AI-powered features)

### Environment Variables

This project requires an OpenAI API key for AI-powered data processing features.

#### Setting OPENAI_API_KEY

**Option 1: Environment Variable (Recommended)**

```bash
# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

**Option 2: .env File**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key-here
```

**Note:** Never commit your API key to version control. The `.env` file should be added to `.gitignore`.

#### Getting Your OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and securely store the key

### Installation

```bash
# Clone the repository
git clone https://github.com/SheikhEl6/moneytaur-pipeline.git
cd moneytaur-pipeline

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies (when requirements.txt is available)
pip install -r requirements.txt
```

## Quick Start

Use the provided Makefile for common tasks:

```bash
# Set up development environment
make setup

# Run tests
make test

# Start development services
make dev

# Build for production
make build

# Deploy
make deploy

# Clean build artifacts
make clean
```

## Development Workflow

1. **Data Ingestion**: Use modules in `ingest/` to collect raw data
2. **ETL Processing**: Transform and clean data using `etl/` pipelines
3. **Data Enrichment**: Enhance data quality and add insights via `enrich/`
4. **API Services**: Expose processed data through `api/` endpoints
5. **Analysis**: Perform exploratory data analysis in `notebooks/`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Security

- Never commit API keys or sensitive credentials
- Use environment variables or secure secret management
- Regularly rotate API keys
- Follow the principle of least privilege for API permissions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please:
1. Check existing [Issues](https://github.com/SheikhEl6/moneytaur-pipeline/issues)
2. Create a new issue with detailed description
3. Include relevant logs and error messages

## Roadmap

- [ ] Add comprehensive test suite
- [ ] Implement CI/CD pipeline
- [ ] Add Docker support
- [ ] Create detailed API documentation
- [ ] Add monitoring and logging
- [ ] Implement data validation frameworks
