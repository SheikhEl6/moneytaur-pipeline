# MoneyTaur Pipeline Makefile
# Build and deployment automation for the MoneyTaur data pipeline

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
VENV_DIR := venv
REQUIREMENTS := requirements.txt

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Check if virtual environment exists
check_venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Virtual environment not found. Creating...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	else \
		echo "$(GREEN)Virtual environment found.$(NC)"; \
	fi

# Environment validation
check_env:
	@echo "$(BLUE)Checking environment variables...$(NC)"
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "$(RED)Error: OPENAI_API_KEY environment variable is not set!$(NC)"; \
		echo "$(YELLOW)Please set your OpenAI API key using:$(NC)"; \
		echo "  export OPENAI_API_KEY='your-api-key-here'"; \
		echo "$(YELLOW)Or create a .env file in the project root.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)OPENAI_API_KEY is set.$(NC)"; \
	fi

# Installation and setup
setup: check_venv ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip && \
		if [ -f $(REQUIREMENTS) ]; then \
			$(PIP) install -r $(REQUIREMENTS); \
		else \
			echo "$(YELLOW)requirements.txt not found. Installing basic packages...$(NC)"; \
			$(PIP) install openai pandas numpy jupyter matplotlib seaborn requests python-dotenv; \
		fi
	@echo "$(GREEN)Setup complete!$(NC)"

# Create requirements.txt
freeze: check_venv ## Generate requirements.txt from current environment
	@echo "$(BLUE)Generating requirements.txt...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PIP) freeze > $(REQUIREMENTS)
	@echo "$(GREEN)requirements.txt updated.$(NC)"

# Install dependencies
install: check_venv ## Install dependencies from requirements.txt
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PIP) install -r $(REQUIREMENTS)
	@echo "$(GREEN)Dependencies installed.$(NC)"

# Development server/services
dev: check_env ## Start development services
	@echo "$(BLUE)Starting development environment...$(NC)"
	@echo "$(YELLOW)Note: Implement specific dev services as needed$(NC)"
	# Add commands to start your development services here
	# Example: start API server, database, etc.

# Jupyter notebook server
notebook: check_venv ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	@. $(VENV_DIR)/bin/activate && jupyter notebook --notebook-dir=notebooks

# Data pipeline operations
ingest: check_env ## Run data ingestion pipeline
	@echo "$(BLUE)Running data ingestion pipeline...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTHON) -m ingest

etl: check_env ## Run ETL pipeline
	@echo "$(BLUE)Running ETL pipeline...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTHON) -m etl

enrich: check_env ## Run data enrichment pipeline
	@echo "$(BLUE)Running data enrichment pipeline...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTHON) -m enrich

api: check_env ## Start API server
	@echo "$(BLUE)Starting API server...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTHON) -m api

# Run full pipeline
pipeline: check_env ingest etl enrich ## Run complete data pipeline
	@echo "$(GREEN)Pipeline execution completed!$(NC)"

# Testing
test: check_venv ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		if [ -d "tests" ]; then \
			$(PYTHON) -m pytest tests/ -v; \
		else \
			echo "$(YELLOW)No tests directory found. Creating test structure...$(NC)"; \
			mkdir -p tests; \
			touch tests/__init__.py tests/test_example.py; \
			echo "$(GREEN)Test structure created. Add your tests and run 'make test' again.$(NC)"; \
		fi

test-coverage: check_venv ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install pytest-cov && \
		$(PYTHON) -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# Code quality
lint: check_venv ## Run code linting
	@echo "$(BLUE)Running code linting...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install flake8 black isort && \
		flake8 . --exclude=$(VENV_DIR),__pycache__ && \
		black --check . --exclude=$(VENV_DIR) && \
		isort --check-only .

format: check_venv ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install black isort && \
		black . --exclude=$(VENV_DIR) && \
		isort .
	@echo "$(GREEN)Code formatted!$(NC)"

# Build and deployment
build: test ## Build project for production
	@echo "$(BLUE)Building project...$(NC)"
	@echo "$(YELLOW)Implement build steps as needed$(NC)"
	# Add build commands here (e.g., Docker build, package creation)

deploy: build ## Deploy to production
	@echo "$(BLUE)Deploying to production...$(NC)"
	@echo "$(YELLOW)Implement deployment steps as needed$(NC)"
	# Add deployment commands here

# Docker operations (if using Docker)
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t moneytaur-pipeline .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -e OPENAI_API_KEY="$$OPENAI_API_KEY" -p 8000:8000 moneytaur-pipeline

# Cleanup
clean: ## Clean up build artifacts and temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	rm -rf $(VENV_DIR)
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

# Documentation
docs: check_venv ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install sphinx sphinx-rtd-theme && \
		if [ ! -d "docs" ]; then \
			mkdir -p docs && \
			sphinx-quickstart docs --quiet --project="MoneyTaur Pipeline" --author="MoneyTaur Team"; \
		fi && \
		cd docs && make html
	@echo "$(GREEN)Documentation generated in docs/_build/html/$(NC)"

# Security
security-check: check_venv ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install safety bandit && \
		safety check && \
		bandit -r . -x $(VENV_DIR) -f json

# Database operations (customize as needed)
db-migrate: check_env ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@echo "$(YELLOW)Implement database migration commands as needed$(NC)"

db-seed: check_env ## Seed database with sample data
	@echo "$(BLUE)Seeding database...$(NC)"
	@echo "$(YELLOW)Implement database seeding commands as needed$(NC)"

# Status and info
status: ## Show project status
	@echo "$(BLUE)MoneyTaur Pipeline Status$(NC)"
	@echo "==============================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Virtual environment: $$(if [ -d $(VENV_DIR) ]; then echo '$(GREEN)Present$(NC)'; else echo '$(RED)Missing$(NC)'; fi)"
	@echo "Requirements file: $$(if [ -f $(REQUIREMENTS) ]; then echo '$(GREEN)Present$(NC)'; else echo '$(YELLOW)Missing$(NC)'; fi)"
	@echo "OPENAI_API_KEY: $$(if [ -z "$$OPENAI_API_KEY" ]; then echo '$(RED)Not set$(NC)'; else echo '$(GREEN)Set$(NC)'; fi)"
	@echo "Folders:"
	@ls -la | grep ^d | awk '{print "  " $$9}'

help: ## Show this help message
	@echo "$(BLUE)MoneyTaur Pipeline - Available Commands$(NC)"
	@echo "========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Environment Setup:$(NC)"
	@echo "  Make sure to set OPENAI_API_KEY environment variable"
	@echo "  Run 'make setup' to initialize the development environment"

# Phony targets
.PHONY: help setup install dev test build deploy clean clean-all check_venv check_env
.PHONY: ingest etl enrich api pipeline notebook lint format docs security-check
.PHONY: docker-build docker-run db-migrate db-seed status freeze test-coverage
