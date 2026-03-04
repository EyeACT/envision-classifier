PROJECT := envision-classifier
PACKAGE := envision_classifier
MODULES := $(wildcard $(PACKAGE)/*.py)

# MAIN TASKS ##################################################################

.PHONY: all
all: format check test ## Run all tasks that determine CI status

# SYSTEM DEPENDENCIES #########################################################

.PHONY: doctor
doctor: ## Confirm system dependencies are available
	poetry --version
	python --version

# PROJECT DEPENDENCIES ########################################################

VIRTUAL_ENV ?= .venv

.PHONY: install
install: .cache ## Install project dependencies
	@ poetry config virtualenvs.in-project true
	poetry install

.cache:
	@ mkdir -p .cache

# TEST ########################################################################

.PHONY: test
test: install ## Run unit tests
	poetry run pytest $(PACKAGE) tests -rx -W ignore::DeprecationWarning

# CHECK #######################################################################

.PHONY: format
format: install ## Run formatters
	poetry run isort $(PACKAGE)
	poetry run black $(PACKAGE)

.PHONY: check
check: install format ## Run linters and static analysis
ifdef CI
	git diff --exit-code
endif

# DOCUMENTATION ###############################################################

.PHONY: docs
docs: install ## Build and serve documentation
	@ cd docs && ln -sf ../README.md index.md 2>/dev/null || true
	@ cd docs/about && ln -sf ../../CHANGELOG.md changelog.md 2>/dev/null || true
	@ cd docs/about && ln -sf ../../CONTRIBUTING.md contributing.md 2>/dev/null || true
	@ cd docs/about && ln -sf ../../LICENSE.md license.md 2>/dev/null || true
	poetry run mkdocs serve

# BUILD #######################################################################

.PHONY: dist
dist: install ## Build distribution packages
	rm -f dist/*
	poetry build

.PHONY: upload
upload: dist ## Upload the current version to PyPI
	git diff --name-only --exit-code
	poetry publish

# CLEANUP #####################################################################

.PHONY: clean
clean: ## Delete all generated and temporary files
	find $(PACKAGE) -name '__pycache__' -delete
	rm -rf *.egg-info .cache .venv .pytest .coverage htmlcov dist build site

# HELP ########################################################################

.PHONY: help
help: ## Show available commands
	@ grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
