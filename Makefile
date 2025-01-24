##
## EPITECH PROJECT, 2024
## Python Makefile
## File description:
## Makefile
##

BINARY_GENERATOR=my_torch_generator
SOURCE_GENERATOR=src/generator/generator.py

BINARY_ANALYZER=my_torch_analyzer
SOURCE_ANALYZER=src/analyzer/analyzer.py

.PHONY: all build run clean install

all: build

build: $(BINARY_GENERATOR) $(BINARY_ANALYZER)

$(BINARY_GENERATOR): $(SOURCE_GENERATOR)
	@echo "Building binary: $(BINARY_GENERATOR)"
	@echo '#!/usr/bin/env python3' > $(BINARY_GENERATOR)
	@cat $(SOURCE_GENERATOR) >> $(BINARY_GENERATOR)
	@chmod +x $(BINARY_GENERATOR)
	@echo "Binary $(BINARY_GENERATOR) created."

$(BINARY_ANALYZER): $(SOURCE_ANALYZER)
	@echo "Building binary: $(BINARY_ANALYZER)"
	@echo '#!/usr/bin/env python3' > $(BINARY_ANALYZER)
	@cat $(SOURCE_ANALYZER) >> $(BINARY_ANALYZER)
	@chmod +x $(BINARY_ANALYZER)
	@echo "Binary $(BINARY_ANALYZER) created."

run: build
	@echo "Running $(BINARY_GENERATOR)..."
	@./$(BINARY_GENERATOR) configs/basic_network.conf generated_networks

clean:
	@echo "Cleaning up..."
	@rm -f $(BINARY_GENERATOR)
	@rm -f $(BINARY_ANALYZER)
	@find . -name "*.pyc" -exec rm -f {} +
	@find . -name "__pycache__" -exec rm -rf {} +

install:
	@pip install -r requirements.txt

venv:
	@python3 -m venv venv
	@venv/bin/pip install --upgrade pip
