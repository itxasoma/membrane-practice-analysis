# Makefile for the membrane practice analysis

PYTHON  = python3

all: $(TARGET)

# Check if the equilibration simulation is a good thermal equilibrium
equi:
	mkdir -p ../figures
	$(PYTHON) equi.py

# Correaltions 
corr:
	mkdir -p ../figures
	$(PYTHON) corr.py