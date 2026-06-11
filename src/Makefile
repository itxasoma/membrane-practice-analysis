# Makefile for the membrane practice analysis

PYTHON  = python3

all: $(TARGET)

# Check if the equilibration simulation is a good thermal equilibrium (if NPT is true)
equi:
	mkdir -p ../figures
	$(PYTHON) equi.py

# Check if the production simulation is a good thermal equilibrium (if NVT is true)
prod:
	mkdir -p ../figures
	$(PYTHON) prod.py

# Correaltions 
corr:
	mkdir -p ../figures
	$(PYTHON) corr.py

corr-plots:
	mkdir -p ../figures
	$(PYTHON) corr-plots.py

# Other interesting functions (like MSD):
stats:
	mkdir -p ../figures
	$(PYTHON) stats.py