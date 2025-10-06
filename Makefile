# Makefile for trajectory_nn project

# --- CONFIG PATHS ---
BRA_DATA  = configs/brachi/data_brachi.yaml
BRA_MODEL = configs/brachi/model_brachi.yaml
BRA_RUN   = configs/brachi/run_brachi.yaml

COL_DATA  = configs/collision/data_collision.yaml
COL_MODEL = configs/collision/model_collision.yaml
COL_RUN   = configs/collision/run_collision.yaml
COL_DATA_HEADING ?= configs/collision_heading/data_collision_heading.yaml
COL_MODEL_HEADING ?= configs/collision_heading/model_collision_heading.yaml

AIR_DATA  = configs/airspace/data_airspace.yaml
AIR_MODEL = configs/airspace/model_airspace.yaml
AIR_RUN   = configs/airspace/run_airspace.yaml


# --- TRAINING ---
train-brachi:
	python -m src.train \
		--data-config $(BRA_DATA) \
		--model-config $(BRA_MODEL) \
		--run-config $(BRA_RUN)

train-collision:
	python -m src.train \
		--data-config $(COL_DATA) \
		--model-config $(COL_MODEL) \
		--run-config $(COL_RUN)

# --- PLOTTING ---
# Usage: make plot-brachi <run-folder-name> [NUM=...]    # NUM overrides --num-samples
plot-brachi:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make plot-brachi <run-folder> [NUM=...]"; \
		exit 1; \
	fi
	python -m tools.plot_brachi \
		--run-dir runs/$(word 2,$(MAKECMDGOALS)) \
		--data-config $(BRA_DATA) \
		--model-config $(BRA_MODEL) \
		--subset test \
		--num-samples $(if $(NUM),$(NUM),5)

.PHONY: plot-collision plot-collision-heading

# Usage: make plot-collision <run-folder-name> [NUM=...]
plot-collision:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make plot-collision <run-folder> [NUM=...]"; \
		exit 1; \
	fi
	python -m tools.plot_collision \
		--run-dir runs/$(word 2,$(MAKECMDGOALS)) \
		--data-config $(COL_DATA) \
		--model-config $(COL_MODEL) \
		--subset test \
		--num-samples $(if $(NUM),$(NUM),4)

plot-collision-heading:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make plot-collision-heading <run-folder> [NUM=...]"; \
		exit 1; \
	fi
	python -m tools.plot_collision_heading \
		--run-dir runs/$(word 2,$(MAKECMDGOALS)) \
		--data-config $(COL_DATA_HEADING) \
		--model-config $(COL_MODEL_HEADING) \
		--subset test \
		--num-samples $(if $(NUM),$(NUM),4)

.PHONY: plot-airspace
plot-airspace:
	@if [ -z "$(RUN)" ]; then \
		echo "Usage: make plot-airspace RUN=<run-folder> [NUM=...] [SEGMENT=1..6] [ALL_SEGMENTS=1]"; \
		exit 1; \
	fi
	python -m tools.plot_airspace \
		--run-dir runs/$(RUN) \
		--data-config configs/airspace/data_airspace.yaml \
		--model-config configs/airspace/model_airspace.yaml \
		--subset test \
		--num-samples $(if $(NUM),$(NUM),5) \
		$(if $(SEGMENT),--segment $(SEGMENT),) \
		$(if $(ALL_SEGMENTS),--all-segments,) \
		--keep-normalized


.PHONY: chain16-airspace
chain16-airspace:
	@if [ -z "$(RUN)" ] || [ -z "$(STARTX)" ] || [ -z "$(STARTY)" ]; then \
		echo "Usage: make chain16-airspace RUN=<run-folder> STARTX=<x0> STARTY=<y0> [STARTSEG=1] [SAVE=1]"; \
		exit 1; \
	fi
	python -m tools.plot_airspace_chain16 \
		--run-dir runs/$(RUN) \
		--data-config configs/airspace/data_airspace.yaml \
		--model-config configs/airspace/model_airspace.yaml \
		--subset test \
		--start-x $(STARTX) \
		--start-y $(STARTY) \
		--start-seg $(if $(STARTSEG),$(STARTSEG),1) \
		$(if $(SAVE),--save-csv,)



# Usage: make plot-metrics <run-folder-name> [SMOOTH=N] [LOGY=1]
# Examples:
#   make plot-metrics brachi_ffn_090725_s42_c80d69
#   make plot-metrics brachi_ffn_090725_s42_c80d69 SMOOTH=3 LOGY=1
plot-metrics:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make plot-metrics <run-folder> [SMOOTH=N] [LOGY=1]"; \
		exit 1; \
	fi
	python -m tools.plot_metrics \
		--run-dir runs/$(word 2,$(MAKECMDGOALS)) \
		$(if $(SMOOTH),--smooth $(SMOOTH),) \
		$(if $(LOGY),--logy,)

