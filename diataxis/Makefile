# Build the diataxis documentation site.
#
# Hugo renders markdown to HTML. Marimo notebooks are preprocessed into
# self-contained WASM bundles before Hugo runs — Hugo does not know how to
# do this on its own, which is the only reason this Makefile exists.
#
# Targets:
#   make build     — export exercises, then run hugo
#   make serve     — export exercises, then run hugo server (live reload)
#   make exercises — export exercises only (leaves public/ alone)
#   make clean     — remove public/ and resources/

EXERCISE_SOURCES := $(wildcard exercises/*.py)
EXERCISE_BUNDLES := $(patsubst exercises/%.py,static/exercises/%/index.html,$(EXERCISE_SOURCES))

.PHONY: build serve exercises clean

build: exercises
	hugo --cleanDestinationDir

serve: exercises
	hugo server

exercises: $(EXERCISE_BUNDLES)

clean:
	rm -rf public resources

static/exercises/%/index.html: exercises/%.py
	@mkdir -p $(@D)
	uv run marimo export html-wasm $< -o $(@D) --mode run -f
