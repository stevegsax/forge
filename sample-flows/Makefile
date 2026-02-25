# Build Mermaid diagrams from .mmd source files.
#
# Requires mermaid-cli:
#   npm install -g @mermaid-js/mermaid-cli

MMDC := mmdc
MMD_FILES := $(shell find . -name '*.mmd')
SVG_FILES := $(MMD_FILES:.mmd=.svg)
PNG_FILES := $(MMD_FILES:.mmd=.png)

.PHONY: all clean svg png install-hint

all: svg

svg: $(SVG_FILES)

png: $(PNG_FILES)

%.svg: %.mmd
	$(MMDC) -i $< -o $@

%.png: %.mmd
	$(MMDC) -i $< -o $@

clean:
	rm -f $(SVG_FILES) $(PNG_FILES)

install-hint:
	@echo "Install mermaid-cli:  npm install -g @mermaid-js/mermaid-cli"
