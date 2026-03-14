SRC = source/slides.md
TEX = source/mp3_presentation.tex
PDF = build/mp3_presentation.pdf

.PHONY: pdf clean

pdf: $(PDF)

$(TEX): $(SRC) md_to_tex.py
	python3 md_to_tex.py $(SRC) $(TEX)

$(PDF): $(TEX)
	mkdir -p build/fonts
	env VARTEXFONTS=$(CURDIR)/build/fonts pdflatex -output-directory=build $(TEX)
	env VARTEXFONTS=$(CURDIR)/build/fonts pdflatex -output-directory=build $(TEX)

clean:
	rm -f build/*.aux build/*.log build/*.nav build/*.snm build/*.toc build/*.out
