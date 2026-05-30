default: metadata doc

all: coefs pytest examples metadata doc cli

coefs:
	cd calib_coefs; uv run calib_coefs.py

pytest:
	uv run pytest # -s

examples:
	cd examples; for f in *.py; do uv run "$f"; done

metadata:
	uv run build-metadata.py

doc: examples
	cd src; uv run ../build_doc.py

cli:
	uv run D95eq -v
	uv run D95eq -i examples/example_data.csv -k '-1(0.1)'

diff:
	git diff --stat -- ':!*.png' ':!*.html' ':!*.pdf' ':!*.csv'

publish:
	uv run flit publish
