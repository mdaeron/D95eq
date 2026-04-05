default: version doc

all: coefs pytest examples version doc cli

version:
	uv run update-version.py

examples:
	cd examples; uv run example.py; uv run large_example.py

coefs:
	cd calib_coefs; uv run calib_coefs.py

pytest:
	uv run pytest # -s

publish:
	uv run flit publish

doc:
	cd code-examples; uv run *.py
	cd src; uv run ../build_doc.py

cli:
	uv run D95thermo -v
	uv run D95thermo -i examples/example_data.csv -k '-1(0.1)'
