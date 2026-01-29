default: version doc

all: coefs test qmc examples version doc

version:
	uv run python update-version.py

examples:
	cd examples; uv run python example.py; uv run python large_example.py

qmc:
	cd examples; uv run python example_qmc.py

coefs:
	cd calib_coefs; uv run python calib_coefs.py

test:
	cd test; uv run python test.py

publish:
	uv run flit publish

doc:
	cd src; uv run python ../build_doc.py
