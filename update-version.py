import tomllib

with open('pyproject.toml', 'rb') as fid:
	v = tomllib.load(fid)['project']['version']

with open('src/D95thermo/__version__.py', 'w') as fid:
	fid.write(f'__version__ = "{v}"')
