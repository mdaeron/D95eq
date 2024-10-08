import pdoc

pdoc.render.configure(template_directory = 'pdoc_templates', search = False)

with open('../docs/index.html', 'w') as fid:
	fid.write(pdoc.pdoc('D95thermo'))

