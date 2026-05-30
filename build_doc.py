import pdoc

pdoc.render.configure(search = False)

with open('../docs/index.html', 'w') as fid:
	fid.write(pdoc.pdoc('D95eq'))
