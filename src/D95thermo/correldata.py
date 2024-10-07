"""
Read/write correlated vectors from/to a csv file
"""


__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2024 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
__date__      = '2024-10-07'
__version__   = '1.0.0'


import numpy as _np
import uncertainties as _uc


class correl_array(_np.ndarray):

    def __new__(cls, a):
        obj = _np.asarray(a).view(cls)
        return obj
    
    n = property(fget = _np.vectorize(lambda x : x.n))
    s = property(fget = _np.vectorize(lambda x : x.s))


def is_symmetric_positive_semidefinite(x):
	return _np.all(_np.linalg.eigvals(x) >= 0) and _np.all(x - x.T == 0)

def smart_type(x):
	'''
	Tries to convert string `x` to a float if it includes a decimal point, or
	to an integer if it does not. If both attempts fail, return the original
	string unchanged.
	'''
	try:
		y = float(x)
	except ValueError:
		return x
	if y % 1 == 0 and '.' not in x:
		return int(y)
	return y


def read_data(data, sep = ',', validate_covar = True):

	data = [[smart_type(e.strip()) for e in l.split(sep)] for l in data.split('\n')]
	N = len(data) - 1
	
	values, se, correl, covar = {}, {}, {}, {}
	j = 0
	while j < len(data[0]):
		field = data[0][j]
		if not (
			field.startswith('SE_')
			or field.startswith('correl_')
			or field.startswith('covar_')
			or field == 'SE'
			or field == 'correl'
			or field == 'covar'
			or len(field) == 0
		):
			values[field] = _np.array([l[j] for l in data[1:]])
			j += 1
			oldfield = field
		elif field.startswith('SE_'):
			se[field[3:]] = _np.array([l[j] for l in data[1:]])
			j += 1
		elif field == 'SE':
			se[oldfield] = _np.array([l[j] for l in data[1:]])
			j += 1
		elif field == 'SE':
			se[oldfield] = _np.array([l[j] for l in data[1:]])
			j += N
		elif field.startswith('correl_'):
			correl[field[7:]] = _np.array([l[j:j+N] for l in data[1:]])
			j += N
		elif field == 'correl':
			correl[oldfield] = _np.array([l[j:j+N] for l in data[1:]])
			j += N
		elif field.startswith('covar_'):
			covar[field[6:]] = _np.array([l[j:j+N] for l in data[1:]])
			j += N
		elif field == 'covar':
			covar[oldfield] = _np.array([l[j:j+N] for l in data[1:]])
			j += N

	nakedvalues = {}
	for k in [_ for _ in values]:
		if (
			k not in se
			and k not in correl
			and k not in covar
		):
			nakedvalues[k] = values.pop(k)

	for x in values:
		if x in covar:
			if x in se:
				raise KeyError(f'Too much information: both SE and covar are specified for variable "{x}".')
			if x in correl:
				raise KeyError(f'Too much information: both correl and covar are specified for variable "{x}".')
		if x in correl:
			if x not in se:
				raise KeyError(f'Not enough information: correl is specified without SE for variable "{x}".')

	for x in correl:
		if x in values:
			covar[x] = _np.diag(se[x]) @ correl[x] @ _np.diag(se[x])
		else:
			for x1 in values:
				for x2 in values:
					if x == f'{x1}_{x2}':
						if x1 in se:
							se1 = se[x1]
						else:
							if x1 in covar:
								se1 = _np.diag(covar[x1])**0.5
							else:
								raise KeyError(f'Not enough information: correl_{x} is specified without SE for variable "{x1}".')
						if x2 in se:
							se2 = se[x2]
						else:
							if x2 in covar:
								se2 = _np.diag(covar[x2])**0.5
							else:
								raise KeyError(f'Not enough information: correl_{x} is specified without SE for variable "{x1}".')

						covar[x] = _np.diag(se1) @ correl[x] @ _np.diag(se2)

	for x in se:
		if x in values and x not in correl:
			covar[x] = _np.diag(se[x]**2)

	for k in [_ for _ in covar]:
		if k not in values:
			for j1 in values:
				for j2 in values:
					if k == f'{j1}_{j2}':
						covar[f'{j2}_{j1}'] = covar[f'{j1}_{j2}'].T
			
	X = _np.array([_ for k in values for _ in values[k]])
	CM = _np.zeros((X.size, X.size))
	for i, vi in enumerate(values):
		for j, vj in enumerate(values):
			if vi == vj:
				if vi in covar:
					CM[N*i:N*i+N,N*j:N*j+N] = covar[vi]
			else:
				if f'{vi}_{vj}' in covar:
					CM[N*i:N*i+N,N*j:N*j+N] = covar[f'{vi}_{vj}']
	
	if validate_covar and not is_symmetric_positive_semidefinite(CM):
		raise _np.linalg.LinAlgError('The complete covariance matrix is not symmetric positive-semidefinite.')
	
	corvalues = correl_array(_uc.correlated_values(X, CM))

	allvalues = nakedvalues
	
	for i, x in enumerate(values):
		allvalues[x] = corvalues[i*N:i*N+N]

	return allvalues


def read_data_from_file(filename, **kwargs):
	with open(filename) as fid:
		return read_data(fid.read(), **kwargs)

def data_string(
	data,
	sep = ',',
	float_fmt = 'zg',
	max_correl_precision = 6,
	fields = None,
	align = '>',
	atol = 1e-9,
	rtol = 1e-9,
):
	if fields is None:
		fields = [_ for _ in data]
	cols, ufields = [], []
	for f in fields:
		if isinstance(data[f], correl_array):
			ufields.append(f)
			N = data[f].size
			cols.append([f] + [f'{_.n:{float_fmt}}' for _ in data[f]])
			cols.append([f'SE_{f}'] + [f'{_.s:{float_fmt}}' for _ in data[f]])
			CM = _uc.correlation_matrix(data[f])
			if not _np.allclose(CM, _np.eye(N), atol = atol, rtol = rtol):
				for i in range(N):
					cols.append(['' if i else f'correl_{f}'] + [f'{CM[i,j] if abs(CM[i,j]) > atol else 0:z{max_correl_precision}f}'.rstrip('0') for j in range(N)])
					
		else:
			cols.append([f] + [str(_) for _ in data[f]])
	
	for i in range(len(ufields)):
		for j in range(i):
			CM = _uc.correlation_matrix((*data[ufields[i]], *data[ufields[j]]))[:N,N:]
			if not _np.allclose(CM, _np.eye(N), atol = atol, rtol = rtol):
				for k in range(N):
					cols.append(['' if k else f'correl_{ufields[i]}_{ufields[j]}'] + [f'{CM[k,l] if abs(CM[k,l]) > atol else 0:z{max_correl_precision}f}'.rstrip('0') for l in range(N)])
	
	lines = list(map(list, zip(*cols)))

	if align:
		lengths = [max([len(e) for e in l]) for l in cols]
		for l in lines:
			for k,ln in enumerate(lengths):
				l[k] = f'{l[k]:{align}{ln}s}'
		return '\n'.join([(sep+' ').join(l) for l in lines])

	return '\n'.join([sep.join(l) for l in lines])
	


def save_data_to_file(data, filename, **kwargs):
	with open(filename, 'w') as fid:
		return fid.write(data_string(data, **kwargs))


if __name__ == '__main__':

	data = read_data_from_file('data.csv')
	
	print(data)
	
	print()

	print(data_string(data))
	save_data_to_file(data, 'saved_data.csv')