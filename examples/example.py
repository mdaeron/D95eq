from correldata import read_data_from_file, save_data_to_file
import numpy as _np
import uncertainties as _uc
from warnings import filterwarnings
filterwarnings('ignore', category = FutureWarning)

from matplotlib import pyplot as _ppl
from D95thermo import *

E = Engine()

slope = _uc.ufloat(-1, 0.1)
p_cutoff = 0.05
eq_color = (0,.5,.2)
diseq_color = (1, 0, .4)

data = read_data_from_file('example_data.csv')
X = data['D47']
Y = data['D48']

D47eq, D48eq, pD47 = E.nearest_D47eq(X, Y, ignore_calib_uncertainties = False)
for _D in D47eq:
	for ignore_calib_uncertainties, color in (
		(True, 'r-'),
		(False, 'b-'),
	):
		Ti, p = E.Teq_pdf(_D, ignore_calib_uncertainties = ignore_calib_uncertainties)
		_ppl.plot(Ti, p, color)
_ppl.xlabel('T')
_ppl.ylabel('PDF')
# _ppl.show()
_ppl.savefig(f'pdfs-example.pdf')

# Teq, p = E.nearest_Teq(X, Y)
D47p, D48p = E.projected_D47eq(X, Y, slope)

fig = _ppl.figure(figsize = (6.5,4.5))
_ppl.title("“$Δ_{95}$ thermometry” ($47+48=95$)")

E.plot_D95_equilibrium()

conf_ellipse(X, Y, ec = 'k')
conf_ellipse(D47eq[pD47 >= p_cutoff], D48eq[pD47 >= p_cutoff], ec = eq_color, fc = (*eq_color, 0.2))
conf_ellipse(D47p[pD47 < p_cutoff], D48p[pD47 < p_cutoff], ec = diseq_color, fc = (*diseq_color, 0.2))

for x, y, xeq, yeq in zip(X[pD47 < p_cutoff], Y[pD47 < p_cutoff], D47p[pD47 < p_cutoff], D48p[pD47 < p_cutoff]):

	v = _np.array([
		xeq.n - x.n,
		yeq.n - y.n,
	])
	i, j = 0.15, 0.85
	kw = dict(
		color = diseq_color,
		lw = 0,
		width = 0.001,
		head_width = 0.005,
	)
	_ppl.arrow(
		x.n + i * v[0],
		y.n + i * v[1],
		(j-i) * v[0],
		(j-i) * v[1],
		**kw,
	)
	for s in (-1.96, +1.96):
		i, j = 0.2, 0.85

		(_xeq,), (_yeq,) = E.projected_D47eq([x], [y], slope + s * slope.s)

		v = _np.array([
			_xeq.n - x.n,
			_yeq.n - y.n,
		])

		_ppl.arrow(
			x.n + i * v[0],
			y.n + i * v[1],
			(j-i) * v[0],
			(j-i) * v[1],
			alpha = 0.25,
			**kw,
		)

for x, y, xeq, yeq, xp, yp, pv in zip(X, Y, D47eq, D48eq, D47p, D48p, pD47):
	if pv >= p_cutoff:
		_ppl.text(
			x.n, y.n + 5*y.s,
			'($Δ_{47}, Δ_{48}$)\nobservation',
			ha = 'center', va = 'top', size = 8,
		)
		_ppl.text(
			x.n, y.n + 5.5*y.s,
			f'Equil. p-value = {pv:.2f}',
			ha = 'center', va = 'bottom', size = 8, color = eq_color,
		)
		t = E.T_as_function_of_D47(xeq)
		_ppl.text(
			xeq.n + 4*xeq.s,
			yeq.n - 5 * yeq.s,
			f'T = {t.n:.1f}±{t.s:.1f}°C',
			ha = 'left', va = 'top', size = 8, color = eq_color,
		)
	else:
		_ppl.text(
			x.n, y.n + 5*y.s,
			'($Δ_{47}, Δ_{48}$)\nobservation',
			ha = 'center', va = 'top', size = 8,
		)
		_ppl.text(
			x.n, y.n + 5.5*y.s,
			f'Equil. p-value = {pv:.0e}',
			ha = 'center', va = 'bottom', size = 8, color = diseq_color,
		)
		t = E.T_as_function_of_D47(xp)
		_ppl.text(
			xp.n + xp.s,
			yp.n - 3 * xp.s,
			f'T = {t.n:.1f}±{t.s:.1f}°C',
			ha = 'left', va = 'top', size = 8, color = diseq_color,
		)
		m = 0.5
		_ppl.text(
			(m * x.n + (1-m) * xp.n),
			(m * y.n + (1-m) * yp.n),
			'disequilibrium slope\n(with uncertainty)\n',
			ha = 'left', va = 'bottom', size = 8, color = diseq_color,
		)

_ppl.text(
	0.5, 0.02,
	"""
Inputs: $Δ_{47}$ and $Δ_{48}$ measurements, with arbitrary errors (covariance matrix).
Outputs: equilibrium p-values and T estimates with T covariance matrix fully accounting for $Δ_{47}$ and $Δ_{48}$
measurement uncertainties, $Δ_{47}$ and $Δ_{48}$ calibration uncertainties, and the disequilibrium slope uncertainty.""",
	size = 6.5, va = 'bottom', ha = 'center', transform = _ppl.gca().transAxes,
)

_ppl.text(
	1, 1.01, 'M. Daëron 2024-10',
	transform = _ppl.gca().transAxes,
	size = 6,
	alpha = 0.25,
	ha = 'right',
	va = 'bottom',
)

_ppl.axis('equal')
_ppl.axis([0.15, 0.78, None, None])
_ppl.savefig('example_plot.pdf')
_ppl.savefig('example_plot.png', dpi = 150)

data['pvalue_eq'] = pD47
data['Teq'] = E.T_as_function_of_D47(D47eq)
data['Tkp'] = E.T_as_function_of_D47(D47p)

save_data_to_file(data, 'output.csv')

save_Teq_report(
	X,
	Y,
	data['Teq'],
	pD47,
	'Teq-report.csv',
)
