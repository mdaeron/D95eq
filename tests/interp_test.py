from D95thermo import *
from pylab import *

def test_interp():

	k = 0
	for D47_coefs in (None, 1.01, 0.99):
		for D48_coefs in (None, 1.01, 0.99):
			E = Engine(
				D47_coefs = None if D47_coefs is None else Engine.D47_calib_coefs * D47_coefs,
				D48_coefs = None if D48_coefs is None else Engine.D48_calib_coefs * D48_coefs,
			)

			k += 1
			fig = figure()
			ax = subplot(111)

			E.plot_D95_equilibrium(lw = 0.5)
			xi = linspace(0.7, 0.2, 51)
			yi = E.D48_ufloat_as_function_of_D47_float(xi)
			xi = E.D47_ufloat_as_function_of_D47_float(xi)
			conf_ellipse(xi, yi)
			plot(xi.n, yi.n, 'r+')

			ax.autoscale_view()
			fig.savefig(f'tests/interp_test_{k:03.0f}.pdf')
			close(fig)
