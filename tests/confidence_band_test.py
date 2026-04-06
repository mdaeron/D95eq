from D95thermo import *
from uncertainties import *
from pylab import *

E = Engine()

def test_confband(
	Tmin = 0,
	Tmax = 1000,
	Ni = 1000,
):
	sqinvTmin = (Tmin + 273.15)**-2
	sqinvTmax = (Tmax + 273.15)**-2
	t = linspace(sqinvTmax, sqinvTmin, Ni)**-0.5 - 273.15

	fx = E.D47_calib_function
	fy = E.D48_calib_function

	fig = figure()
	plot(fx(t).n, fy(t).n, 'k-', lw = 0.75)
	a = 1
	for p in [0.5, 0.9, 0.99]:
		a *= 0.5
		C = confidence_band(t, fx, fy, p)
		plot(*C.T, 'r-', alpha = a, lw = 0.75)
	fig.savefig('tests/confidence_band.pdf')

test_confband()
