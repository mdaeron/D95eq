from D95eq import *
from pylab import *
import uncertainties as _uc
import correldata as _cd

def test_nearest_Teq():

	print()
	for D47_coefs in (None, 0.99, 1.01):
		for D48_coefs in (None, 0.99, 1.01):
			E = Engine(
				D47_coefs = None if D47_coefs is None else Engine.D47_calib_coefs * D47_coefs,
				D48_coefs = None if D48_coefs is None else Engine.D48_calib_coefs * D48_coefs,
			)

			for D47, sD47, D48, sD48 in (
				(0.60, 0.005, 0.35, 0.025),
				(0.60, 0.005, 0.35, 0.100),
			):
				(D47eq,), (D48eq,), (p,) = E.nearest_D47eq(
					D47 = _cd.uarray([_uc.ufloat(D47, sD47)]),
					D48 = _cd.uarray([_uc.ufloat(D48, sD48)]),
				)
				T = E.T_as_function_of_D47(D47eq)
				print('nearest')
				print(f"D47 = {D47:.3f} ± {sD47:.3f}, D48 = {D48:.3f} ± {sD48:.3f}, T = {T:.2f}, p = {p:.8f}")

				(D47eq,), (D48eq,), (p,) = E.joint_nearest_D47eq(
					D47 = _cd.uarray([_uc.ufloat(D47, sD47)]),
					D48 = _cd.uarray([_uc.ufloat(D48, sD48)]),
				)
				T = E.T_as_function_of_D47(D47eq)
				print('joint')
				print(f"D47 = {D47:.3f} ± {sD47:.3f}, D48 = {D48:.3f} ± {sD48:.3f}, T = {T:.2f}, p = {p:.8f}")

				(D47eq,), (D48eq,), (p,) = E.lazy_joint_nearest_D47eq(
					D47 = _cd.uarray([_uc.ufloat(D47, sD47)]),
					D48 = _cd.uarray([_uc.ufloat(D48, sD48)]),
				)
				T = E.T_as_function_of_D47(D47eq)
				print('lazy')
				print(f"D47 = {D47:.3f} ± {sD47:.3f}, D48 = {D48:.3f} ± {sD48:.3f}, T = {T:.2f}, p = {p:.8f}")
