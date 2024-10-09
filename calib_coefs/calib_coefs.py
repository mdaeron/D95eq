import numpy as _np
import correldata
from D95thermo import *
from D95thermo import _compute_D48_calib_coefficients


coefs = D47_calib_coefs
data = {
	'degree' : _np.array([k for k,v in enumerate(coefs)]),
	'coef': correldata.uarray(coefs)}
correldata.save_data_to_file(data, 'D47_calib_coefs.csv', max_correl_precision = 9)

coefs = _compute_D48_calib_coefficients()
data = {
	'degree' : _np.array([k for k,v in enumerate(coefs)]),
	'coef': correldata.uarray(coefs)}
correldata.save_data_to_file(data, 'D48_calib_coefs.csv', max_correl_precision = 9)
