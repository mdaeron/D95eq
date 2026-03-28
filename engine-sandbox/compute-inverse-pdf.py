from pylab import *
from scipy import stats
import numpy as np
from scipy import stats, interpolate

def transform_pdf(f, x0, sigma_x, x_range=1, n_points=40000):
    """
    Compute the PDF of f(X) where X ~ Normal(x0, sigma_x).

    Uses Monte Carlo sampling + KDE for arbitrary continuous functions.
    Falls back cleanly even for non-monotonic f.

    Args:
        f:        callable, the transformation f(X)
        x0:       mean of X
        sigma_x:  std dev of X
        x_range:  how many sigma to cover (default 5)
        n_points: number of sample points

    Returns:
        y_vals:   array of y values where PDF is evaluated
        pdf_vals: corresponding PDF values
    """
    # Sample X from its Gaussian distribution
    x_samples = np.random.normal(x0, sigma_x, size=n_points)

    # Apply transformation
    y_samples = f(x_samples)

    # Filter NaNs
    y_samples = y_samples[~np.isnan(y_samples)]

    # Estimate PDF using KDE
    kde = stats.gaussian_kde(y_samples)

    y_vals = np.linspace(y_samples.min(), y_samples.max(), n_points)
    pdf_vals = kde(y_vals)

    return y_vals, pdf_vals, y_samples


D47  = 0.300
sD47 = 0.050

def transform_pdf_monotonic(f_inv, df_inv, x0, sigma_x, y_vals):
    """
    Exact PDF for monotonic f using change of variables:
        p_Y(y) = p_X(f⁻¹(y)) · |df⁻¹/dy|

    Args:
        f_inv:   inverse of f
        df_inv:  derivative of f_inv (can use autograd/sympy or finite diff)
    """
    x_vals = f_inv(y_vals)
    px = stats.norm.pdf(x_vals, loc=x0, scale=sigma_x)
    return px * np.abs(df_inv(y_vals))

# Example: f(x) = exp(x), exact log-normal PDF
Ti = np.linspace(0, 1000, 1001)
p = transform_pdf_monotonic(
    f_inv = lambda T: 0.18 + 36e3 / (T+273.15)**2,
    df_inv  = lambda T: -2 * 36e3 / (T+273.15)**3,
    x0      = D47,
    sigma_x = sD47,
    y_vals  = Ti
)

y_vals, pdf_vals, y_samples = transform_pdf(
	lambda D: ((D-0.18)/36e3)**-0.5 - 273.15,
	D47,
	sD47,
)

hist(y_samples, bins = Ti[::20])
pmax = axis()[-1]*0.97
plot(Ti, p/p.max()*pmax, 'r-')
show()
