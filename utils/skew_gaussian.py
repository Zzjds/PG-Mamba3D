import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import (multivariate_normal as mvn, norm)
from   scipy.stats._multivariate import _squeeze_output


class multivariate_skewnorm:
    
    def __init__(self, shape, cov=None):
        self.dim   = len(shape)
        self.shape = np.asarray(shape)
        self.mean  = np.zeros(self.dim)
        self.cov   = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x):
        return np.exp(self.logpdf(x))
        
    def logpdf(self, x):
        x    = mvn._process_quantiles(x, self.dim)
        pdf  = mvn(self.mean, self.cov).logpdf(x)
        cdf  = norm(0, 1).logcdf(np.dot(x, self.shape))
        return _squeeze_output(np.log(2) + pdf + cdf)
    
    def rvs_fast(self, size=1):
        aCa      = self.shape @ self.cov @ self.shape
        delta    = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
        cov_star = np.block([[np.ones(1),     delta],
                            [delta[:, None], self.cov]])
        x        = mvn(np.zeros(self.dim+1), cov_star).rvs(size)
        x0, x1   = x[:, 0], x[:, 1:]
        inds     = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1


def generate_skew_kernel(radius=3, sigma=4, bias=(0, 0)):
    kernel = []
    generator = multivariate_skewnorm(bias, (sigma**2, sigma**2))
    size = 2 * radius +1
    x = np.arange(size)-radius
    y = np.arange(size)-radius
    kernel = []
    for xi in x:
        for yi in y:
            kernel.append(generator.pdf([xi, yi]))
    kernel = np.array(kernel).reshape((size, size)) / np.sum(kernel)
    return kernel


if __name__ == "__main__":
    x = generate_skew_kernel(radius=3, sigma=4, bias=(0, 0))