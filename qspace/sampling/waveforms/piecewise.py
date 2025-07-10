#!/usr/bin/env python3
"""Let's refactor the piecewise polynomial basis construction to make it more
versatile"""
from scipy.interpolate import (BPoly, PPoly)
import numpy as np
from scipy.linalg import (eigh, null_space)
from scipy.special import binom
from matplotlib import pyplot as plt
from collections.abc import Sequence


gamma = 267.513e6

class PiecewiseBasis(Sequence):
    """Builds an orthonormal basis of piecewise polynomial functions satisfying
    some constraints. 
    """
    def __init__(self, order, breakpoints):
        self.order = order
        self.te = np.max(breakpoints)
        self.breakpoints = breakpoints
        self.nb_intervals = len(breakpoints) - 1
        self.constraints = []


    def vanishing_constraint(self, breakpoint, k=0):
        """Adds a vanishing constraint of the k-th derivative at 
        breakpoint[index].
        """
        # raises a ValueError if breakpoint not in self.breakpoints
        i = self.breakpoints.index(breakpoint)
        if i > 0:
            # Add a vanishing constraint at the left of breakpoint
            new_constraint = np.zeros((self.order + 1, self.nb_intervals))
            for j in range(self.order + 1):
                 new_constraint[j, i - 1] = \
                   _bernstein_kth_derivative(self.order, j, k, loc=1)
            self.constraints.append(new_constraint)
        if i < self.nb_intervals:
            # Add a vanishing constraint at the right of breakpoint
            new_constraint = np.zeros((self.order + 1, self.nb_intervals))
            for j in range(self.order + 1):
                 new_constraint[j, i] = \
                   _bernstein_kth_derivative(self.order, j, k, loc=0)
            self.constraints.append(new_constraint)


    def regularity_constraint(self, breakpoint, smoothness):
        """Adds a regularity constraint, i.e. all derivatives up to 
        `smoothness` should be continuous at `breakpoint`.
        """
        # raises a ValueError if breakpoint not in self.breakpoints
        i = self.breakpoints.index(breakpoint)
        intervals = np.diff(self.breakpoints)
        if i == 0 or i == self.nb_intervals:
            return

        for k in range(smoothness + 1):
            new_constraint = np.zeros((self.order + 1, 
                                              self.nb_intervals))
            for p in range(self.order + 1):
                new_constraint[p, i - 1] = -1 / intervals[i-1]**k * \
                  _bernstein_kth_derivative(self.order, p, k, loc=1)
                new_constraint[p, i]     = 1 / intervals[i]**k * \
                  _bernstein_kth_derivative(self.order, p, k, loc=0)
            self.constraints.append(new_constraint)


    def symmetry_constraint(self, interval1, interval2, sign=-1, reverse=True):
        """Adds a symmetry constraint, such that values of the functions' 
        derivative in interval1 should match values in interval2, in reverse
        (mirror) and opposite (sign=-1). This is quite specialized for 
        refocusing in diffusion MRI.
        """
        assert sign in [-1, 1]
        index1 = self.breakpoints.index(interval1[0])
        assert(self.breakpoints.index(interval1[1]) == (index1 + 1))
        index2 = self.breakpoints.index(interval2[0])
        assert(self.breakpoints.index(interval2[1]) == (index2 + 1))

        if reverse:
            for j in range(self.order):
                new_constraint = np.zeros((self.order + 1, 
                                                  self.nb_intervals))
                new_constraint[             j, index1] =  sign
                new_constraint[           j+1, index1] = -sign
                new_constraint[  self.order-j, index2] =  1
                new_constraint[self.order-j-1, index2] = -1
                self.constraints.append(new_constraint)
        else:
            for j in range(self.order):
                new_constraint = np.zeros((self.order + 1, 
                                                  self.nb_intervals))
                new_constraint[  j, index1] =  sign
                new_constraint[j+1, index1] = -sign
                new_constraint[  j, index2] =  1
                new_constraint[j+1, index2] = -1
                self.constraints.append(new_constraint)


    def order_constraint(self, order, interval):
        """Imposes maximum order over a given interval.
        """
        index = self.breakpoints.index(interval[0])
        assert(self.breakpoints.index(interval[1]) == (index + 1))
        M = _bernstein_to_power(self.order, interval)

        for k in range(order + 1, self.order + 1):
            l = self.order - k
            new_constraint = np.zeros((self.order + 1, self.nb_intervals))
            new_constraint[:, index] = M[l]
            self.constraints.append(new_constraint)


    def compute(self):
        nb_constraints = len(self.constraints)
        dimension = (self.order + 1) * self.nb_intervals
        constraints = np.array(self.constraints)
        constraints = constraints.reshape((nb_constraints, dimension))
        solutions = null_space(constraints)
        nb_solutions = solutions.shape[1]
     
        # Let's compute an orthonormal basis of functions from these solutions
        # We construct the matrix M s.t. M[i, j] = \int_0^{te} f_i f_j
        M = np.zeros((nb_solutions, nb_solutions))
        for i in range(nb_solutions):
            ai = solutions[:, i].reshape((self.order + 1, self.nb_intervals))
            for j in range(i, nb_solutions):
                aj = solutions[:, j].reshape((self.order + 1, 
                                              self.nb_intervals))
                prod = BPoly(_bernstein_product(ai, aj), self.breakpoints)
                M[i, j] = M[j, i] = prod.integrate(0, self.te)
        
        # The eigenbasis of this matrix defines a set of orthogonal functions.
        w, v = eigh(M)
        basis = []
        for i in range(nb_solutions):
            coefficients = 1 / np.sqrt(w[i]) * np.dot(solutions, v[:, i])
            coefficients = coefficients.reshape((self.order + 1, 
                                                 self.nb_intervals))
            f = BPoly(coefficients, self.breakpoints)
            basis.append(f)
        self.basis = basis
        self.dimension = nb_solutions


    def __getitem__(self, index):
        return self.basis[index]


    def __len__(self):
        return len(self.basis)


    def __str__(self):
        return f"Piecewise_order_{self.order}"


def _bernstein_kth_derivative(order, i, k, loc=0, interval=[0, 1]):
    """Computes the `k`-th derivative of Bernstein polynomial `i`.

    Parameters
    ----------
    order : int
        Order of the Bernstein polynomial.
    i : int
        Index of the Bernstein polynomial.
    k : int
        degree of the derivative.
    """
    assert(i >=0 and i <= order)
    c = np.zeros((order + 1, 1))
    c[i, 0] = 1
    f = BPoly(c, interval)
    for j in range(k):
        f = f.derivative()
    return f(loc)


def _bernstein_product(a, b):
    """Provided two bernstein piecewise polynomials represented by their 
    coefficients, returns the coefficients of the product [1]. NB: it is 
    assumed that the intervals of the input polynomials match.

    Parameters
    ----------
    a : array-like, shape ()
        

    1. https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node10.html
    """
    m = a.shape[0] - 1
    n = b.shape[0] - 1
    assert(a.shape[1] == b.shape[1])
    nb_intervals = a.shape[1]
    result = np.zeros((m + n + 1, nb_intervals))
    for i in range(m + n + 1):
        j0 = max(0, i - n)
        j1 = min(m, i) + 1
        for j in range(j0, j1):
            result[i] += binom(m, j) * binom(n, i - j) / binom(m + n, i) * \
                         a[j] * b[i - j]
    return result


def _bernstein_to_power(order, interval):
    """Computes the matrix to change from bernstein coefficients to power 
    basis.
    """
    M = np.zeros((order + 1, order + 1))
    for p in range(order + 1):
        coefficients = np.zeros((order + 1, 1))
        coefficients[p, 0] = 1
        f = BPoly(coefficients, interval)
        M[:, p] = PPoly.from_bernstein_basis(f).c[:, 0]
    return M


def _power_to_bernstein(order, interval):
    """Computes the matrix to change from power coefficients to Bernstein
    basis.
    """
    M = np.zeros((order + 1, order + 1))
    for p in range(order + 1):
        coefficients = np.zeros((order + 1, 1))
        coefficients[p, 0] = 1
        f = PPoly(coefficients, interval)
        M[p] = BPoly.from_bernstein_basis(f).c[:, 0]
    return M


class QPiecewiseConstant(PiecewiseBasis):
    """Prepares a basis of q(t) corresponding to piecewise constant gradients
    separated with linear transitions.
    """
    def __init__(self, te, delta_rf, nb_chunks, delta_ramp):
        breakpoints = [0]
        delta_chunk = (te - delta_rf - 2 * (nb_chunks + 1) * delta_ramp) \
                    / (2 * nb_chunks)
        breakpoint = breakpoints[-1]
        for i in range(nb_chunks):
            breakpoint += delta_ramp
            breakpoints.append(breakpoint)
            breakpoint += delta_chunk
            breakpoints.append(breakpoint)
        breakpoints.append(0.5 * (te - delta_rf))
        breakpoints.append(0.5 * (te + delta_rf))
        breakpoint = breakpoints[-1]
        for i in range(nb_chunks):
            breakpoint += delta_ramp
            breakpoints.append(breakpoint)
            breakpoint += delta_chunk
            breakpoints.append(breakpoint)
        breakpoints.append(te)
        nb_breakpoints = len(breakpoints)
        order = 2
        super().__init__(order, breakpoints)
        self.nb_chunks = nb_chunks

        self.vanishing_constraint(0, k=0)
        self.vanishing_constraint(te, k=0)
        self.vanishing_constraint(0, k=1)
        self.vanishing_constraint(te, k=1)
        smoothness = 1
        for breakpoint in breakpoints[1:-1]:
            self.regularity_constraint(breakpoint, smoothness)
        for i in range(2 * nb_chunks + 1):
            interval1 = breakpoints[i:i + 2]
            interval2 = breakpoints[nb_breakpoints - i - 2:nb_breakpoints - i]
            self.symmetry_constraint(interval1, interval2)
        for i in range(nb_chunks):
            interval = breakpoints[2 * i + 1:2 * i + 3]
            self.order_constraint(1, interval)
        interval_rf = breakpoints[2 * nb_chunks + 1:2 * nb_chunks + 3]
        self.order_constraint(0, interval_rf)
        self.compute()


    def __str__(self):
        return f"QPiecewiseConstant_nb_chunks_{self.nb_chunks}"
        
         
class QPiecewisePolynomial(PiecewiseBasis):
    """Prepares a basis of q(t) corresponding to piecewise polynomial 
    gradients.
    """
    def __init__(self, te, delta_rf, nb_chunks, order, first_block=None,
                 symmetrical=True, smoothness=2):
        """
        Parameters
        ----------
        first block : float
            The duration of the first diffusion encoding block [s]. If None, 
            we will assume same duration before/after the refocusing RF pulse.
        """
        if first_block is None:
            first_block = (te - delta_rf) / 2
        second_block = te - delta_rf - first_block
        breakpoints = [0]
        delta_chunk1 = first_block / nb_chunks
        breakpoint = breakpoints[-1]
        for i in range(nb_chunks - 1):
            breakpoint += delta_chunk1
            breakpoints.append(breakpoint)
        breakpoints.append(first_block)
        breakpoints.append(first_block + delta_rf)

        breakpoint = breakpoints[-1]
        delta_chunk2 = second_block / nb_chunks
        for i in range(nb_chunks - 1):
            breakpoint += delta_chunk2
            breakpoints.append(breakpoint)
        breakpoints.append(te)
        nb_breakpoints = len(breakpoints)
        super().__init__(order, breakpoints)
        self.nb_chunks = nb_chunks

        self.vanishing_constraint(0, k=0)
        self.vanishing_constraint(te, k=0)
        self.vanishing_constraint(0, k=1)
        self.vanishing_constraint(te, k=1)
        for breakpoint in breakpoints[1:-1]:
            self.regularity_constraint(breakpoint, smoothness)
        if symmetrical:
            for i in range(nb_chunks):
                interval1 = breakpoints[i:i + 2]
                interval2 = breakpoints[nb_breakpoints - i - 2:nb_breakpoints - i]
                self.symmetry_constraint(interval1, interval2)
        interval_rf = breakpoints[nb_chunks:nb_chunks + 2]
        self.order_constraint(0, interval_rf)
        self.compute()


    def __str__(self):
        return (f"QPiecewisePolynomial_nb_chunks_{self.nb_chunks}_"
                f"order_{self.order}")


def generate_random(b, basis, gmax=None, smax=None, nb_restart=10):
    """Generates a random waveform in the provided basis with a target b-value.

    Parameters
    ----------
    b : double
        Diffusion weighting factor (b-value) [s/m^2]
    basis : BernsteinBasis
    gmax : double
        Maximum gradient magnitude [T/m]
    smax : double
        Maximum gradient slewrate [T/m/s]
    """
    from scipy import optimize as scopt
    def cost(coefficients, x0=None):
        return np.sum((coefficients - x0)**2) / np.sum(x0**2)

    def eq_constraints(coefficients, b=b):
        return np.sum(coefficients ** 2) - b
    
    def ineq_constraints(coefficients, gmax=gmax, basis=basis):
        N = 100
        ts = np.linspace(0, basis.te, N)
        constraints = np.zeros(2 * N)
        for i in range(basis.dimension):
            f = basis[i]
            constraints[:N] += f.derivative()(ts) * coefficients[i]
            constraints[N:] += -f.derivative()(ts) * coefficients[i]
        constraints /= gamma
        constraints += gmax
        return constraints


    dimension = basis.dimension
    success = False
    max_iter = nb_restart
    while not success and max_iter > 0:
        max_iter -= 1
        x0 = np.random.normal(0.0, 1.0, size=dimension)
        x0 *= np.sqrt(b) / np.linalg.norm(x0)
        if gmax is None and smax is None:
            return x0
        if gmax is None or smax is None:
            # Not implemented, would need to cover multiple cases and I'm lazy
            raise ValueError("Provide gmax XNOR smax.")
    
        solution = scopt.minimize(cost, x0, args=(x0, ),
           constraints=({"type": "eq", "fun": eq_constraints, "args": (b,)},
             {"type": "ineq","fun": ineq_constraints, "args": (gmax, )}))
        if solution["success"]:
            success = True
    return solution["x"]

