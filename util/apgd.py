# ############################################################################
# apgd.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Accelerated Proximal Gradient Descent (APGD) Solver.
"""

import time

import numpy as np
import pyunlocbox as opt
from pyunlocbox.functions import dummy

from util.utils import eighMax

class l2_loss(opt.functions.func):
    r"""
    L2 loss function of the form

    :math:

    f(\bbx; \bbSigma, \bbA) = \norm{\bbSigma - \bbA \diag(\bbx) \bbA^{H}}{F}^{2}
    """

    def __init__(self, S, A):
        """
        Parameters
        ----------
        S : :py:class:`~numpy.ndarray`
            (M, M) visibility matrix.
        A : :py:class:`~numpy.ndarray`
            (M, N) system steering matrix.
        """
        M, N = A.shape
        if not ((S.shape[0] == S.shape[1]) and (S.shape[0] == M)):
            raise ValueError('Parameters[S, A] are inconsistent.')
        if not np.allclose(S, S.conj().T):
            raise ValueError('Parameter[S] must be Hermitian.')

        super().__init__()
        self._S = S.copy()
        self._A = A.copy()

    def _eval(self, x):
        """
        Function evaluation.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            The evaluation point.

            If `x` is a matrix, the function gets evaluated for each column, as
            if it was a set of independent problems.

        Returns
        -------
        z : float
            The objective function evaluated at `x`.

            If `x` is a matrix, the sum of the objectives is returned.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        M, N = self._A.shape
        Q = x.shape[1]
        B = ((self._A.reshape(1, M, N) * x.reshape(N, 1, Q).T) @
             self._A.conj().T) - self._S

        z = np.sum(B * B.conj()).real
        return z

    def _grad(self, x):
        """
        Function gradient.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            The evaluation point.

            If `x` is a matrix, the function gets evaluated for each column, as
            if it was a set of independent problems.

        Returns
        -------
        z : :py:class:`~numpy.ndarray`
            The objective function gradient evaluated for each column of `x`.
        """
        was_1d = (x.ndim == 1)
        if was_1d:
            x = x.reshape(-1, 1)

        M, N = self._A.shape
        Q = x.shape[1]
        B = ((self._A.reshape(1, M, N) * x.reshape(N, 1, Q).T) @
             self._A.conj().T) - self._S

        z = 2 * np.sum(self._A.conj() * (B @ self._A), axis=1).real.T
        if was_1d:
            z = z.reshape(-1)
        return z


class elastic_net_loss(opt.functions.func):
    """
    Elastic-net regularizer.
    """

    def __init__(self, lambda_, gamma):
        r"""
        Parameters
        ----------
        lambda_ : float
            Regularization parameter \ge 0.
        gamma : float
            Linear trade-off between lasso and ridge regularizers.
        """
        if lambda_ < 0:
            raise ValueError('Parameter[lambda_] must be positive.')
        if not (0 <= gamma <= 1):
            raise ValueError('Parameter[gamma] must be in (0, 1).')

        super().__init__()
        self._lambda = lambda_
        self._gamma = gamma

    def _eval(self, x):
        """
        Function evaluation.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            The evaluation point.

            If `x` is a matrix, the function gets evaluated for each column, as
            if it was a set of independent problems.

        Returns
        -------
        z : float
            The objective function evaluated at `x`.

            If `x` is a matrix, the sum of the objectives is returned.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        L1_term = self._gamma * np.sum(np.abs(x), axis=0)
        L2_term = (1 - self._gamma) * np.sum(x ** 2, axis=0)

        z = np.sum(self._lambda * (L1_term + L2_term))
        return z

    def _prox(self, x, alpha):
        r"""
        Function proximal operator.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            The evaluation point.

            If `x` is a matrix, the function gets evaluated for each column, as
            if it was a set of independent problems.

        alpha : float
            Regularization parameter.

        Returns
        -------
        z : :py:class:`~numpy.ndarray`
            The proximal operator evaluated for each column of `x`.

        Notes
        -----
        The proximal operator is given by

        :math:
        \prox_{\alpha g}(\bbx) = \argmin_{\bbu} \frac{1}{2} \norm{\bbx - \bbu}{2}^{2} + \alpha g(\bbu)
        """
        C1 = self._lambda * alpha * self._gamma
        C2 = 2 * self._lambda * alpha * (1 - self._gamma) + 1

        z = np.clip((x - C1) / C2, a_min=0, a_max=None)
        return z


class ground_truth_accel(opt.acceleration.accel):
    r"""
    Acceleration scheme used to evaluate Acoustic Camera ground-truth.

    Highlights
    ----------
    * Chooses GD step size as :math:`1/L_{\grad{f}}`;
    * Implements Nesterov acceleration.
    """

    def __init__(self, d, L, momentum=True):
        """
        Parameters
        ----------
        d : float
            Weight parameter as defined in [1].
        L : float
            Lipschitz constant of the gradient of the smooth function being
            optimized.
        momentum : bool
            If :py:obj:`False`, disable Nesterov acceleration.

        Notes
        -----
        [1] On the Convergence of the Iterates of the "Fast Iterative
            [Shrinkage/Thresholding Algorithm" [Chambolle, Dossal]
        """
        super().__init__()

        if d < 2:
            raise ValueError('Parameter[d] is out of range.')

        self._d = d
        self._step = 1 / L
        self._sol_prev = 0
        self._momentum = momentum

    def _pre(self, functions, x0):
        """
        Pre-processing specific to the acceleration scheme.
        """
        pass

    def _update_step(self, solver, objective, niter):
        """
        Update the step size for the next iteration.

        Parameters
        ----------
        solver : :py:class:`~pyunlocbox.solvers.solver`
            Solver on which to act.
        objective : list(float)
            Evaluations of the objective function since the beginning of the iterative process.
        niter : int
            Current iteration number >= 1.

        Returns
        -------
        step : float
            Updated step size.
        """
        return self._step

    def _update_sol(self, solver, objective, niter):
        """
        Update the solution point for the next iteration.

        Parameters
        ----------
        solver : :py:class:`~pyunlocbox.solvers.solver`
            Solver on which to act.
        objective : list(float)
            Evaluations of the objective function since the beginning of the iterative process.
        niter : int
            Current iteration number >= 1.

        Returns
        -------
        step : :py:class:`~numpy.ndarray`
            (N,) updated solution point.
        """
        if self._momentum is True:
            step = (niter - 1) / (niter + self._d)
            sol = solver.sol + step * (solver.sol - self._sol_prev)
        else:
            sol = solver.sol
        self._sol_prev = solver.sol
        return sol

    def _post(self):
        """
        Post-processing specific to the acceleration scheme.
        """
        pass


def _solve(functions, x0, solver=None, atol=None, dtol=None, rtol=1e-3, xtol=None, maxit=200, verbosity='LOW'):
    r"""
    Solve an optimization problem whose objective function is the sum of some
    convex functions.

    This function minimizes the objective function :math:`f(x) =
    \sum\limits_{k=0}^{k=K} f_k(x)`, i.e. solves
    :math:`\operatorname{arg\,min}\limits_x f(x)` for :math:`x \in
    \mathbb{R}^{n \times N}` where :math:`n` is the dimensionality of the data
    and :math:`N` the number of independent problems. It returns a dictionary
    with the found solution and some information about the algorithm
    execution.

    Note
    ----
    This code is taken from pyunlocbox. Our goal is to modify the function to also
    return intermediate solutions.

    Parameters
    ----------
    functions : list of objects
        A list of convex functions to minimize. These are objects who must
        implement the :meth:`pyunlocbox.functions.func.eval` method. The
        :meth:`pyunlocbox.functions.func.grad` and / or
        :meth:`pyunlocbox.functions.func.prox` methods are required by some
        solvers. Note also that some solvers can only handle two convex
        functions while others may handle more. Please refer to the
        documentation of the considered solver.
    x0 : array_like
        Starting point of the algorithm, :math:`x_0 \in \mathbb{R}^{n \times
        N}`. Note that if you pass a numpy array it will be modified in place
        during execution to save memory. It will then contain the solution. Be
        careful to pass data of the type (int, float32, float64) you want your
        computations to use.
    solver : solver class instance, optional
        The solver algorithm. It is an object who must inherit from
        :class:`pyunlocbox.solvers.solver` and implement the :meth:`_pre`,
        :meth:`_algo` and :meth:`_post` methods. If no solver object are
        provided, a standard one will be chosen given the number of convex
        function objects and their implemented methods.
    atol : float, optional
        The absolute tolerance stopping criterion. The algorithm stops when
        :math:`f(x^t) < atol` where :math:`f(x^t)` is the objective function at
        iteration :math:`t`. Default is None.
    dtol : float, optional
        Stop when the objective function is stable enough, i.e. when
        :math:`\left|f(x^t) - f(x^{t-1})\right| < dtol`. Default is None.
    rtol : float, optional
        The relative tolerance stopping criterion. The algorithm stops when
        :math:`\left|\frac{ f(x^t) - f(x^{t-1}) }{ f(x^t) }\right| < rtol`.
        Default is :math:`10^{-3}`.
    xtol : float, optional
        Stop when the variable is stable enough, i.e. when :math:`\frac{\|x^t -
        x^{t-1}\|_2}{\sqrt{n N}} < xtol`. Note that additional memory will be
        used to store :math:`x^{t-1}`. Default is None.
    maxit : int, optional
        The maximum number of iterations. Default is 200.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        The log level : ``'NONE'`` for no log, ``'LOW'`` for resume at
        convergence, ``'HIGH'`` for info at all solving steps, ``'ALL'`` for
        all possible outputs, including at each steps of the proximal operators
        computation. Default is ``'LOW'``.

    Returns
    -------
    sol : ndarray
        The problem solution.
    solver : str
        The used solver.
    crit : {'ATOL', 'DTOL', 'RTOL', 'XTOL', 'MAXIT'}
        The used stopping criterion. See above for definitions.
    niter : int
        The number of iterations.
    time : float
        The execution time in seconds.
    objective : ndarray
        The successive evaluations of the objective function at each iteration.
    backtrace : ndarray
        (N_iter + 1, len(sol)) past values of solution.
    """
    if verbosity not in ['NONE', 'LOW', 'HIGH', 'ALL']:
        raise ValueError('Verbosity should be either NONE, LOW, HIGH or ALL.')

    # Add a second dummy convex function if only one function is provided.
    if len(functions) < 1:
        raise ValueError('At least 1 convex function should be provided.')
    elif len(functions) == 1:
        functions.append(dummy())
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            print('INFO: Dummy objective function added.')

    # Choose a solver if none provided.
    if not solver:
        if len(functions) == 2:
            fb0 = 'GRAD' in functions[0].cap(x0) and \
                  'PROX' in functions[1].cap(x0)
            fb1 = 'GRAD' in functions[1].cap(x0) and \
                  'PROX' in functions[0].cap(x0)
            dg0 = 'PROX' in functions[0].cap(x0) and \
                  'PROX' in functions[1].cap(x0)
            if fb0 or fb1:
                solver = forward_backward()  # Need one prox and 1 grad.
            elif dg0:
                solver = douglas_rachford()  # Need two prox.
            else:
                raise ValueError('No suitable solver for the given functions.')
        elif len(functions) > 2:
            solver = generalized_forward_backward()
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('INFO: Selected solver: {}'.format(name))

    # Set solver and functions verbosity.
    translation = {'ALL': 'HIGH', 'HIGH': 'HIGH', 'LOW': 'LOW', 'NONE': 'NONE'}
    solver.verbosity = translation[verbosity]
    translation = {'ALL': 'HIGH', 'HIGH': 'LOW', 'LOW': 'NONE', 'NONE': 'NONE'}
    functions_verbosity = []
    for f in functions:
        functions_verbosity.append(f.verbosity)
        f.verbosity = translation[verbosity]

    tstart = time.time()
    crit = None
    niter = 0
    objective = [[f.eval(x0) for f in functions]]
    rtol_only_zeros = True

    # Solver specific initialization.
    solver.pre(functions, x0)
    tape_buffer = np.zeros((1000, len(x0)))
    tape_buffer[0] = x0

    while not crit:
        niter += 1

        if xtol is not None:
            last_sol = np.array(solver.sol, copy=True)

        if verbosity in ['HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('Iteration {} of {}:'.format(niter, name))

        # Solver iterative algorithm.
        solver.algo(objective, niter)
        tape_buffer[niter] = solver.sol

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Verify stopping criteria.
        if atol is not None and current < atol:
            crit = 'ATOL'
        if dtol is not None and np.abs(current - last) < dtol:
            crit = 'DTOL'
        if rtol is not None:
            div = current  # Prevent division by 0.
            if div == 0:
                if verbosity in ['LOW', 'HIGH', 'ALL']:
                    print('WARNING: (rtol) objective function is equal to 0 !')
                if last != 0:
                    div = last
                else:
                    div = 1.0  # Result will be zero anyway.
            else:
                rtol_only_zeros = False
            relative = np.abs((current - last) / div)
            if relative < rtol and not rtol_only_zeros:
                crit = 'RTOL'
        if xtol is not None:
            err = np.linalg.norm(solver.sol - last_sol)
            err /= np.sqrt(last_sol.size)
            if err < xtol:
                crit = 'XTOL'
        if maxit is not None and niter >= maxit:
            crit = 'MAXIT'

        if verbosity in ['HIGH', 'ALL']:
            print('    objective = {:.2e}'.format(current))

    # Restore verbosity for functions. In case they are called outside solve().
    for k, f in enumerate(functions):
        f.verbosity = functions_verbosity[k]

    if verbosity in ['LOW', 'HIGH', 'ALL']:
        print('Solution found after {} iterations:'.format(niter))
        print('    objective function f(sol) = {:e}'.format(current))
        print('    stopping criterion: {}'.format(crit))

    # Returned dictionary.
    result = {'sol': solver.sol,
              'solver': solver.__class__.__name__,  # algo for consistency ?
              'crit': crit,
              'niter': niter,
              'time': time.time() - tstart,
              'objective': objective}
    try:
        # Update dictionary for primal-dual solvers
        result['dual_sol'] = solver.dual_sol
    except AttributeError:
        pass

    # Solver specific post-processing (e.g. delete references).
    solver.post()

    result['backtrace'] = tape_buffer[:(niter + 1)]
    return result


def solve(S, A, lambda_=None, gamma=0.5, L=None, d=50, x0=None, eps=1e-3,
          N_iter_max=200, verbosity='LOW', momentum=True):
    """
    APGD solution to the Acoustic Camera problem. (Algorithm 3.1)

    Parameters
    ----------
    S : :py:class:`~numpy.ndarray`
        (M, M) visibility matrix
    A : :py:class:`~numpy.ndarray`
        (M, N) system steering matrix.
    lambda_ : float
        Regularization parameter.

        If `None`, then it is chosen according to Remark 3.4.
    gamma : float
        Linear trade-off between lasso and ridge regularizers.
    L : float
        Lipschitz constant of the gradient of the smooth function being
        optimized.

        If `None`, then it is estimated using
        :py:func:`~deepwave.tools.math.linalg.eighMax`.
    d : float
        Weight parameter as defined in [1].
    x0 : :py:class:`~numpy.ndarray`
        (N,) initial intensity field estimate.

        Defaults to 0 if not explicitly initialized.
    eps : float
        Relative tolerance stopping threshold.
    N_iter_max : int
        Maximum number of iterations.
    verbosity : str
        One of 'NONE', 'LOW', 'HIGH', 'ALL'.
    momentum : bool
        If :py:class:`False`, disable Nesterov acceleration.

    Returns
    -------
    I_opt : dict
        sol : :py:class:`~numpy.ndarray`
            (N,) optimal intensity field.
        solver : str
            The used solver.
        crit : {‘ATOL’, ‘DTOL’, ‘RTOL’, ‘XTOL’, ‘MAXIT’}
            The used stopping criterion.
        niter : int
            The number of iterations.
        time : float
            The execution time in seconds.
        objective : :py:class:`~numpy.ndarray`
            The successive evaluations of the objective function at each
            iteration.
        backtrace : :py:class:`~numpy.ndarray`
            (N_iter + 1, N) successive values of the objective parameter at
            each iteration.
            backtrace[0] holds the initial solution.
        L : float
            Lipschitz constant of the gradient of the smooth function being optimized.
        lambda_ : float
            Regularization parameter
        gamma : float
            Linear trade-off between lasso and ridge regularizers.

    Notes
    -----
    [1] On the Convergence of the Iterates of the "Fast Iterative
        [Shrinkage/Thresholding Algorithm" [Chambolle, Dossal]
    """
    M, N = A.shape
    if not ((S.shape[0] == S.shape[1]) and (S.shape[0] == M)):
        raise ValueError('Parameters[S, A] are inconsistent.')
    if not np.allclose(S, S.conj().T):
        raise ValueError('Parameter[S] must be Hermitian.')

    if not (0 <= gamma <= 1):
        raise ValueError('Parameter[gamma] is must lie in [0, 1].')

    if L is None:
        L = 2 * eighMax(A)
    elif L <= 0:
        raise ValueError('Parameter[L] must be positive.')

    if d < 2:
        raise ValueError(r'Parameter[d] must be \ge 2.')

    if x0 is None:
        x0 = np.zeros((N,), dtype=np.float64)
    elif np.any(x0 < 0):
        raise ValueError('Parameter[x0] must be non-negative.')

    if not (0 < eps < 1):
        raise ValueError('Parameter[eps] must lie in (0, 1).')

    if N_iter_max < 1:
        raise ValueError('Parameter[N_iter_max] must be positive.')

    if verbosity not in ('NONE', 'LOW', 'HIGH', 'ALL'):
        raise ValueError('Unknown verbosity specification.')

    if lambda_ is None:
        if gamma > 0:  # Procedure of Remark 3.4
            # When gamma == 0, we fall into the ridge-regularizer case, so no
            # need to do the following.
            func = [l2_loss(S, A), elastic_net_loss(lambda_=0, gamma=gamma)]
            solver = opt.solvers.forward_backward(accel=ground_truth_accel(d, L, momentum=False))
            I_opt = _solve(functions=func,
                           x0=np.zeros((N,)),
                           solver=solver,
                           rtol=eps,
                           maxit=1,
                           verbosity=verbosity)
            alpha = 1 / L
            lambda_ = np.max(I_opt['sol']) / (10 * alpha * gamma)
        else:
            lambda_ = 1  # Anything will do.
    elif lambda_ < 0:
        raise ValueError('Parameter[lambda_] must be non-negative.')

    func = [l2_loss(S, A), elastic_net_loss(lambda_, gamma)]
    solver = opt.solvers.forward_backward(accel=ground_truth_accel(d, L, momentum))
    I_opt = _solve(functions=func,
                   x0=x0.copy(),
                   solver=solver,
                   rtol=eps,
                   maxit=N_iter_max,
                   verbosity=verbosity)
    I_opt['gamma'] = gamma
    I_opt['lambda_'] = lambda_
    I_opt['L'] = L
    return I_opt
