"""
L-BFGS-B optimizer for PyTorch with vectorized operations.

Based on nlesc-dirac/pytorch lbfgsb.py by Sarod Yatawatta.
Primary reference:
  MATLAB code https://github.com/bgranzow/L-BFGS-B by Brian Granzow
Theory:
  A Limited Memory Algorithm for Bound Constrained Optimization,
  Byrd et al. 1995
  Numerical Optimization, Nocedal and Wright, 2006

Vectorized for large-scale problems (225K+ parameters) by replacing
Python for-loops with torch tensor operations.
"""

import math

import torch
from functools import reduce
from torch.optim.optimizer import Optimizer


class LBFGSB(Optimizer):
    """L-BFGS-B: L-BFGS with bound constraints.

    Uses projected gradient and Cauchy point computation to properly
    handle bound constraints, unlike clamping which creates flat
    gradient regions.

    Arguments:
        params: iterable of parameters to optimize
        lower_bound: tensor of lower bounds (same shape as flat params)
        upper_bound: tensor of upper bounds (same shape as flat params)
        max_iter (int): max iterations per optimizer step (default: 10)
        tolerance_grad (float): convergence tolerance on projected
            gradient infinity norm (default: 1e-5)
        tolerance_change (float): convergence tolerance on optimality
            (default: 1e-20)
        history_size (int): L-BFGS history size (default: 7)
    """

    def __init__(
        self,
        params,
        lower_bound,
        upper_bound,
        max_iter=10,
        tolerance_grad=1e-5,
        tolerance_change=1e-20,
        history_size=7,
    ):
        defaults = dict(
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGSB doesn't support per-parameter options "
                "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self._device = self._params[0].device
        self._dtype = self._params[0].dtype

        self._l = lower_bound.clone().contiguous().to(self._device)
        self._u = upper_bound.clone().contiguous().to(self._device)
        self._m = history_size
        self._n = self._numel()

        # Storage matrices for L-BFGS approximation
        self._W = torch.zeros(
            self._n, self._m * 2, dtype=self._dtype, device=self._device
        )
        self._Y = torch.zeros(
            self._n, self._m, dtype=self._dtype, device=self._device
        )
        self._S = torch.zeros(
            self._n, self._m, dtype=self._dtype, device=self._device
        )
        self._M = torch.zeros(
            self._m * 2, self._m * 2, dtype=self._dtype, device=self._device
        )

        self._fit_to_constraints()

        self._eps = tolerance_change
        self._realmax = 1e20
        self._theta = 1

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().contiguous().view(-1)
            else:
                view = p.grad.data.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(
                update[offset : offset + numel].view_as(p.data),
                alpha=step_size,
            )
            offset += numel

    def _copy_params_out(self):
        return [
            p.detach().flatten().clone(memory_format=torch.contiguous_format)
            for p in self._params
        ]

    def _copy_params_in(self, new_params):
        with torch.no_grad():
            for p, pdata in zip(self._params, new_params):
                p.copy_(pdata.view_as(p))

    def _fit_to_constraints(self):
        """Project parameters into [lower, upper] bounds."""
        x = torch.cat(self._copy_params_out(), 0)
        x = torch.clamp(x, self._l, self._u)
        offset = 0
        with torch.no_grad():
            for p in self._params:
                numel = p.numel()
                p.copy_(x[offset : offset + numel].view_as(p))
                offset += numel

    def _get_optimality(self, g):
        """Projected gradient infinity norm (eq 6.1)."""
        x = torch.cat(self._copy_params_out(), 0)
        projected_g = torch.clamp(x - g, self._l, self._u) - x
        return projected_g.abs().max().item()

    def _get_breakpoints(self, x, g):
        """Compute breakpoints for Cauchy point (eq 4.1, 4.2)."""
        t = torch.full(
            (self._n,), self._realmax, dtype=self._dtype, device=self._device
        )
        d = -g.clone()

        neg_mask = g < 0.0
        pos_mask = g > 0.0

        t[neg_mask] = (x[neg_mask] - self._u[neg_mask]) / g[neg_mask]
        t[pos_mask] = (x[pos_mask] - self._l[pos_mask]) / g[pos_mask]

        small_t = t < self._eps
        d[small_t] = 0.0

        F = torch.argsort(t)

        return t, d.unsqueeze(-1), F

    def _get_cauchy_point(self, g):
        """Generalized Cauchy point (algorithm CP, pp. 8-9)."""
        x = torch.cat(self._copy_params_out(), 0)
        tt, d, F = self._get_breakpoints(x, g)
        xc = x.clone()
        c = torch.zeros(
            2 * self._m, 1, dtype=self._dtype, device=self._device
        )
        p = torch.mm(self._W.transpose(0, 1), d)
        fp = -torch.mm(d.transpose(0, 1), d)
        fpp = -self._theta * fp - torch.mm(
            p.transpose(0, 1), torch.mm(self._M, p)
        )
        fp = fp.squeeze()
        fpp = fpp.squeeze()
        fpp0 = -self._theta * fp
        if fpp != 0.0:
            dt_min = -fp / fpp
        else:
            dt_min = -fp / self._eps
        t_old = 0

        # Find first index with positive breakpoint
        positive_mask = tt[F] > 0
        if positive_mask.any():
            first_pos = positive_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            first_pos = self._n

        i = first_pos
        if i < self._n:
            b = F[i].item()
            t_val = tt[b].item()
            dt = t_val - t_old
        else:
            dt = float("inf")

        # Cauchy point loop — iterates through breakpoints
        # In practice converges quickly (typically << n iterations)
        while i < self._n and dt_min > dt:
            if d[b] > 0.0:
                xc[b] = self._u[b]
            elif d[b] < 0.0:
                xc[b] = self._l[b]

            zb = xc[b] - x[b]
            gb = g[b]
            c = c + dt * p
            Wbt = self._W[b, :].unsqueeze(0)  # 1 x 2m
            fp = (
                fp
                + dt * fpp
                + gb * gb
                + self._theta * gb * zb
                - gb * torch.mm(Wbt, torch.mm(self._M, c))
            )
            fpp = (
                fpp
                - self._theta * gb * gb
                - 2.0 * gb * torch.mm(Wbt, torch.mm(self._M, p))
                - gb
                * gb
                * torch.mm(Wbt, torch.mm(self._M, Wbt.transpose(0, 1)))
            )
            fp = fp.squeeze()
            fpp = fpp.squeeze()
            fpp = max(self._eps * fpp0, fpp)
            p = p + gb * Wbt.transpose(0, 1)
            d[b] = 0.0
            if fpp != 0.0:
                dt_min = -fp / fpp
            else:
                dt_min = -fp / self._eps
            t_old = t_val
            i += 1
            if i < self._n:
                b = F[i].item()
                t_val = tt[b].item()
                dt = t_val - t_old

        dt_min = max(dt_min, 0.0)
        t_old = t_old + dt_min

        # Vectorized update for remaining variables
        if i < self._n:
            remaining_idx = F[i:]
            xc[remaining_idx] = x[remaining_idx] + t_old * d[remaining_idx, 0]

        c = c + dt_min * p
        return xc, c

    def _subspace_min(self, g, xc, c):
        """Subspace minimization over free variables (pp. 12)."""
        # Vectorized free variable detection
        free_mask = (xc != self._u) & (xc != self._l)
        free_vars_index = free_mask.nonzero(as_tuple=True)[0]
        n_free_vars = free_vars_index.numel()

        if n_free_vars == 0:
            return xc.clone(), False

        # Vectorized WtZ construction
        WtZ = self._W[free_vars_index, :].transpose(0, 1)  # 2m x n_free

        x = torch.cat(self._copy_params_out(), 0)
        rr = (
            g
            + self._theta * (xc - x)
            - torch.mm(self._W, torch.mm(self._M, c)).squeeze()
        )
        r = rr[free_vars_index].unsqueeze(-1)  # n_free x 1

        invtheta = 1.0 / self._theta
        v = torch.mm(self._M, torch.mm(WtZ, r))
        N = invtheta * torch.mm(WtZ, WtZ.transpose(0, 1))
        N = (
            torch.eye(2 * self._m, dtype=self._dtype, device=self._device)
            - torch.mm(self._M, N)
        )
        v, _, _, _ = torch.linalg.lstsq(N, v, rcond=None)
        du = -invtheta * r - invtheta * invtheta * torch.mm(
            WtZ.transpose(0, 1), v
        )

        alpha_star = self._find_alpha(xc, du, free_vars_index)
        d_star = alpha_star * du

        xbar = xc.clone()
        xbar[free_vars_index] += d_star.squeeze()

        return xbar, True

    def _find_alpha(self, xc, du, free_vars_index):
        """Find step length respecting bounds (eq 5.8)."""
        du_flat = du.squeeze()
        xc_free = xc[free_vars_index]
        u_free = self._u[free_vars_index]
        l_free = self._l[free_vars_index]

        alpha_star = torch.tensor(
            1.0, dtype=self._dtype, device=self._device
        )

        pos_mask = du_flat > 0.0
        neg_mask = du_flat < 0.0

        if pos_mask.any():
            ratios_pos = (u_free[pos_mask] - xc_free[pos_mask]) / du_flat[
                pos_mask
            ]
            alpha_star = torch.min(alpha_star, ratios_pos.min())

        if neg_mask.any():
            ratios_neg = (l_free[neg_mask] - xc_free[neg_mask]) / du_flat[
                neg_mask
            ]
            alpha_star = torch.min(alpha_star, ratios_neg.min())

        return alpha_star.item()

    def _strong_wolfe(self, closure, f0, g0, p):
        """Line search satisfying strong Wolfe conditions (Alg 3.5)."""
        c1 = 1e-4
        c2 = 0.9
        alpha_max = 2.5
        alpha_im1 = 0
        alpha_i = 1
        f_im1 = f0
        dphi0 = torch.dot(g0, p)

        x0list = self._copy_params_out()
        x0 = [x.clone() for x in x0list]

        i = 0
        max_iters = 20
        while True:
            self._copy_params_in(x0)
            self._add_grad(alpha_i, p)
            f_i = float(closure())
            if (f_i > f0 + c1 * dphi0) or ((i > 1) and (f_i > f_im1)):
                alpha = self._alpha_zoom(
                    closure, x0, f0, g0, p, alpha_im1, alpha_i
                )
                break
            g_i = self._gather_flat_grad()
            dphi = torch.dot(g_i, p)
            if abs(dphi) <= -c2 * dphi0:
                alpha = alpha_i
                break
            if dphi >= 0.0:
                alpha = self._alpha_zoom(
                    closure, x0, f0, g0, p, alpha_i, alpha_im1
                )
                break
            alpha_im1 = alpha_i
            f_im1 = f_i
            alpha_i = alpha_i + 0.8 * (alpha_max - alpha_i)
            if i > max_iters:
                alpha = alpha_i
                break
            i += 1

        self._copy_params_in(x0)
        return alpha

    def _alpha_zoom(self, closure, x0, f0, g0, p, alpha_lo, alpha_hi):
        """Zoom phase of strong Wolfe line search (Alg 3.6)."""
        c1 = 1e-4
        c2 = 0.9
        max_iters = 20
        dphi0 = torch.dot(g0, p)

        for _ in range(max_iters + 1):
            alpha_i = 0.5 * (alpha_lo + alpha_hi)
            alpha = alpha_i

            self._copy_params_in(x0)
            self._add_grad(alpha_i, p)
            f_i = float(closure())
            g_i = self._gather_flat_grad()

            self._copy_params_in(x0)
            self._add_grad(alpha_lo, p)
            f_lo = float(closure())

            if (f_i > f0 + c1 * alpha_i * dphi0) or (f_i >= f_lo):
                alpha_hi = alpha_i
            else:
                dphi = torch.dot(g_i, p)
                if abs(dphi) <= -c2 * dphi0:
                    alpha = alpha_i
                    break
                if dphi * (alpha_hi - alpha_lo) >= 0.0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_i

        return alpha

    def step(self, closure):
        """Perform a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group["max_iter"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]

        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        orig_loss = closure()
        f = float(orig_loss)
        state["func_evals"] += 1

        g = self._gather_flat_grad()
        abs_grad_sum = g.abs().sum()

        if torch.isnan(abs_grad_sum) or abs_grad_sum <= tolerance_grad:
            return orig_loss

        n_iter = 0

        while (
            self._get_optimality(g) > tolerance_change
        ) and n_iter < max_iter:
            x_old = torch.cat(self._copy_params_out(), 0)
            g_old = g.clone()
            xc, c = self._get_cauchy_point(g)
            xbar, line_search_flag = self._subspace_min(g, xc, c)
            alpha = 1.0
            p = xbar - x_old
            if line_search_flag:
                alpha = self._strong_wolfe(closure, f, g, p)

            self._add_grad(alpha, p)

            f = float(closure())
            g = self._gather_flat_grad()
            y = g - g_old
            x = torch.cat(self._copy_params_out(), 0)
            s = x - x_old
            curv = torch.dot(s, y)
            n_iter += 1
            state["n_iter"] += 1

            if curv < self._eps:
                continue

            if n_iter < self._m:
                self._Y[:, n_iter] = y.squeeze()
                self._S[:, n_iter] = s.squeeze()
            else:
                self._Y[:, 0 : self._m - 1] = self._Y[:, 1 : self._m]
                self._S[:, 0 : self._m - 1] = self._S[:, 1 : self._m]
                self._Y[:, -1] = y.squeeze()
                self._S[:, -1] = s.squeeze()

            self._theta = torch.dot(y, y) / torch.dot(y, s)
            self._W[:, 0 : self._m] = self._Y
            self._W[:, self._m : 2 * self._m] = self._theta * self._S
            A = torch.mm(self._S.transpose(0, 1), self._Y)
            L = torch.tril(A, -1)
            D = -1.0 * torch.diag(torch.diag(A))
            MM = torch.zeros(
                2 * self._m,
                2 * self._m,
                dtype=self._dtype,
                device=self._device,
            )
            MM[0 : self._m, 0 : self._m] = D
            MM[0 : self._m, self._m : 2 * self._m] = L.transpose(0, 1)
            MM[self._m : 2 * self._m, 0 : self._m] = L
            MM[self._m : 2 * self._m, self._m : 2 * self._m] = (
                self._theta * torch.mm(self._S.transpose(0, 1), self._S)
            )
            self._M = torch.linalg.pinv(MM)

        return f
