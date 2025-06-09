import random
import numpy as np
random.seed(42)
np.random.seed(42)


class SVM:
    def __init__(self, C, gamma, tol, eps, n_iter):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.eps = eps
        self.n_iter = n_iter

    def _rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = np.where(y <= 0, -1, 1)
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.error_cache = -self.y.copy()  # initial error h(x)=0

        self.K = self._rbf_kernel(X, X)

        num_changed = 0
        examine_all = True
        iter_count = 0

        while (num_changed > 0 or examine_all) and iter_count < self.n_iter:
            num_changed = 0
            if examine_all:
                # Loop over all training examples
                for i in range(n_samples):
                    num_changed += self.examineExample(i)
            else:
                # Loop over examples where alpha != 0 and != C
                idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
                for i in idx:
                    num_changed += self.examineExample(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iter_count += 1

        # Save support vectors
        sv_idx = self.alpha > 1e-5
        self.X_sv = self.X[sv_idx]
        self.y_sv = self.y[sv_idx]
        self.alpha_sv = self.alpha[sv_idx]

    def examineExample(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        E2 = self.error_cache[i2]
        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            # 1st heuristic: try to find best i1
            non_bound_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            if len(non_bound_idx) > 1:
                i1 = self._second_choice_heuristic(E2)
                if self.takeStep(i1, i2):
                    return 1

            # 2nd heuristic: loop over non-bound alpha
            np.random.shuffle(non_bound_idx)
            for i1 in non_bound_idx:
                if self.takeStep(i1, i2):
                    return 1

            # 3rd heuristic: loop over all alpha
            all_idx = np.arange(self.X.shape[0])
            np.random.shuffle(all_idx)
            for i1 in all_idx:
                if self.takeStep(i1, i2):
                    return 1

        return 0

    def takeStep(self, i1, i2):
        if i1 == i2:
            return 0

        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.error_cache[i1]
        E2 = self.error_cache[i2]
        s = y1 * y2

        # Compute L and H
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return 0

        # Compute eta
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = alpha2 + y2 * (E1 - E2) / eta
            a2_new = np.clip(a2_new, L, H)
        else:
            # eta <= 0 → handle with objective function heuristic
            Lobj = self._objective_function(i1, i2, alpha1, alpha2, L)
            Hobj = self._objective_function(i1, i2, alpha1, alpha2, H)
            if Lobj < Hobj - self.eps:
                a2_new = L
            elif Lobj > Hobj + self.eps:
                a2_new = H
            else:
                a2_new = alpha2

        if np.abs(a2_new - alpha2) < self.eps * (a2_new + alpha2 + self.eps):
            return 0

        # Update alpha1 accordingly
        a1_new = alpha1 + s * (alpha2 - a2_new)

        # Update bias b
        b1 = self.b - E1 - y1 * (a1_new - alpha1) * k11 - y2 * (a2_new - alpha2) * k12
        b2 = self.b - E2 - y1 * (a1_new - alpha1) * k12 - y2 * (a2_new - alpha2) * k22

        if 0 < a1_new < self.C:
            self.b = b1
        elif 0 < a2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # Update alpha array
        self.alpha[i1] = a1_new
        self.alpha[i2] = a2_new

        # Update error cache
        self.error_cache[i1] = self._compute_output(i1) - y1
        self.error_cache[i2] = self._compute_output(i2) - y2

        return 1

    def _compute_output(self, i):
        # h(x_i) = sum_j alpha_j * y_j * K(x_j, x_i) + b
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b

    def _objective_function(self, i1, i2, alpha1_old, alpha2_old, alpha2_new):
        s = self.y[i1] * self.y[i2]
        a1_new = alpha1_old + s * (alpha2_old - alpha2_new)

        term1 = a1_new + alpha2_new
        term2 = 0.5 * self.K[i1, i1] * a1_new ** 2
        term3 = 0.5 * self.K[i2, i2] * alpha2_new ** 2
        term4 = s * self.K[i1, i2] * a1_new * alpha2_new

        return term1 - (term2 + term3 + term4)

    def _second_choice_heuristic(self, E2):
        # Select i1 that maximizes |E1 - E2|
        E_diff = np.abs(self.error_cache - E2)
        return np.argmax(E_diff)

    def project(self, X):
        K = self._rbf_kernel(X, self.X_sv)
        return np.sum(self.alpha_sv * self.y_sv * K, axis=1) + self.b

    def predict(self, X):
        return np.sign(self.project(X))
