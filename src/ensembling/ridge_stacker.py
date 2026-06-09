"""
Ridge meta-learner stacker — EXP-007.

Architecture (reproducing top-score notebooks):
  final = BLEND_ALPHA * ridge_stack_pred + (1 - BLEND_ALPHA) * heuristic_pred
  where BLEND_ALPHA = 0.30 (Ridge weight), 0.70 on PF/beam heuristic.

Usage:
    stacker = RidgeStacker(ridge_alpha=1.0, blend_alpha=0.30)
    oof_stack = stacker.fit_predict_oof(oof_matrix, y_true, cv=5)
    stacker.fit(oof_matrix, y_true)
    test_stack = stacker.predict_stack(test_matrix)
    final = stacker.blend(test_stack, heuristic_pred)
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Optional


class RidgeStacker:
    def __init__(
        self,
        ridge_alpha: float = 1.0,
        blend_alpha: float = 0.30,
        normalize: bool = True,
        seed: int = 42,
    ):
        self.ridge_alpha = ridge_alpha
        self.blend_alpha = blend_alpha  # weight on Ridge stack; (1 - blend_alpha) on heuristic
        self.normalize = normalize
        self.seed = seed
        self.scaler: Optional[StandardScaler] = StandardScaler() if normalize else None
        self.ridge = Ridge(alpha=ridge_alpha)
        self._fitted = False

    def fit(self, oof_matrix: np.ndarray, y_true: np.ndarray) -> "RidgeStacker":
        """
        Fit Ridge meta-learner on full OOF matrix.
        oof_matrix: (n_samples, n_base_models)
        y_true:     (n_samples,)
        """
        X = self.scaler.fit_transform(oof_matrix) if self.scaler else oof_matrix
        self.ridge.fit(X, y_true)
        self._fitted = True
        return self

    def predict_stack(self, pred_matrix: np.ndarray) -> np.ndarray:
        """Predict using fitted Ridge meta-learner. pred_matrix: (n_samples, n_base_models)."""
        assert self._fitted, "Call fit() or fit_predict_oof() first"
        X = self.scaler.transform(pred_matrix) if self.scaler else pred_matrix
        return self.ridge.predict(X)

    def blend(self, stack_pred: np.ndarray, heuristic_pred: np.ndarray) -> np.ndarray:
        """final = blend_alpha * stack + (1 - blend_alpha) * heuristic"""
        return self.blend_alpha * stack_pred + (1 - self.blend_alpha) * heuristic_pred

    def fit_predict_oof(
        self,
        oof_matrix: np.ndarray,
        y_true: np.ndarray,
        cv: int = 5,
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Produce OOF predictions of the Ridge stacker itself for ablation.
        Also fits Ridge on the full OOF matrix (for test-time inference).

        groups: optional well-level group array for GroupKFold
        Returns: oof_stack_preds (n_samples,)
        """
        X = self.scaler.fit_transform(oof_matrix) if self.scaler else oof_matrix

        oof_stack = np.zeros(len(y_true))

        if groups is not None:
            from sklearn.model_selection import GroupKFold
            kf = GroupKFold(n_splits=cv)
            splits = list(kf.split(X, y_true, groups=groups))
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
            splits = list(kf.split(X))

        for tr_idx, val_idx in splits:
            r = Ridge(alpha=self.ridge_alpha)
            X_tr = X[tr_idx] if self.scaler else oof_matrix[tr_idx]
            if self.scaler:
                sc = StandardScaler()
                X_tr = sc.fit_transform(oof_matrix[tr_idx])
                X_val = sc.transform(oof_matrix[val_idx])
            else:
                X_tr = oof_matrix[tr_idx]
                X_val = oof_matrix[val_idx]
            r.fit(X_tr, y_true[tr_idx])
            oof_stack[val_idx] = r.predict(X_val)

        # Fit final Ridge on full data for test inference
        self.ridge.fit(X, y_true)
        if self.scaler:
            self.scaler.fit(oof_matrix)
        self._fitted = True

        return oof_stack

    def get_model_weights(self) -> np.ndarray:
        """Return Ridge coefficients (one per base model)."""
        assert self._fitted, "Call fit() first"
        return self.ridge.coef_
