"""Rolling supervised baselines using the BuildSys tabular features.

One fresh model per meter type and forecast origin. The target is log1p(load),
while the weekly load lag remains in original units, as in the reference code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import time


class SupervisedForecaster:
    requires_fit = True
    lag_hours = 168

    def __init__(self, *, name: str, input_mode: str, seed: int, device: str):
        self.name = name
        self.checkpoint = name
        self.input_mode = input_mode
        self.seed = seed
        self.device = device
        # History controls the training set, not a TSFM context cap.
        self.max_context = None
        self.models = {}
        self.forecast_options = {
            "training_policy": "refit_each_origin_per_meter_type",
            "validation_days": 7,
            "target_transform": "log1p",
            "prediction_transform": "maximum(expm1(prediction), 0)",
            "target_lag_hours": 168,
            "features": ["meter_id", "target_lag_168"],
        }
        if input_mode == "arx":
            self.forecast_options["features"] += [
                "airTemperature", "weekday", "hour", "is_weekend",
                "is_holiday", "temp_rolling_mean_24h",
            ]
        if name == "mlp":
            self.forecast_options.update({
                "hidden_layers": [64, 64], "activation": "relu",
                "optimizer": "Adam", "learning_rate": 0.001,
                "max_epochs": _MLP.max_epochs, "early_stopping_patience": _MLP.patience,
                "training_batch_size": _MLP.batch_size,
                "numeric_scaling": "training_rows_only",
                "meter_encoding": "one_hot",
            })
        else:
            self.forecast_options.update({
                "n_estimators": 2000, "early_stopping_rounds": 100,
                "objective": "regression", "validation_metric": "mae",
            })

    def _features(self, values, covariates, index):
        """Meter-major rows; every target lag comes from an earlier timestamp."""
        parts = []
        if self.input_mode == "arx":
            weather = covariates.loc[index].copy()
            weather["temp_rolling_mean_24h"] = covariates.airTemperature.rolling(
                24, min_periods=1
            ).mean().loc[index]
        for meter_id in values.columns:
            part = pd.DataFrame(index=index)
            part["meter_id"] = self.meter_codes[meter_id]
            part["target_lag_168"] = values[meter_id].reindex(
                index - pd.Timedelta(hours=self.lag_hours)
            ).to_numpy()
            if self.input_mode == "arx":
                part = part.join(weather)
            parts.append(part[self.forecast_options["features"]])
        return pd.concat(parts)

    def fit_window(self, values, covariates, meter_types, train_start):
        self.meter_codes = {key: i for i, key in enumerate(values.columns)}
        self.meter_types = meter_types
        self.models = {}
        self.fit_details = []
        index = values.index[values.index >= train_start]
        # Same chronological validation split as the supplied BuildSys code.
        valid_start = index.max() - pd.Timedelta(days=7)
        for meter_type in sorted(meter_types.unique()):
            self._synchronize()
            started = time.perf_counter()
            ids = meter_types.index[meter_types == meter_type].tolist()
            x = self._features(values[ids], covariates, index)
            y = np.log1p(values.loc[index, ids].to_numpy().T.reshape(-1))
            usable = x.notna().all(axis=1).to_numpy()
            x, y = x.loc[usable], y[usable]
            train = x.index < valid_start
            if not train.any() or train.all():
                raise ValueError("History must contain training rows and a 7-day validation set")
            if self.name == "lightgbm":
                from lightgbm import LGBMRegressor, early_stopping

                model = LGBMRegressor(
                    objective="regression", metric="mae", n_estimators=2000,
                    random_state=self.seed, verbosity=-1, n_jobs=-1,
                )
                model.fit(
                    x.loc[train], y[train],
                    eval_set=[(x.loc[~train], y[~train])],
                    categorical_feature=["meter_id"], eval_metric="mae",
                    callbacks=[early_stopping(100, verbose=False)],
                )
                iterations = int(model.best_iteration_)
            else:
                model = _MLP(seed=self.seed, device=self.device,
                             meter_count=len(self.meter_codes))
                model.fit(x.loc[train], y[train], x.loc[~train], y[~train])
                iterations = model.best_epoch
            self.models[meter_type] = model
            self._synchronize()
            self.fit_details.append({
                "training_seconds": time.perf_counter() - started,
                "meter_type": meter_type, "training_rows": int(train.sum()),
                "validation_rows": int((~train).sum()),
                "training_first_timestamp": x.index[train].min().isoformat(),
                "training_last_timestamp": x.index[train].max().isoformat(),
                "validation_first_timestamp": x.index[~train].min().isoformat(),
                "validation_last_timestamp": x.index[~train].max().isoformat(),
                "best_iteration_or_epoch": iterations,
            })

    def _synchronize(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)

    def forecast(self, contexts, horizon, **metadata):
        if horizon > self.lag_hours:
            raise ValueError("Weekly-lag baselines support horizon <= 168 h")
        ids = metadata["series_ids"]
        values = pd.DataFrame(dict(zip(ids, contexts)), index=metadata["context_index"])
        covariates = None
        if self.input_mode == "arx":
            covariates = pd.concat([
                metadata["context_covariates"], metadata["future_covariates"],
            ])
        index = metadata["future_index"][:horizon]
        prediction = np.empty((len(ids), horizon), dtype=np.float32)
        for meter_type, model in self.models.items():
            positions = [i for i, key in enumerate(ids) if self.meter_types[key] == meter_type]
            if not positions:
                continue
            group_ids = [ids[i] for i in positions]
            x = self._features(values[group_ids], covariates, index)
            point = np.maximum(np.expm1(model.predict(x)), 0)
            prediction[positions] = point.reshape(len(group_ids), horizon)
        return prediction


class _MLP:
    """Small tabular network: standardized features + one-hot meter, 64/64 ReLU."""
    max_epochs = 100
    patience = 10
    batch_size = 1024

    def __init__(self, *, seed, device, meter_count):
        self.seed, self.device, self.meter_count = seed, device, meter_count

    def _encode(self, x):
        numeric = x.drop(columns="meter_id").to_numpy(dtype=np.float32)
        numeric = (numeric - self.mean) / self.scale
        meter = np.eye(self.meter_count, dtype=np.float32)[x.meter_id.to_numpy(dtype=int)]
        return np.concatenate([numeric, meter], axis=1)

    def fit(self, x, y, valid_x, valid_y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        numeric = x.drop(columns="meter_id").to_numpy(dtype=np.float32)
        self.mean = numeric.mean(axis=0)
        self.scale = np.maximum(numeric.std(axis=0), 1e-6)
        self.y_mean, self.y_scale = float(y.mean()), max(float(y.std()), 1e-6)
        xt = torch.as_tensor(self._encode(x), device=self.device)
        yt = torch.as_tensor((y - self.y_mean) / self.y_scale, dtype=torch.float32, device=self.device)
        xv = torch.as_tensor(self._encode(valid_x), device=self.device)
        yv = torch.as_tensor((valid_y - self.y_mean) / self.y_scale, dtype=torch.float32, device=self.device)
        self.network = nn.Sequential(
            nn.Linear(xt.shape[1], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        ).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        best_loss, stale = float("inf"), 0
        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            order = torch.randperm(len(xt), device=self.device)
            for batch in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.network(xt[batch]).flatten(), yt[batch])
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                validation_loss = (self.network(xv).flatten() - yv).abs().mean().item()
            if validation_loss < best_loss:
                best_loss, stale, self.best_epoch = validation_loss, 0, epoch
                best_weights = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
            if stale >= self.patience:
                break
        self.network.load_state_dict(best_weights)

    def predict(self, x):
        import torch

        with torch.no_grad():
            xt = torch.as_tensor(self._encode(x), device=self.device)
            return self.network(xt).flatten().cpu().numpy() * self.y_scale + self.y_mean
