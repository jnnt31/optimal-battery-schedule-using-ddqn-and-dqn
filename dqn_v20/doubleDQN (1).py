#!/usr/bin/env python
# coding: utf-8

# DQN for BESS — Local Version
#
# This is a local execution version with:
# - 70% train / 20% test / 10% validation split by day
# - 8000 total training episodes across the training set
# - Optional synthetic/randomized training-day augmentation
# - Validation-based model selection
#

import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU.")



import os
import copy
import random
import argparse
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    input_files: List[str] = None
    output_dir: str = "dqn_v20"

    # Battery specification
    battery_capacity_mwh: float = 20.0
    soc_init: float = 0.15
    soc_min: float = 0.05
    soc_max: float = 0.95
    max_charge_mw: float = 7.5
    max_discharge_mw: float = 7.5
    action_step_mw: float = 1.25

    # Grid specification
    max_import_mw: float = 23.0
    max_export_mw: float = 7.0

    # Efficiency
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    # Time step: 5 minutes
    dt_hours: float = 5.0 / 60.0

    # Battery degradation model (REDUCED 10x for proper arbitrage learning)
    battery_cost_usd_per_kwh: float = 135.0
    alpha_deg: float = 694.0
    beta_deg: float = 0.795

    # Training
    episodes: int = 8000              # TOTAL training episodes, not per day
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_size: int = 100000
    min_replay_to_train: int = 3000
    target_update_every: int = 1000
    train_every: int = 1
    hidden_size: int = 256

    # Exploration
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9994      # Reach minimum at ~6500 episodes
    
    # Learning rate scheduling
    lr_decay_gamma: float = 0.9997    # Exponential decay: multiply by 0.9995 each episode

    # Lookahead for price awareness
    lookahead_steps: int = 12

    # Penalties
    invalid_action_penalty: float = 150.0
    simultaneous_import_export_penalty: float = 100.0
    
    # Arbitrage reward shaping (INCREASED FOR BETTER LEARNING SIGNAL)
    arbitrage_bonus_magnitude: float = 500.0  # Bonus for charging at low / discharging at high (5x increase)
    soc_health_bonus: float = 100.0  # Bonus for maintaining healthy 30-70% SOC (10x increase)

    # Early stopping
    patience: int = 13  # Stop if validation doesn't improve for N validation checks (every 100 episodes)

    # Split ratios
    # Data are split once by day. Training uses train_days,
    # model selection uses val_days, and final reporting uses test_days.
    train_ratio: float = 0.70
    test_ratio: float = 0.20
    val_ratio: float = 0.10

    # Synthetic data / randomness (training only)
    use_augmentation: bool = True
    aug_prob: float = 0.50
    solar_noise_std_frac: float = 0.05    # 5% std of solar max
    load_noise_std_frac: float = 0.03     # 3% std of load max
    price_noise_std_frac: float = 0.02    # 2% std of price max
    solar_scale_range: Tuple[float, float] = (0.90, 1.10)
    load_scale_range: Tuple[float, float] = (0.95, 1.05)
    price_scale_range: Tuple[float, float] = (0.95, 1.05)

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.input_files is None:
            self.input_files = []

    @property
    def battery_cost_total_usd(self) -> float:
        return self.battery_capacity_mwh * 1000.0 * self.battery_cost_usd_per_kwh

    @property
    def actions(self) -> np.ndarray:
        return np.arange(
            -self.max_discharge_mw,
            self.max_charge_mw + 1e-9,
            self.action_step_mw,
            dtype=np.float32,
        )

CFG = Config()

# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def month_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    return base.replace(".xlsx", "").replace(" IN MW RANGE", "").strip()


def normalize_value(x: float, xmin: float, xmax: float) -> float:
    if xmax - xmin < 1e-12:
        return 0.0
    return (x - xmin) / (xmax - xmin)


def split_days(days: List[Dict], train_ratio: float, test_ratio: float, val_ratio: float, seed: int):
    if abs((train_ratio + test_ratio + val_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + test_ratio + val_ratio must sum to 1.0")

    n = len(days)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test

    if n >= 3:
        n_train = max(1, n_train)
        n_test = max(1, n_test)
        n_val = n - n_train - n_test
        if n_val < 1:
            n_val = 1
            if n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_test > 1:
                n_test -= 1

    train_ids = idx[:n_train]
    test_ids = idx[n_train:n_train + n_test]
    val_ids = idx[n_train + n_test:]

    train_days = [days[i] for i in train_ids]
    test_days = [days[i] for i in test_ids]
    val_days = [days[i] for i in val_ids]

    return train_days, test_days, val_days


def clip_nonnegative(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, None)


def augment_day(day: Dict, cfg: Config) -> Dict:
    aug = copy.deepcopy(day)
    rng = np.random.default_rng()

    solar = aug["pv_mw"].astype(np.float32).copy()
    load = aug["load_mw"].astype(np.float32).copy()
    buy_price = aug["buy_price"].astype(np.float32).copy()
    sell_price = aug["sell_price"].astype(np.float32).copy()

    solar_scale = rng.uniform(*cfg.solar_scale_range)
    load_scale = rng.uniform(*cfg.load_scale_range)
    price_scale = rng.uniform(*cfg.price_scale_range)

    solar_noise = rng.normal(0.0, cfg.solar_noise_std_frac * max(1e-6, solar.max()), size=solar.shape)
    load_noise = rng.normal(0.0, cfg.load_noise_std_frac * max(1e-6, load.max()), size=load.shape)
    buy_noise = rng.normal(0.0, cfg.price_noise_std_frac * max(1e-6, buy_price.max()), size=buy_price.shape)
    sell_noise = rng.normal(0.0, cfg.price_noise_std_frac * max(1e-6, sell_price.max()), size=sell_price.shape)

    aug["pv_mw"] = clip_nonnegative((solar * solar_scale + solar_noise).astype(np.float32))
    aug["load_mw"] = clip_nonnegative((load * load_scale + load_noise).astype(np.float32))
    aug["buy_price"] = clip_nonnegative((buy_price * price_scale + buy_noise).astype(np.float32))
    aug["sell_price"] = clip_nonnegative((sell_price * price_scale + sell_noise).astype(np.float32))
    aug["day_name"] = f'{day["day_name"]}_AUG'
    return aug


def maybe_augment_day(day: Dict, cfg: Config) -> Dict:
    if cfg.use_augmentation and random.random() < cfg.aug_prob:
        return augment_day(day, cfg)
    return day


# ============================================================
# DATA LOADING
# ============================================================


def load_month_file(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing input file: {file_path}")

    df = pd.read_excel(file_path)

    required_cols = [
        "TIME (MINUTE)",
        "LOAD(MW)",
        "BUYING PRICE (USD/MWhr)",
        "SELLING PRICE (USD/MWhr)",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{file_path} is missing required column: {col}")

    day_cols = [c for c in df.columns if c.startswith("DAY ") and c.endswith(" POWER (MW)")]
    if not day_cols:
        raise ValueError(f"{file_path} has no day-wise PV columns like 'DAY 1 POWER (MW)'")

    return {
        "file_path": file_path,
        "month_name": month_name_from_path(file_path),
        "time_min": df["TIME (MINUTE)"].to_numpy(dtype=np.float32),
        "load_mw": df["LOAD(MW)"].to_numpy(dtype=np.float32),
        "buy_price": df["BUYING PRICE (USD/MWhr)"].to_numpy(dtype=np.float32),
        "sell_price": df["SELLING PRICE (USD/MWhr)"].to_numpy(dtype=np.float32),
        "day_columns": day_cols,
        "pv_days": {col: df[col].to_numpy(dtype=np.float32) for col in day_cols},
    }


def load_all_data(config: Config) -> List[Dict]:
    return [load_month_file(fp) for fp in config.input_files]


def build_all_days(months: List[Dict]) -> List[Dict]:
    all_days: List[Dict] = []
    for month in months:
        for day_col in month["day_columns"]:
            all_days.append(
                {
                    "month_name": month["month_name"],
                    "day_name": day_col.replace(" POWER (MW)", ""),
                    "time_min": month["time_min"],
                    "load_mw": month["load_mw"],
                    "buy_price": month["buy_price"],
                    "sell_price": month["sell_price"],
                    "pv_mw": month["pv_days"][day_col],
                }
            )
    return all_days


def build_stats(days: List[Dict]) -> Dict:
    all_time = np.concatenate([d["time_min"] for d in days])
    all_load = np.concatenate([d["load_mw"] for d in days])
    all_solar = np.concatenate([d["pv_mw"] for d in days])
    all_buy = np.concatenate([d["buy_price"] for d in days])
    all_sell = np.concatenate([d["sell_price"] for d in days])
    all_spread = all_sell - all_buy

    return {
        "time_min": float(all_time.min()),
        "time_max": float(all_time.max()),
        "load_min": float(all_load.min()),
        "load_max": float(all_load.max()),
        "solar_min": float(all_solar.min()),
        "solar_max": float(all_solar.max()),
        "buy_min": float(all_buy.min()),
        "buy_max": float(all_buy.max()),
        "buy_median": float(np.median(all_buy)),
        "sell_min": float(all_sell.min()),
        "sell_max": float(all_sell.max()),
        "sell_median": float(np.median(all_sell)),
        "spread_min": float(all_spread.min()),
        "spread_max": float(all_spread.max()),
    }


# ============================================================
# ENVIRONMENT
# ============================================================


class BatteryEnv:
    def __init__(self, config: Config, day_data: Dict, stats: Dict):
        self.cfg = config
        self.day = day_data
        self.stats = stats
        self.n_steps = len(day_data["time_min"])
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.soc = self.cfg.soc_init
        self.done = False
        self.records: List[Dict] = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        load = float(self.day["load_mw"][self.t])
        solar = float(self.day["pv_mw"][self.t])
        buy_p = float(self.day["buy_price"][self.t])
        sell_p = float(self.day["sell_price"][self.t])
        time_min = float(self.day["time_min"][self.t])

        state_components = [
            normalize_value(time_min, self.stats["time_min"], self.stats["time_max"]),
            normalize_value(load, self.stats["load_min"], self.stats["load_max"]),
            normalize_value(solar, self.stats["solar_min"], self.stats["solar_max"]),
            normalize_value(self.soc, self.cfg.soc_min, self.cfg.soc_max),
            normalize_value(buy_p, self.stats["buy_min"], self.stats["buy_max"]),
            normalize_value(sell_p, self.stats["sell_min"], self.stats["sell_max"]),
        ]

        for step in range(1, self.cfg.lookahead_steps + 1):
            future_idx = min(self.t + step, self.n_steps - 1)
            future_buy = float(self.day["buy_price"][future_idx])
            future_sell = float(self.day["sell_price"][future_idx])
            state_components.append(normalize_value(future_buy, self.stats["buy_min"], self.stats["buy_max"]))
            state_components.append(normalize_value(future_sell, self.stats["sell_min"], self.stats["sell_max"]))

        price_spread = sell_p - buy_p
        spread_norm = normalize_value(price_spread, self.stats["spread_min"], self.stats["spread_max"])

        future_indices = [min(self.t + step, self.n_steps - 1) for step in range(1, min(6, self.cfg.lookahead_steps + 1))]
        future_buy_prices = [float(self.day["buy_price"][idx]) for idx in future_indices]
        future_sell_prices = [float(self.day["sell_price"][idx]) for idx in future_indices]
        future_buy_avg = np.mean(future_buy_prices) if future_buy_prices else buy_p
        future_sell_avg = np.mean(future_sell_prices) if future_sell_prices else sell_p
        momentum = (future_buy_avg - buy_p) + (future_sell_avg - sell_p)
        momentum_norm = normalize_value(momentum, -100, 100)
        buy_position = normalize_value(buy_p, self.stats["buy_min"], self.stats["buy_max"])
        sell_position = normalize_value(sell_p, self.stats["sell_min"], self.stats["sell_max"])

        state_components.extend([spread_norm, momentum_norm, buy_position, sell_position])
        return np.array(state_components, dtype=np.float32)

    def _life_cycles(self, dod: float) -> float:
        dod = max(dod, 1e-6)
        return self.cfg.alpha_deg * (dod ** (-self.cfg.beta_deg))

    def _degradation_cost(self, dod1: float, dod2: float, abs_batt_power_mw: float) -> float:
        l1 = self._life_cycles(dod1)
        l2 = self._life_cycles(dod2)
        c_dod = self.cfg.battery_cost_total_usd * abs((1.0 / l2) - (1.0 / l1))
        return c_dod * abs_batt_power_mw * self.cfg.dt_hours

    def _is_action_feasible(self, a_mw: float, load_mw: float, solar_mw: float) -> bool:
        if a_mw > 0:
            delta_soc = (self.cfg.eta_ch * a_mw * self.cfg.dt_hours) / self.cfg.battery_capacity_mwh
            soc_next = self.soc + delta_soc
        elif a_mw < 0:
            delta_soc = (abs(a_mw) * self.cfg.dt_hours) / (self.cfg.eta_dis * self.cfg.battery_capacity_mwh)
            soc_next = self.soc - delta_soc
        else:
            soc_next = self.soc

        if soc_next < self.cfg.soc_min - 1e-9 or soc_next > self.cfg.soc_max + 1e-9:
            return False

        net = solar_mw - load_mw - a_mw
        export_mw = max(net, 0.0)
        import_mw = max(-net, 0.0)

        if import_mw > self.cfg.max_import_mw + 1e-9:
            return False
        if export_mw > self.cfg.max_export_mw + 1e-9:
            return False
        return True

    def valid_action_mask(self) -> np.ndarray:
        load = float(self.day["load_mw"][self.t])
        solar = float(self.day["pv_mw"][self.t])
        mask = np.zeros(len(self.cfg.actions), dtype=bool)
        for i, a in enumerate(self.cfg.actions):
            mask[i] = self._is_action_feasible(float(a), load, solar)
        return mask

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode is already finished.")

        a_mw = float(self.cfg.actions[action_idx])
        load_mw = float(self.day["load_mw"][self.t])
        solar_mw = float(self.day["pv_mw"][self.t])
        buy_price = float(self.day["buy_price"][self.t])
        sell_price = float(self.day["sell_price"][self.t])
        time_min = float(self.day["time_min"][self.t])

        penalty = 0.0
        if not self._is_action_feasible(a_mw, load_mw, solar_mw):
            penalty -= self.cfg.invalid_action_penalty
            a_mw = 0.0

        old_soc = self.soc
        if a_mw > 0:
            delta_soc = (self.cfg.eta_ch * a_mw * self.cfg.dt_hours) / self.cfg.battery_capacity_mwh
            new_soc = old_soc + delta_soc
        elif a_mw < 0:
            delta_soc = (abs(a_mw) * self.cfg.dt_hours) / (self.cfg.eta_dis * self.cfg.battery_capacity_mwh)
            new_soc = old_soc - delta_soc
        else:
            new_soc = old_soc

        new_soc = float(np.clip(new_soc, self.cfg.soc_min, self.cfg.soc_max))

        net = solar_mw - load_mw - a_mw
        export_mw = max(net, 0.0)
        import_mw = max(-net, 0.0)

        if import_mw > 0 and export_mw > 0:
            penalty -= self.cfg.simultaneous_import_export_penalty

        import_mw = min(import_mw, self.cfg.max_import_mw)
        export_mw = min(export_mw, self.cfg.max_export_mw)

        import_mwh = import_mw * self.cfg.dt_hours
        export_mwh = export_mw * self.cfg.dt_hours

        buy_cost = import_mwh * buy_price
        sell_revenue = export_mwh * sell_price

        old_dod = 1.0 - old_soc
        new_dod = 1.0 - new_soc
        deg_cost = self._degradation_cost(old_dod, new_dod, abs(a_mw))

        arbitrage_bonus = 0.0
        if a_mw > 0:
            if buy_price < self.stats["buy_median"]:
                arbitrage_bonus = self.cfg.arbitrage_bonus_magnitude * (1.0 - buy_price / (self.stats["buy_max"] + 1e-6))
        elif a_mw < 0:
            if sell_price > self.stats["sell_median"]:
                arbitrage_bonus = self.cfg.arbitrage_bonus_magnitude * (sell_price / (self.stats["sell_max"] + 1e-6))

        soc_bonus = 0.0
        if 0.30 < new_soc < 0.70:
            soc_bonus = self.cfg.soc_health_bonus

        reward = sell_revenue - buy_cost - deg_cost + arbitrage_bonus + soc_bonus + penalty

        self.soc = new_soc
        self.records.append(
            {
                "TIME (MINUTE)": time_min,
                "LOAD(MW)": load_mw,
                "SOLAR(MW)": solar_mw,
                "BUYING PRICE (USD/MWhr)": buy_price,
                "SELLING PRICE (USD/MWhr)": sell_price,
                "BATTERY POWER (MW)": a_mw,
                "SOC": self.soc,
                "DOD": 1.0 - self.soc,
                "GRID IMPORT (MW)": import_mw,
                "GRID EXPORT (MW)": export_mw,
                "GRID IMPORT (MWh)": import_mwh,
                "GRID EXPORT (MWh)": export_mwh,
                "BUY COST (USD)": buy_cost,
                "SELL REVENUE (USD)": sell_revenue,
                "DEGRADATION COST (USD)": deg_cost,
                "STEP REWARD": reward,
            }
        )

        self.t += 1
        if self.t >= self.n_steps:
            self.done = True
            state_dim = 6 + 2 * self.cfg.lookahead_steps + 4
            next_state = np.zeros(state_dim, dtype=np.float32)
        else:
            next_state = self._get_state()

        return next_state, reward, self.done, {}

    def get_episode_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def get_episode_summary(self, split_name: str, month_name: str, day_name: str) -> Dict:
        df = self.get_episode_dataframe()
        return {
            "SPLIT": split_name,
            "MONTH": month_name,
            "DAY": day_name,
            "ENERGY BOUGHT (MWh)": df["GRID IMPORT (MWh)"].sum(),
            "ENERGY SOLD (MWh)": df["GRID EXPORT (MWh)"].sum(),
            "BUYING COST (USD)": df["BUY COST (USD)"].sum(),
            "SELLING REVENUE (USD)": df["SELL REVENUE (USD)"].sum(),
            "DEGRADATION COST (USD)": df["DEGRADATION COST (USD)"].sum(),
            "NET COST (USD)": df["BUY COST (USD)"].sum() + df["DEGRADATION COST (USD)"].sum() - df["SELL REVENUE (USD)"].sum(),
            "FINAL SOC": float(df["SOC"].iloc[-1]) if len(df) else self.cfg.soc_init,
        }


# ============================================================
# REPLAY BUFFER AND NETWORKS
# ============================================================

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, config: Config, state_dim: int, action_dim: int):
        self.cfg = config
        self.device = torch.device(config.device)

        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=config.lr_decay_gamma)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(config.replay_size)
        self.epsilon = config.epsilon_start

        self.reward_mean = 0.0
        self.reward_sq_mean = 0.0
        self.reward_count = 0
        self.reward_std = 1.0

    def update_reward_stats(self, reward: float) -> None:
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_sq_mean += (delta * delta2)

        if self.reward_count > 1:
            self.reward_std = max(1.0, np.sqrt(self.reward_sq_mean / self.reward_count))

    def normalize_reward(self, reward: float) -> float:
        if self.reward_std < 1e-6:
            return reward
        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return float(np.clip(normalized, -5.0, 5.0))

    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, greedy: bool = False) -> int:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return int(np.argmin(np.abs(self.cfg.actions - 0.0)))

        if (not greedy) and (random.random() < self.epsilon):
            return int(random.choice(valid_indices))

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).cpu().numpy().flatten()

        masked_q = np.full_like(q_values, -1e18)
        masked_q[valid_indices] = q_values[valid_indices]
        return int(np.argmax(masked_q))

    def update(self):
        if len(self.replay) < max(self.cfg.min_replay_to_train, self.cfg.batch_size):
            return None

        batch = self.replay.sample(self.cfg.batch_size)

        state = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_current = self.policy_net(state).gather(1, action)

        with torch.no_grad():
            q_next = self.target_net(next_state).max(dim=1, keepdim=True).values
            q_target = reward + (1.0 - done) * self.cfg.gamma * q_next

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


# ============================================================
# TRAINING AND EVALUATION
# ============================================================


def run_greedy_rollout(config: Config, agent: DQNAgent, day_data: Dict, stats: Dict, split_name: str):
    env = BatteryEnv(config, day_data, stats)
    state = env.reset()

    total_reward = 0.0
    while True:
        valid_mask = env.valid_action_mask()
        action_idx = agent.select_action(state, valid_mask, greedy=True)
        next_state, reward, done, _ = env.step(action_idx)
        state = next_state
        total_reward += reward
        if done:
            break

    summary = env.get_episode_summary(split_name, day_data["month_name"], day_data["day_name"])
    summary["TOTAL REWARD"] = total_reward
    return summary, env.get_episode_dataframe()


def evaluate_split(config: Config, agent: DQNAgent, days: List[Dict], stats: Dict, split_name: str):
    summaries = []
    details = []

    for day in days:
        summary, detail = run_greedy_rollout(config, agent, day, stats, split_name)
        detail.insert(0, "SPLIT", split_name)
        detail.insert(1, "MONTH", day["month_name"])
        detail.insert(2, "DAY", day["day_name"])
        summaries.append(summary)
        details.append(detail)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary_df, detail_df


def train_agent(config: Config, train_days: List[Dict], val_days: List[Dict], stats: Dict):
    state_dim = 6 + 2 * config.lookahead_steps + 4
    agent = DQNAgent(config, state_dim=state_dim, action_dim=len(config.actions))
    history: List[Dict] = []
    global_step = 0

    best_val_reward = -np.inf
    best_state_dict = None
    patience_counter = 0

    runs_dir = os.path.join(config.output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(runs_dir)

    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")
    print(f"Total training episodes: {config.episodes}")
    print(f"TensorBoard logs saved to: {runs_dir}")

    for episode in range(1, config.episodes + 1):
        base_day = random.choice(train_days)
        day = maybe_augment_day(base_day, config)

        env = BatteryEnv(config, day, stats)
        state = env.reset()

        total_reward = 0.0
        losses = []
        did_update_this_episode = False

        while True:
            valid_mask = env.valid_action_mask()
            action_idx = agent.select_action(state, valid_mask, greedy=False)
            next_state, reward, done, _ = env.step(action_idx)

            agent.update_reward_stats(reward)
            normalized_reward = agent.normalize_reward(reward)

            agent.replay.push(state, action_idx, normalized_reward, next_state, done)
            state = next_state
            total_reward += reward
            global_step += 1

            if global_step % config.train_every == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                    did_update_this_episode = True

            if global_step % config.target_update_every == 0:
                agent.update_target()

            if done:
                break

        agent.decay_epsilon()
        if did_update_this_episode:
            agent.lr_scheduler.step()

        record = {
            "EPISODE": episode,
            "TRAIN_SOURCE_MONTH": base_day["month_name"],
            "TRAIN_SOURCE_DAY": base_day["day_name"],
            "USED_AUGMENTATION": int(day["day_name"].endswith("_AUG")),
            "TRAIN_TOTAL_REWARD": total_reward,
            "AVG_LOSS": float(np.mean(losses)) if losses else np.nan,
            "EPSILON": agent.epsilon,
        }

        writer.add_scalar("Loss/train", record["AVG_LOSS"], episode)
        writer.add_scalar("Reward/train", total_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        current_lr = agent.optimizer.param_groups[0]["lr"]
        writer.add_scalar("LearningRate", current_lr, episode)

        if episode % 100 == 0 or episode == 1:
            val_summary_df, _ = evaluate_split(config, agent, val_days, stats, "validation")
            val_mean_reward = float(val_summary_df["TOTAL REWARD"].mean()) if len(val_summary_df) else np.nan
            record["VAL_MEAN_REWARD"] = val_mean_reward

            if np.isfinite(val_mean_reward) and val_mean_reward > best_val_reward:
                best_val_reward = val_mean_reward
                best_state_dict = copy.deepcopy(agent.policy_net.state_dict())
                patience_counter = 0
                best_model_path = os.path.join(config.output_dir, "dqn_bess_model_best.pt")
                torch.save(best_state_dict, best_model_path)
                print(f"  ✓ Saved best model checkpoint (reward: {val_mean_reward:.2f})")
            else:
                patience_counter += 1

            writer.add_scalar("Reward/validation", val_mean_reward, episode)
            print(
                f"Episode {episode}/{config.episodes} | "
                f"Train Reward: {total_reward:.2f} | "
                f"Val Mean Reward: {val_mean_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Patience: {patience_counter}/{config.patience}"
            )

            if patience_counter >= config.patience:
                print(f"\nEarly stopping triggered at episode {episode}. No improvement for {config.patience} validation checks.")
                break

        history.append(record)

    if best_state_dict is not None:
        agent.policy_net.load_state_dict(best_state_dict)
        agent.update_target()
        print(f"✓ Loaded best validation model (mean validation reward = {best_val_reward:.2f})")
    else:
        print("⚠ No best model found, using final trained model")

    print(f"\nTensorBoard training complete. View with: tensorboard --logdir={runs_dir}")
    return agent, pd.DataFrame(history), writer


def aggregate_to_hourly(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    df = detail_df.copy()
    df["HOUR"] = (df["TIME (MINUTE)"] / 60.0).astype(int)

    agg_dict = {
        "TIME (MINUTE)": "first",
        "LOAD(MW)": "mean",
        "SOLAR(MW)": "mean",
        "BUYING PRICE (USD/MWhr)": "mean",
        "SELLING PRICE (USD/MWhr)": "mean",
        "BATTERY POWER (MW)": "mean",
        "SOC": "last",
        "DOD": "last",
        "GRID IMPORT (MW)": "mean",
        "GRID EXPORT (MW)": "mean",
        "GRID IMPORT (MWh)": "sum",
        "GRID EXPORT (MWh)": "sum",
        "BUY COST (USD)": "sum",
        "SELL REVENUE (USD)": "sum",
        "DEGRADATION COST (USD)": "sum",
        "STEP REWARD": "sum",
    }

    return df.groupby(["SPLIT", "MONTH", "DAY", "HOUR"], as_index=False).agg(agg_dict)


def save_outputs(config: Config, history_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, train_detail: pd.DataFrame, val_detail: pd.DataFrame, test_detail: pd.DataFrame, agent: DQNAgent) -> None:
    safe_mkdir(config.output_dir)

    history_path = os.path.join(config.output_dir, "training_history.xlsx")
    history_df.to_excel(history_path, index=False)

    history_csv_path = os.path.join(config.output_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)

    summary_path = os.path.join(config.output_dir, "split_summaries.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="train_summary", index=False)
        val_df.to_excel(writer, sheet_name="validation_summary", index=False)
        test_df.to_excel(writer, sheet_name="test_summary", index=False)

    detail_path = os.path.join(config.output_dir, "split_time_series.xlsx")
    with pd.ExcelWriter(detail_path, engine="openpyxl") as writer:
        train_detail.to_excel(writer, sheet_name="train_time_series", index=False)
        val_detail.to_excel(writer, sheet_name="validation_time_series", index=False)
        test_detail.to_excel(writer, sheet_name="test_time_series", index=False)

    train_hourly = aggregate_to_hourly(train_detail)
    val_hourly = aggregate_to_hourly(val_detail)
    test_hourly = aggregate_to_hourly(test_detail)

    hourly_path = os.path.join(config.output_dir, "split_time_series_hourly.xlsx")
    with pd.ExcelWriter(hourly_path, engine="openpyxl") as writer:
        train_hourly.to_excel(writer, sheet_name="train_hourly", index=False)
        val_hourly.to_excel(writer, sheet_name="validation_hourly", index=False)
        test_hourly.to_excel(writer, sheet_name="test_hourly", index=False)

    model_path = os.path.join(config.output_dir, "dqn_bess_model.pt")
    torch.save(agent.policy_net.state_dict(), model_path)

    print(f"Saved: {history_path}")
    print(f"Saved: {history_csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {hourly_path}")
    print(f"Saved: {model_path}")


# ============================================================
# MAIN
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a DQN BESS controller.")
    parser.add_argument("--input-files", nargs="+", required=True, help="Excel files containing the input data.")
    parser.add_argument("--output-dir", default=CFG.output_dir, help="Directory for generated results and model files.")
    return parser.parse_args()


def main() -> None:
    set_seed(CFG.seed)
    safe_mkdir(CFG.output_dir)

    if not CFG.input_files:
        raise ValueError("No Excel files selected.")

    print("Loading Excel files...")
    months = load_all_data(CFG)
    all_days = build_all_days(months)
    print(f"Total real days found: {len(all_days)}")

    train_days, test_days, val_days = split_days(all_days, CFG.train_ratio, CFG.test_ratio, CFG.val_ratio, CFG.seed)
    print(f"Train days: {len(train_days)}")
    print(f"Test days: {len(test_days)}")
    print(f"Validation days: {len(val_days)}")

    stats = build_stats(train_days)

    print("Training DQN...")
    agent, history_df, writer = train_agent(CFG, train_days, val_days, stats)

    print("Evaluating trained agent on train / validation / test...")
    train_summary_df, train_detail_df = evaluate_split(CFG, agent, train_days, stats, "train")
    val_summary_df, val_detail_df = evaluate_split(CFG, agent, val_days, stats, "validation")
    test_summary_df, test_detail_df = evaluate_split(CFG, agent, test_days, stats, "test")

    if len(train_summary_df):
        writer.add_scalar("Final_Splits/Train_Net_Cost", train_summary_df["NET COST (USD)"].mean(), 0)
    if len(val_summary_df):
        writer.add_scalar("Final_Splits/Validation_Net_Cost", val_summary_df["NET COST (USD)"].mean(), 0)
    if len(test_summary_df):
        writer.add_scalar("Final_Splits/Test_Net_Cost", test_summary_df["NET COST (USD)"].mean(), 0)

    writer.close()

    print("\nMean net cost by split:")
    if len(train_summary_df):
        print("Train      :", train_summary_df["NET COST (USD)"].mean())
    if len(val_summary_df):
        print("Validation :", val_summary_df["NET COST (USD)"].mean())
    if len(test_summary_df):
        print("Test       :", test_summary_df["NET COST (USD)"].mean())

    print("Saving outputs...")
    save_outputs(
        CFG,
        history_df,
        train_summary_df,
        val_summary_df,
        test_summary_df,
        train_detail_df,
        val_detail_df,
        test_detail_df,
        agent,
    )

    print("Finished successfully.")


if __name__ == "__main__":
    args = parse_args()
    CFG.input_files = args.input_files
    CFG.output_dir = args.output_dir
    print("\n" + "=" * 60)
    print("DQN for Battery Energy Storage System (BESS)")
    print("=" * 60)
    print("Configured input files:", CFG.input_files)
    print("Output directory:", CFG.output_dir)
    print("Device selected:", CFG.device)
    print("Episodes configured:", CFG.episodes)
    print("Augmentation enabled:", CFG.use_augmentation)
    print("=" * 60 + "\n")

    try:
        main()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
