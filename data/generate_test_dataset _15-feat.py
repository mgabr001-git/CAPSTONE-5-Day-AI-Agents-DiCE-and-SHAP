# Copyright (c) 2026 MyCompany LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

np.random.seed(42)

# Number of experiments
n = 15000

# -----------------------------
# Generate synthetic features
# -----------------------------

Pressure = np.random.uniform(4, 8, n)                 # atm
Temperature = np.random.uniform(90, 140, n)           # °C
Catalyst = np.random.uniform(1, 5, n)                 # %
Time = np.random.uniform(10, 60, n)                   # min
FlowRate = np.random.uniform(20, 80, n)               # L/min
StirringSpeed = np.random.uniform(100, 800, n)        # rpm
pH = np.random.uniform(4, 10, n)
Humidity = np.random.uniform(20, 80, n)               # %
FeedConcentration = np.random.uniform(0.5, 2.0, n)    # mol/L
ReactorVolume = np.random.uniform(1, 10, n)           # L
CoolingRate = np.random.uniform(1, 10, n)
Voltage = np.random.uniform(100, 240, n)
Current = np.random.uniform(1, 20, n)
Impurity = np.random.uniform(0, 5, n)                 # %
OperatorExperience = np.random.uniform(1, 10, n)      # years

# -----------------------------
# Ideal (noise-free) yield model
# -----------------------------

yield_clean = (
    15
    + 7.0 * Pressure
    + 5.5 * Catalyst
    + 0.35 * Time
    - 0.06 * (Temperature - 105) ** 2
    + 0.08 * FlowRate
    + 0.01 * StirringSpeed
    + 1.5 * FeedConcentration
    - 0.8 * Impurity
    + 0.4 * OperatorExperience
    + 0.6 * Pressure * Catalyst
    - 0.015 * Pressure * (Temperature - 105)
)

# -----------------------------
# Add 5% Gaussian measurement noise
# -----------------------------

noise = np.random.normal(
    loc=0,
    scale=0.05 * np.std(yield_clean),
    size=n
)

yield_val = yield_clean + noise

# Keep yields in a realistic range
yield_val = np.clip(yield_val, 40, 95)

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "Pressure": Pressure.round(2),
    "Temperature": Temperature.round(1),
    "Catalyst": Catalyst.round(2),
    "Time": Time.round(1),
    "FlowRate": FlowRate.round(1),
    "StirringSpeed": StirringSpeed.round(0),
    "pH": pH.round(2),
    "Humidity": Humidity.round(1),
    "FeedConcentration": FeedConcentration.round(2),
    "ReactorVolume": ReactorVolume.round(2),
    "CoolingRate": CoolingRate.round(2),
    "Voltage": Voltage.round(1),
    "Current": Current.round(2),
    "Impurity": Impurity.round(2),
    "OperatorExperience": OperatorExperience.round(1),
    "Yield": yield_val.round(2)
})

df.to_csv("synthetic_yield_dataset_15_features.csv", index=False)

print(df.head())
