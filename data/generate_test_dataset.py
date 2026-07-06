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

import pandas as pd
import numpy as np

np.random.seed(42)

n = 7000

pressure = np.random.uniform(4.5, 12.0, n)
temperature = np.random.uniform(95, 135, n)
catalyst = np.random.uniform(2.0, 5.0, n)
time = np.random.uniform(20, 40, n)

# synthetic nonlinear "yield function"
yield_val = (
    20
    + 8 * pressure
    + 6 * catalyst
    + 0.4 * time
    - 0.5 * (temperature - 115) ** 2
    + np.random.normal(0, 2, n)
)

yield_val = np.clip(yield_val, 40, 100)

df = pd.DataFrame({
    "Pressure": pressure.round(2),
    "Temperature": temperature.round(1),
    "Catalyst": catalyst.round(2),
    "Time": time.round(1),
    "Yield": yield_val.round(1)
})

df.to_csv("synthetic_yield_data_7k.csv", index=False)

print(df.head())
