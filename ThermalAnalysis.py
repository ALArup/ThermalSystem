import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================= Testing ==================

HX_data = pd.read_excel("data.xlsm", sheet_name="3 - CRAH")

T_ha_raw   = HX_data.iloc[9:2020, 9].to_numpy()
T_chws_raw = HX_data.iloc[9:2020, 10].to_numpy()
T_ln_raw   = HX_data.iloc[9:2020, 14].to_numpy()

V_in_1_raw = HX_data.iloc[9:2020, 7].to_numpy()
V_in_2_raw = HX_data.iloc[9:2020, 8].to_numpy()

T_ha   = pd.Series(T_ha_raw, index = range(-10, -10 + len(T_ha_raw)))
T_chws = pd.Series(T_chws_raw, index = range(-10, -10 + len(T_chws_raw)))
T_ln   = pd.Series(T_ln_raw, index = range(-10, -10 + len(T_ln_raw)))
V_in_1 = pd.Series(V_in_1_raw, index = range(-10, -10 + len(V_in_1_raw)))
V_in_2 = pd.Series(V_in_2_raw, index = range(-10, -10 + len(V_in_2_raw)))

WCC_data = pd.read_excel("data.xlsm", sheet_name="1 - WCC")

V_in_wcc_raw = WCC_data.iloc[10:2020, 7].to_numpy()
T_in_wcc_raw = WCC_data.iloc[10:2020, 8].to_numpy()

V_in_wcc_data = pd.Series(V_in_wcc_raw, index = range(-10, -10 + len(V_in_wcc_raw)))
T_in_wcc_data = pd.Series(T_in_wcc_raw, index = range(-10, -10 + len(T_in_wcc_raw)))

TT_data = pd.read_excel("data.xlsm", sheet_name="2 - Thermal Tank AC")

V_in_TT_raw = TT_data.iloc[10:2020, 7].to_numpy()
T_in_TT_raw = TT_data.iloc[10:2020, 8].to_numpy()

V_in_TT_data = pd.Series(V_in_TT_raw, index = range(-10, -10 + len(V_in_TT_raw)))
T_in_TT_data = pd.Series(T_in_TT_raw, index = range(-10, -10 + len(T_in_TT_raw)))

HA_data = pd.read_excel("data.xlsm", sheet_name="2 - Hot Aisle")

V_in_HA_raw = HA_data.iloc[10:2020, 7].to_numpy()
T_in_HA_raw = HA_data.iloc[10:2020, 8].to_numpy()

V_in_HA_data = pd.Series(V_in_HA_raw, index = range(-10, -10 + len(V_in_HA_raw)))
T_in_HA_data = pd.Series(T_in_HA_raw, index = range(-10, -10 + len(T_in_HA_raw)))

CA_data = pd.read_excel("data.xlsm", sheet_name="2 - Cold Aisle")

V_in_CA_raw = CA_data.iloc[10:2020, 7].to_numpy()
T_in_CA_raw = CA_data.iloc[10:2020, 8].to_numpy()

V_in_CA_data = pd.Series(V_in_CA_raw, index = range(-10, -10 + len(V_in_CA_raw)))
T_in_CA_data = pd.Series(T_in_CA_raw, index = range(-10, -10 + len(T_in_CA_raw)))

# ============================================

def compute_eps(NTU: float, C_r: float, HX_type: str ):
    if HX_type == "Parallel Flow":
        return (1.0 + np.exp(-NTU * (1.0 + C_r))) / (1.0 + C_r)
    elif HX_type == "Counter Flow":
        return (1.0 - np.exp(-NTU * (1.0 - C_r))) / (1.0 - C_r * np.exp(-NTU * (1.0 - C_r)))
    else:
        print("Error! Heat exchanger not recognised!")

class Material:
    def __init__(self, _density: float, _cp: float):
        self.density = _density
        self.cp      = _cp

class WCC:
    def __init__(self, _cap_per_unit, _n_duty, _n_standby, _n_remain, _set_pt_temp,
                 t0: float, t1: float, t2: float, t3: float
                 ):
        
        self.cap_per_unit = _cap_per_unit
        self.n_duty       = _n_duty
        self.n_standby    = _n_standby
        self.n_remain     = _n_remain
        self.set_pt_temp  = _set_pt_temp

        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

    def avail_cool(self, t: float, set_point_load: float):
        if t < self.t0:
            return -set_point_load
        elif t < self.t1:
            return -self.cap_per_unit * self.n_remain
        elif t < self.t2:
            return -(self.cap_per_unit * self.n_remain + self.cap_per_unit * (self.n_duty - self.n_standby - self.n_remain) * ((t - self.t1) / (self.t2 - self.t1)) * 0.8)
        elif t < self.t3:
            return -(self.cap_per_unit * self.n_remain + self.cap_per_unit * (self.n_duty - self.n_standby - self.n_remain) * (0.8 + 0.2 * ((t - self.t2) / (self.t3 - self.t2))))
        elif t >= self.t3:
            return -self.cap_per_unit * (self.n_duty - self.n_standby - self.n_remain)

class Solver:
    def __init__(self, _dt: float, _n_steps: int, water: Material, air: Material, wcc: WCC
                                , _T_in_CA_0: float, _T_out_CA_0: float, _V_in_CA_0: float, _V_cv_CA: float          # Cold Aisle
                                , _Q_IT: float                                                   # IT (no initial temperature needed; just flow in and heat in)
                                , _T_out_HA_0: float, _V_cv_HA: float                            # Hot Aisle (temperature obtained from flow in)
                                , _T_in_WCC_0: float, _V_in_WCC_0: float
                                , _T_out_TT_0: float, _V_cv_TT: float                            # Thermal tank
                                , _Q_DH: float, T_out_w_CRAH_0: float, T_out_a_CRAH_0: float
                                , _HX_type: str, _K_a: float, _K_w: float
                                , _V_a_max: float, _V_w_max: float, _V_a_normal: float, _V_w_normal: float 
                                ):

        self.dt      = _dt
        self.n_steps = _n_steps
        self.water   = water
        self.air     = air

        self.Q_IT    = _Q_IT
        self.Q_DH    = _Q_DH

        self.V_cv_CA = _V_cv_CA
        self.V_cv_HA = _V_cv_HA
        self.V_cv_TT = _V_cv_TT

        self.HX_type = _HX_type

        self.K_a = _K_a
        self.K_w = _K_w

        self.V_a_max = _V_a_max
        self.V_w_max = _V_w_max

        self.V_a_normal = _V_a_normal
        self.V_w_normal = _V_w_normal

        self.steps    = {}
        self.step_idx = -10

        # Cooling Aisle:
        V_in_CA   = _V_in_CA_0
        T_in_CA   = _T_in_CA_0
        T_out_CA  = _T_out_CA_0
        V_out_CA  = V_in_CA

        # IT Racks:
        V_in_IT  = V_out_CA           # All flow from CA -> ITR
        T_in_IT  = T_out_CA
        T_out_IT = T_in_IT + self.Q_IT / (self.air.cp * self.air.density * V_in_IT)
        V_out_IT = V_in_IT

        # Hot Aisle:
        V_in_HA    = V_out_IT           # All flow from ITR -> HA
        T_in_HA    = T_out_IT
        T_out_HA   = _T_out_HA_0
        V_out_HA   = V_in_HA

        # Water-Cooled Chillers:
        V_in_WCC = _V_in_WCC_0
        T_in_WCC = _T_in_WCC_0

        Q_req_WCC    = -(T_in_WCC - wcc.set_pt_temp) * water.density * water.cp * V_in_WCC          # -ve for cooling
        Q_avail_WCC  = wcc.avail_cool(self.step_idx * self.dt, -Q_req_WCC)       # -ve for cooling
        Q_actual_WCC = max(Q_avail_WCC, Q_req_WCC)

        T_out_WCC = T_in_WCC + Q_actual_WCC / (water.density * water.cp * V_in_WCC)
        V_out_WCC = V_in_WCC

        # Thermal Tanks:
        V_in_TT  = V_out_WCC
        T_in_TT  = T_out_WCC

        T_out_TT = _T_out_TT_0
        V_out_TT = V_in_TT

        # Computer Room Air Handling:
        V_1h_CRAH = V_out_HA                # Hot Aisle hot air in
        # V_2c_CRAH = V_out_TT                # Thermal Tank cold water in

        T_1h_CRAH = T_out_HA
        T_2c_CRAH = T_out_TT

        T_1c_CRAH = T_out_a_CRAH_0          # Defining exit conditions for first step
        T_2h_CRAH = T_out_w_CRAH_0          # Defining exit conditions for first step

        Q_CRAH = self.Q_DH
        V_2c_CRAH = Q_CRAH / (water.cp * water.density * (T_2h_CRAH - T_2c_CRAH))

        dTln = (np.abs(T_1h_CRAH - T_2h_CRAH) - np.abs(T_2c_CRAH - T_1c_CRAH)) / (np.log(np.abs(T_1h_CRAH - T_2h_CRAH) / np.abs(T_2c_CRAH - T_1c_CRAH)))
        UA = Q_CRAH / dTln

        C_water = water.density * water.cp * V_2c_CRAH
        C_air   = air.density * air.cp * V_out_HA
        
        C_min = min(C_water, C_air)
        C_r   = C_min / max(C_water, C_air)
        
        Q_max = C_min * (T_1h_CRAH - T_2c_CRAH)

        NTU = UA / C_min

        eps = compute_eps(NTU, C_r, self.HX_type)
        Q_actual_CRAH =  Q_max * eps

        T_1c_CRAH_next = T_1h_CRAH - Q_actual_CRAH / C_air
        T_2h_CRAH_next = T_2c_CRAH + Q_actual_CRAH / C_water

        self.T_ref_1 = T_1h_CRAH
        self.T_ref_2 = T_1c_CRAH

        V_1_out_CRAH = min(V_1h_CRAH + (T_1h_CRAH - self.T_ref_1) * self.K_a * (self.V_a_max - self.V_a_normal), self.V_a_max)                                 # Air to Cold Aisle
        V_2_out_CRAH = min(V_2c_CRAH + (T_1c_CRAH - self.T_ref_2) * self.K_w * (self.V_w_max - self.V_w_normal), self.V_w_max)

        self.steps[self.step_idx] = {
            "t"              : self.step_idx * self.dt,
            "V_in_CA"        : V_in_CA,
            "T_in_CA"        : T_in_CA,
            "V_out_CA"       : V_out_CA,
            "T_out_CA"       : T_out_CA,

            "V_in_IT"        : V_in_IT,
            "T_in_IT"        : T_in_IT,
            "T_out_IT"       : T_out_IT,
            "V_out_IT"       : V_out_IT,

            "V_in_HA"        : V_in_HA,
            "T_in_HA"        : T_in_HA,
            "V_out_HA"       : V_out_HA,
            "T_out_HA"       : T_out_HA,

            "V_in_WCC"       : V_in_WCC,
            "T_in_WCC"       : T_in_WCC,
            "V_out_WCC"      : V_out_WCC,
            "T_out_WCC"      : T_out_WCC,

            "V_in_TT"        : V_in_TT,
            "T_in_TT"        : T_in_TT,
            "T_out_TT"       : T_out_TT,

            "V_1h_CRAH"      : V_1h_CRAH,
            "V_2c_CRAH"      : V_2c_CRAH,

            "T_1h_CRAH"      : T_1h_CRAH,
            "T_2c_CRAH"      : T_2c_CRAH,

            "C_min"          : C_min,
            "C_r"            : C_r,
            "Q_max"          : Q_max,
            "dTln"           : dTln,
            "UA"             : UA,
            "NTU"            : NTU,
            "eps"            : eps,
            "Q_actual_CRAH"  : Q_actual_CRAH,

            "T_1c_CRAH"      : T_1c_CRAH,
            "T_2h_CRAH"      : T_2h_CRAH,

            "V_1_out_CRAH"   : V_1_out_CRAH,
            "V_2_out_CRAH"   : V_2_out_CRAH,
        
            "T_1c_CRAH_next" : T_1c_CRAH_next,
            "T_2h_CRAH_next" : T_2h_CRAH_next,

            "dT1"            : T_1h_CRAH - self.T_ref_1, 
            "dT2"            : T_1c_CRAH - self.T_ref_2
        }
        
        self.step_idx += 1

    def CRAH_test(self):
        T_1h = T_ha
        T_2c = T_chws

        T_1h_0 = T_ha[-10]
        T_2c_0 = T_chws[-10]

        T_1c_0 = 30.5
        T_2h_0 = 32.0

        V_1_in_0 = 519.2939525358590
        V_2_in_0 = 0.2050217076491

        Q_0 = 6804.7

        C_air   = air.cp * air.density * V_1_in_0
        C_water = water.cp * water.density * V_2_in_0

        C_min_0 = min(C_air, C_water)
        C_r_0   = C_min_0 / max(C_air, C_water)

        Q_max_0 = C_min_0 * (T_1h_0 - T_2c_0)

        dT_1_0 = np.abs(T_1h_0 - T_2h_0)
        dT_2_0 = np.abs(T_1c_0 - T_2c_0)

        dTln_0 = (
            (dT_1_0 - dT_2_0)
            / np.log(dT_1_0 / dT_2_0)
        )

        UA_0 = Q_0 / dTln_0
        NTU_0 = UA_0 / C_min_0

        eps_0 = (1.0 - np.exp(-NTU_0 * (1.0 - C_r_0))) / (1.0 - C_r_0 * np.exp(-NTU_0 * (1.0 - C_r_0)))

        T_1c_0 = T_1h_0 - Q_0 / C_air
        T_2h_0 = T_2c_0 + Q_0 / C_water

        T1ref = 41.7
        T2ref = 30.5

        dT1 = T_1h_0 - T1ref
        dT2 = T_1c_0 - T2ref

        V_1_in_next = min(V_1_in_0 + dT1 * 3.5492, 608.243)
        V_2_in_next = min(V_2_in_0 + dT2 * 0.0009, 0.2401)

        table = pd.DataFrame([{
            "t"     :      -10,
            "V_1_in":   V_1_in_0,
            "V_2_in":   V_2_in_0,
            "T_1c":     T_1c_0,
            "T_2h":     T_2h_0,
            "C_min":    C_min_0,
            "C_r":      C_r_0,
            "Q_max":    Q_max_0,
            "dTln":     dTln_0,
            "UA":       UA_0,
            "NTU":      NTU_0,
            "eps":      eps_0,
            "Q_actual": Q_0,
            "T_1h":     T_1h_0,
            "T_2c":     T_2c_0,
            "dT1":      dT1,
            "dT2":      dT2,
            "V_1_in_next": V_1_in_next, 
            "V_2_in_next": V_2_in_next, 
        }])

        for t in range(-9, 2000):
            # Previous row
            V_1_in = table["V_1_in_next"].iloc[-1]
            V_2_in = table["V_2_in_next"].iloc[-1]

            T_2h = table["T_2h"].iloc[-1]
            T_1c = table["T_1c"].iloc[-1]

            Q_prev = table["Q_actual"].iloc[-1]

            # Current inlet temperatures
            T_1h_i = T_1h.loc[t]
            T_2c_i = T_2c.loc[t]

            # Capacity rates
            C_air   = air.cp * air.density * V_1_in
            C_water = water.cp * water.density * V_2_in

            C_min = min(C_air, C_water)
            C_r   = C_min / max(C_air, C_water)

            # Maximum possible heat transfer
            Q_max = C_min * (T_1h_i - T_2c_i)

            # LMTD
            # LMTD
            dT_1 = np.abs(T_1h_i - T_2h)
            dT_2 = np.abs(T_1c - T_2c_i)

            if np.isclose(dT_1, dT_2):
                dTln = dT_1
            else:
                dTln = (
                    (dT_1 - dT_2)
                    / np.log(dT_1 / dT_2)
                )

            # UA / NTU
            UA = Q_prev / dTln
            NTU = UA / C_min

            # UA / NTU
            UA  = Q_prev / dTln
            NTU = UA / C_min

            # Effectiveness
            eps = (
                1.0 - np.exp(-NTU * (1.0 - C_r))
            ) / (
                1.0 - C_r * np.exp(-NTU * (1.0 - C_r))
            )

            # Heat transfer
            Q_actual = Q_max * eps

            # Outlet temperatures
            T_1c_new = T_1h_i - Q_actual / C_air
            T_2h_new = T_2c_i + Q_actual / C_water

            # Controller temperature differences
            dT1 = T_1h_i - T1ref
            dT2 = T_1c_new - T2ref

            # Next flow rates
            V_1_in_next = min(
                V_1_in + dT1 * 3.5492,
                608.243
            )

            V_2_in_next = min(
                V_2_in + dT2 * 0.0009,
                0.2401
            )

            # Save timestep
            table.loc[len(table)] = {
                "t":             t,
                "V_1_in":        V_1_in,
                "V_2_in":        V_2_in,
                "T_1c":          T_1c_new,
                "T_2h":          T_2h_new,
                "C_min":         C_min,
                "C_r":           C_r,
                "Q_max":         Q_max,
                "dTln":          dTln,
                "UA":            UA,
                "NTU":           NTU,
                "eps":           eps,
                "Q_actual":      Q_actual,
                "T_1h":          T_1h_i,
                "T_2c":          T_2c_i,
                "dT1":           dT1,
                "dT2":           dT2,
                "V_1_in_next":   V_1_in_next,
                "V_2_in_next":   V_2_in_next
            }


        print(table.to_string(index=False))

        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(table["t"], table["UA"], label = "UA")
        ax.grid()
        ax.legend()
        plt.savefig("test1.png")

        fig, ax = plt.subplots(figsize = (8, 4))
        
        ax.plot(table["t"], table["Q_actual"], label = "Q_actual")
        ax.grid()
        ax.legend()
        plt.savefig("test2.png")

    def compute_next_step(self):
        prev_step = self.steps[self.step_idx - 1]

        # Cold Aisle:
        V_in_CA  = prev_step["V_1_out_CRAH"]
        T_in_CA  = prev_step["T_1c_CRAH"]
       
        # Testing:
        # V_in_CA = V_in_CA_data[self.step_idx]
        # T_in_CA = T_in_CA_data[self.step_idx]
       
        T_out_CA = prev_step["T_out_CA"] * (1.0 - (V_in_CA * self.dt) / self.V_cv_CA) + T_in_CA * (V_in_CA / self.V_cv_CA) * self.dt
        V_out_CA = V_in_CA

        # IT Room:
        V_in_IT  = prev_step["V_out_CA"]
        T_in_IT  = prev_step["T_out_CA"]
        T_out_IT = T_out_CA + self.Q_IT / (self.air.density * self.air.cp * V_in_IT)
        V_out_IT = V_in_IT

        # Hot Aisle:
        V_in_HA = prev_step["V_out_IT"]
        T_in_HA = prev_step["T_out_IT"]

        # Testing:
        # V_in_HA = V_in_HA_data[self.step_idx]
        # T_in_HA = T_in_HA_data[self.step_idx]

        T_out_HA = prev_step["T_out_HA"] * (1.0 - (V_in_HA * self.dt) / self.V_cv_HA) + T_in_HA * (V_in_HA / self.V_cv_HA) * self.dt
        V_out_HA = V_in_HA

        # Water-Cooled Chillers:
        # V_in_WCC = prev_step["V_2_out_CRAH"]
        # T_in_WCC = prev_step["T_2h_CRAH_next"]

        # Testing:
        V_in_WCC = V_in_wcc_data[self.step_idx]
        T_in_WCC = T_in_wcc_data[self.step_idx]

        Q_req_WCC = -(T_in_WCC - wcc.set_pt_temp) * self.water.cp * self.water.density * V_in_WCC          # -ve for cooling
        Q_avail_WCC = wcc.avail_cool(self.step_idx * self.dt, -Q_req_WCC)       # -ve for cooling
        Q_actual_WCC = max(Q_avail_WCC, Q_req_WCC)

        T_out_WCC = T_in_WCC + Q_actual_WCC / (water.density * self.water.cp * V_in_WCC)

        V_out_WCC = V_in_WCC

        # Thermal Tanks:
        V_in_TT = prev_step["V_out_WCC"]
        T_in_TT = prev_step["T_out_WCC"]

        # V_in_TT = V_in_TT_data[self.step_idx]
        # T_in_TT = T_in_TT_data[self.step_idx]

        T_out_TT = prev_step["T_out_TT"] * (1.0 - (V_in_TT * self.dt) / self.V_cv_TT) + T_in_TT * (V_in_TT / self.V_cv_TT) * self.dt
        V_out_TT = V_in_TT

        # Computer Room Air Handling:
        V_1h_CRAH = prev_step["V_1_out_CRAH"]                # Updated from previous step
        V_2c_CRAH = prev_step["V_2_out_CRAH"]                # Updated from previous step

        T_1h_CRAH = T_ha[self.step_idx]
        T_2c_CRAH = T_chws[self.step_idx]

        T_1c_CRAH = prev_step["T_1c_CRAH_next"]
        T_2h_CRAH = prev_step["T_2h_CRAH_next"]

        Q_prev    = prev_step["Q_actual_CRAH"]

        C_air   = air.cp * air.density * V_1h_CRAH
        C_water = water.cp * water.density * V_2c_CRAH

        C_min = min(C_air, C_water)
        C_r   = C_min / max(C_air, C_water)

        Q_max = C_min * (T_1h_CRAH - T_2c_CRAH)

        dT_1 = np.abs(T_1h_CRAH - T_2h_CRAH)
        dT_2 = np.abs(T_1c_CRAH - T_2c_CRAH)

        if np.isclose(dT_1, dT_2):
            dTln = dT_1
        else:
            dTln = (
                (dT_1 - dT_2)
                / np.log(dT_1 / dT_2)
            )

        # dTln = T_ln[self.step_idx]

        UA = Q_prev / dTln
        NTU = UA / C_min

        # eps = compute_eps(NTU, C_r, self.HX_type)

        eps = (
            1.0 - np.exp(-NTU * (1.0 - C_r))
        ) / (
            1.0 - C_r * np.exp(-NTU * (1.0 - C_r))
        )

        Q_actual_CRAH =  Q_max * eps

        T_1c_CRAH_next = T_1h_CRAH - Q_actual_CRAH / C_air
        T_2h_CRAH_next = T_2c_CRAH + Q_actual_CRAH / C_water

        V_1_out_CRAH = min(V_1h_CRAH + (T_1h_CRAH - self.T_ref_1) * self.K_a * (self.V_a_max - self.V_a_normal), self.V_a_max)                                 # Air to Cold Aisle
        V_2_out_CRAH = min(V_2c_CRAH + (T_1c_CRAH_next - self.T_ref_2) * self.K_w * (self.V_w_max - self.V_w_normal), self.V_w_max)

        self.steps[self.step_idx] = {
                    "t"              : self.step_idx * self.dt,
                    "V_in_CA"        : V_in_CA,
                    "T_in_CA"        : T_in_CA,
                    "V_out_CA"       : V_out_CA,
                    "T_out_CA"       : T_out_CA,

                    "V_in_IT"        : V_in_IT,
                    "T_in_IT"        : T_in_IT,
                    "T_out_IT"       : T_out_IT,
                    "V_out_IT"       : V_out_IT,

                    "V_in_HA"        : V_in_HA,
                    "T_in_HA"        : T_in_HA,
                    "V_out_HA"       : V_out_HA,
                    "T_out_HA"       : T_out_HA,

                    "V_in_WCC"       : V_in_WCC,
                    "T_in_WCC"       : T_in_WCC,
                    "V_out_WCC"      : V_out_WCC,
                    "T_out_WCC"      : T_out_WCC,

                    "V_in_TT"   : V_in_TT,
                    "T_in_TT"   : T_in_TT,
                    "T_out_TT"  : T_out_TT,

                    "V_1h_CRAH"      : V_1h_CRAH,
                    "V_2c_CRAH"      : V_2c_CRAH,

                    "T_1h_CRAH"      : T_1h_CRAH,
                    "T_2c_CRAH"      : T_2c_CRAH,

                    "C_min"          : C_min,
                    "C_r"            : C_r,
                    "Q_max"          : Q_max,
                    "dTln"           : dTln,
                    "UA"             : UA,
                    "NTU"            : NTU,
                    "eps"            : eps,
                    "Q_actual_CRAH"  : Q_actual_CRAH,

                    "T_1c_CRAH"      : T_1c_CRAH,
                    "T_2h_CRAH"      : T_2h_CRAH,

                    "V_1_out_CRAH"   : V_1_out_CRAH,
                    "V_2_out_CRAH"   : V_2_out_CRAH,
                
                    "T_1c_CRAH_next" : T_1c_CRAH_next,
                    "T_2h_CRAH_next" : T_2h_CRAH_next,
                    
                    "dT1"            : T_1h_CRAH - self.T_ref_1, 
                    "dT2"            : T_1c_CRAH - self.T_ref_2
                }
                
        self.step_idx += 1

    def print_data(self):
        self.data = pd.DataFrame.from_dict( self.steps, orient = "index")
        self.data.to_excel("ThermalData.xlsx", index = False, sheet_name = "Sheet 1", float_format = "%.10f")

    def plot_data(self):
        t              = self.data.iloc[:, 0]

        V_in_CA        = self.data.iloc[:, 1]
        T_in_CA        = self.data.iloc[:, 2]
        V_out_CA       = self.data.iloc[:, 3]
        T_out_CA       = self.data.iloc[:, 4]

        V_in_IT        = self.data.iloc[:, 5]
        T_in_IT        = self.data.iloc[:, 6]
        T_out_IT       = self.data.iloc[:, 7]
        V_out_IT       = self.data.iloc[:, 8]

        V_in_HA        = self.data.iloc[:, 9]
        T_in_HA        = self.data.iloc[:, 10]
        V_out_HA       = self.data.iloc[:, 11]
        T_out_HA       = self.data.iloc[:, 12]

        V_in_WCC       = self.data.iloc[:, 13]
        T_in_WCC       = self.data.iloc[:, 14]
        V_out_WCC      = self.data.iloc[:, 15]
        T_out_WCC      = self.data.iloc[:, 16]

        V_in_TT        = self.data.iloc[:, 17]
        T_in_TT        = self.data.iloc[:, 18]
        T_out_TT       = self.data.iloc[:, 19]

        V_1h_CRAH      = self.data.iloc[:, 20]
        V_2c_CRAH      = self.data.iloc[:, 21]

        T_1h_CRAH      = self.data.iloc[:, 22]
        T_2c_CRAH      = self.data.iloc[:, 23]

        C_min          = self.data.iloc[:, 24]
        C_r            = self.data.iloc[:, 25]
        Q_max          = self.data.iloc[:, 26]
        dTln           = self.data.iloc[:, 27]
        UA             = self.data.iloc[:, 28]
        NTU            = self.data.iloc[:, 29]
        eps            = self.data.iloc[:, 30]
        Q_actual_CRAH  = self.data.iloc[:, 31]

        T_1c_CRAH      = self.data.iloc[:, 32]
        T_2h_CRAH      = self.data.iloc[:, 33]

        V_1_out_CRAH   = self.data.iloc[:, 34]
        V_2_out_CRAH   = self.data.iloc[:, 35]

        T_1c_CRAH_next = self.data.iloc[:, 36]
        T_2h_CRAH_next = self.data.iloc[:, 37]


        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(t, T_2c_CRAH, label = "T_2c_CRAH")
        ax.plot(t, T_1h_CRAH, label = "T_1h_CRAH")
        ax.plot(t, T_1c_CRAH, label = "T_1c_CRAH")
        ax.plot(t, T_2h_CRAH, label = "T_2h_CRAH")

        ax.grid()
        ax.legend()

        plt.savefig("HX_temps.png")

        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(t, Q_actual_CRAH, label = "Q_actual_CRAH")
        ax.grid()
        ax.legend()

        plt.savefig("HX_energy.png")

        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(t, V_2c_CRAH, label = "V_2c_CRAH")
        ax.plot(t, V_1h_CRAH, label = "V_1h_CRAH")
        ax.grid()
        ax.legend()

        plt.savefig("HX_flow_rates.png")

        dT1 = T_1h_CRAH - self.T_ref_1
        dT2 = T_1c_CRAH - self.T_ref_2

        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(t, dT1, label = "dT1")
        ax.plot(t, dT2, label = "dT2")
        ax.grid()
        ax.legend()

        plt.savefig("HX_dT.png")

        print(f"Max dT1: {np.max(dT1)}")
        print(f"Max dT2: {np.max(dT2)}")
        print(f"Min dTln: {np.min(dTln)}")
        print(f"Max T_1c: {np.max(T_1c_CRAH)}")
        print(f"Max T_2h: {np.max(T_2h_CRAH)}")

        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(t, dTln, label = "dTln")
        ax.grid()
        ax.legend()

        plt.savefig("dTln.png")

        fig, ax = plt.subplots(figsize = (8, 4))

        ax.plot(t, UA, label = "UA")
        ax.grid()
        ax.legend()

        plt.savefig("UA.png")

        fig, ax = plt.subplots(figsize = (8, 4))
        
        ax.plot(t, Q_max, label = "Q_max")
        ax.grid()
        ax.legend()

        plt.savefig("Q_max.png")

        fig, ax = plt.subplots(figsize = (8, 4))
        
        ax.plot(t, eps, label = "eps")
        ax.grid()
        ax.legend()

        plt.savefig("eps.png")

    def run(self):
        while self.step_idx < self.n_steps:
            self.compute_next_step()
        self.print_data()
        self.plot_data()

        self.CRAH_test()

density_air, cp_air = 1.163, 1.006
density_water, cp_water = 997.3, 4.16
cap_per_unit, n_duty, n_standby, n_remain, set_pt_temp = 1260.0, 6, 1, 0, 24.0
t0, t1, t2, t3 = 0, 93, 188, 228

air   = Material(density_air, cp_air)
water = Material(density_water, cp_water)
wcc   = WCC(cap_per_unit, n_duty, n_standby, n_remain, set_pt_temp, t0, t1, t2, t3)

n_steps = 2000

T_out_CA_0 = 30.5
T_in_CA_0  = 30.5
V_in_CA_0  = 519.2939525358590
V_cv_CA    = 3123.645
Q_IT       = 6804.7

T_out_HA_0 = 41.7
V_cv_HA    = 3784.125

T_in_WCC_0 = 32.0
V_in_WCC_0 = 0.22778

T_out_TT_0 = 24.0
V_cv_TT    = 27.6

Q_DH = Q_IT
T_out_w_CRAH_0 = 32.0
T_out_a_CRAH_0 = 30.5
HX_type = "Counter Flow"
K_a = 0.04
K_w = 0.025

V_a_max = 608.0242534475010
V_w_max = 0.240053191694

V_a_normal = 519.29395253586
V_w_normal = 0.2050217076

sol_1 = Solver(1.0, n_steps, water, air, wcc,
               T_in_CA_0, T_out_CA_0, V_in_CA_0, V_cv_CA,
               Q_IT,
               T_out_HA_0, V_cv_HA,
               T_in_WCC_0, V_in_WCC_0,
               T_out_TT_0, V_cv_TT,
               Q_DH, T_out_w_CRAH_0, T_out_a_CRAH_0,
               HX_type, K_a, K_w,
               V_a_max, V_w_max, V_a_normal, V_w_normal
)

sol_1.run()