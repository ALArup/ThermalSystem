import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", None)

Kp1 = 0.04
Kp2 = 0.025

HX_data  = pd.read_excel("data.xlsm", sheet_name = "3 - CRAH")
HA_data  = pd.read_excel("data.xlsm", sheet_name = "2 - Hot Aisle")
CA_data  = pd.read_excel("data.xlsm", sheet_name = "2 - Cold Aisle")
ITR_data = pd.read_excel("data.xlsm", sheet_name = "1 - AC IT Racks")

T_ha     = list(HX_data.iloc[9:2020, 9])
T_chws   = list(HX_data.iloc[9:2020, 10])

V_HA_in  = list(HA_data.iloc[9:2020, 7])
T_HA_in  = list(HA_data.iloc[9:2020, 8])

V_CA_in  = list(CA_data.iloc[9:2020, 7])
T_CA_in  = list(CA_data.iloc[9:2020, 8])

V_ITR_in = list(ITR_data.iloc[9:2020, 7])
T_ITR_in = list(ITR_data.iloc[9:2020, 8])

class HX_step():
    def __init__(self):
        pass

    def generate_step_properties(self, 
                                 _t: float,
                                 _V_air: float,
                                 _V_water: float,
                                 density_air: float,
                                 density_water: float,
                                 cp_air: float,
                                 cp_water: float,

                                 T_2c_active: float,
                                 T_2h_prev: float,
                                 T_1c_prev: float,
                                 T_1h_active: float,

                                 Q_actual_prev: float,
                                 exchanger_type: str,

                                 V_air_max: float,
                                 V_water_max: float,

                                 T_1h_0: float,
                                 T_1c_0: float,
                                 V_air_0: float,
                                 V_water_0: float,
                                 ):
        self.t         = _t
        self.V_air     = _V_air
        self.V_water   = _V_water

        C_air   = density_air * self.V_air * cp_air
        C_water   = density_water * self.V_water * cp_water
        self.C_min = min(C_air, C_water)
        self.C_r   = self.C_min / max(C_air, C_water)

        dT1 = np.abs(T_1h_active - T_2h_prev)
        dT2 = np.abs(T_1c_prev - T_2c_active)
        self.T_ln  = (dT1 - dT2) / (np.log(dT1 / dT2))

        self.q_max = self.C_min * (T_1h_active - T_2c_active)
        self.UA    = Q_actual_prev / self.T_ln                  # UA calculated from previous actual heat
        self.NTU   = self.UA / self.C_min
        
        if exchanger_type == "Parallel":
            self.eps = (1.0 + np.exp(-self.NTU * (1.0 + self.C_r))) / (1.0 + self.C_r)
        elif exchanger_type == "Counter-current":
            self.eps = (1.0 - np.exp(-self.NTU * (1.0 -self.C_r))) / (1.0 - self.C_r * np.exp(-self.NTU * (1.0 - self.C_r)))
        else:
            print("ERROR! Invalid heat exchanger type detected!")   
            print("eps not defined")

        # Store temperatures (for reference):
        self.T_1h = T_1h_active
        self.T_2c = T_2c_active

        # Update future conditions for next step:
        self.q_actual_new = self.eps * self.q_max
        self.T_1c = T_1h_active - self.q_actual_new / C_air
        self.T_2h = T_2c_active + self.q_actual_new / C_water

        self.V_air_new = min(self.V_air + (self.T_1h - T_1h_0) * Kp1 * (V_air_max - V_air_0), V_air_max)
        self.V_water_new = min(self.V_water + (self.T_1c - T_1c_0) * Kp2 * (V_water_max - V_water_0), V_water_max)

class HeatExchanger():
    steps = {}
    idx = 0
    
    def __init__(self, 
                _Q_total: float,            # in kW
                _exchanger_type: str,
                _density_air: float,
                _density_water: float,
                _cp_air: float,
                _cp_water: float,

                _V_air_0: float,
                _V_water_0: float,

                _T_2c_0: float,
                _T_2h_0: float,
                _T_1c_0: float,
                _T_1h_0: float, 

                _V_air_max: float,
                _V_water_max: float,
                _dt: float
                ):

        self.Q_total        = _Q_total
        self.exchanger_type = _exchanger_type
        self.density_air      = _density_air
        self.density_water      = _density_water
        self.cp_air           = _cp_air
        self.cp_water           = _cp_water
        self.V_air            = _V_air_0
        self.V_water            = _V_water_0
        self.T_1c_0         = _T_1c_0
        self.T_1h_0         = _T_1h_0
        self.T_2c_0         = _T_2c_0
        self.T_2h_0         = _T_2h_0
        self.V_air_max        = _V_air_max
        self.V_water_max        = _V_water_max
        self.dt             = _dt

    def print_properties(self):
        print("\n========== HEAT EXCHANGER ==========")

        print(f"Total heat load:        {self.Q_total:,.2f} W")
        print(f"Exchanger type:         {self.exchanger_type}")
        print(f"Time step:              {self.dt:.2f} s")

        print("\n--- Side 1: Data Hall / Load ---")
        print(f"Density:                {self.density_air:.3f} kg/m³")
        print(f"Specific heat capacity: {self.cp_air:.2f} J/(kg·K)")
        print(f"Volume flow rate:       {self.V_air:.4f} m³/s")
        print(f"Maximum volume flow:    {self.V_air_max:.4f} m³/s")
        print(f"Cold temperature:       {self.T_1c_0:.2f} °C")
        print(f"Hot temperature:        {self.T_1h_0:.2f} °C")

        print("\n--- Side 2: Chiller ---")
        print(f"Density:                {self.density_water:.3f} kg/m³")
        print(f"Specific heat capacity: {self.cp_water:.2f} J/(kg·K)")
        print(f"Volume flow rate:       {self.V_water:.4f} m³/s")
        print(f"Maximum volume flow:    {self.V_water_max:.4f} m³/s")
        print(f"Cold temperature:       {self.T_2c_0:.2f} °C")
        print(f"Hot temperature:        {self.T_2h_0:.2f} °C")

        print("====================================\n")

    def compute_initial_step(self):
        step_0 = HX_step()
        step_0.generate_step_properties(
            0.0,
            self.V_air,
            self.V_water,
            self.density_air,
            self.density_water,
            self.cp_air,
            self.cp_water,

            self.T_2c_0,
            self.T_2h_0,
            self.T_1c_0,
            self.T_1h_0,

            self.Q_total,
            self.exchanger_type,
            
            self.V_air_max,
            self.V_water_max,

            self.T_1h_0,
            self.T_1c_0,
            self.V_air,
            self.V_water
        )

        self.steps[self.idx] = step_0
        self.idx += 1
        
    def compute_next_step(self):
        prev_step = self.steps[self.idx - 1]

        step = HX_step()
        step.generate_step_properties(
            self.idx * self.dt,
            prev_step.V_air_new,
            prev_step.V_water_new,
            self.density_air,
            self.density_water,
            self.cp_air,
            self.cp_water,

            T_chws[self.idx],
            prev_step.T_2h,
            prev_step.T_1c,
            T_ha[self.idx],

            prev_step.q_actual_new,
            self.exchanger_type,

            self.V_air_max,
            self.V_water_max,

            self.T_1h_0,
            self.T_1c_0,
            self.V_air,
            self.V_water
        )

        self.steps[self.idx] = step
        self.idx += 1

    def export_data(self):
        self.data = pd.DataFrame([vars(step) for step in self.steps.values()])
        self.data.to_excel("ThermalData.xlsx", index = False, sheet_name = 'HX')

    def run(self):
        self.compute_initial_step()

        while self.idx < 2000:
            self.compute_next_step()

        self.export_data()

class A_step():
    def __init__(self, _V_cv: float, _dt: float, _T_h_prev: float, _T_in: float, _V_in: float):
        a_i = 1.0 - (_V_in * _dt) / _V_cv
        b_i = _V_in / _V_cv

        self.V_in  = _V_in                                              # Just for data exporting purposes
        self.T_in  = _T_in                                              # Just for data exporting purposes
        self.T_mix = a_i * _T_h_prev + b_i * _T_in                      # Mixed control volume temperature
        self.T_out = self.T_mix                                         # Assumed to be the same as the mixed temperature

class Aisle():
    steps    = {}
    step_idx = 0
    
    def __init__(self, _V_cv: float, _dt: float, _T_h_0: float, _T_in_0: float, _V_in_0: float, T_A_in, V_A_in, _aisle_name: str):
        self.aisle_name = _aisle_name

        self.V_cv   = _V_cv
        self.dt     = _dt
        self.T_h_0  = _T_h_0
        self.T_in_0 = _T_in_0
        self.V_in   = _V_in_0

        self.T_A_in = T_A_in
        self.V_A_in = V_A_in

        step_0 = A_step(self.V_cv, self.dt, self.T_h_0, self.T_in_0, self.V_in)

        self.steps[self.step_idx] = step_0
        self.step_idx += 1

    def compute_next_step(self):
        step = A_step(self.V_cv, self.dt, self.steps[self.step_idx - 1].T_mix, self.T_A_in[self.step_idx - 1], self.V_A_in[self.step_idx - 1])

        self.steps[self.step_idx] = step 
        self.step_idx += 1

    def export_data(self):
        self.data = pd.DataFrame([vars(step) for step in self.steps.values()])
        self.data.to_excel("ThermalData.xlsx", index = False, sheet_name = self.aisle_name)

    def run(self):
        while self.step_idx < 2000:
            self.compute_next_step()

        self.export_data()

class ITR_step:
    def __init__(self, _V_in: float, _T_in: float, _Q: float, _density: float, _cp: float, _dt: float):
        self.V_in  = _V_in
        self.T_in  = _T_in
        self.Q     = _Q

        self.T_out = self.T_in + _dt * self.Q / (_cp * _density * self.V_in)   

class ITRacks():
    steps = {}
    step_idx = 0
    
    def __init__(self, _dt: float, _Q: float, _T_in: float, _V_in: float, _density: float, _cp: float):
        self.dt          = _dt
        self.Q           = _Q
        self.T_in        = _T_in
        self.V_in        = _V_in

        self.density     = _density
        self.cp          = _cp

        step_0 = ITR_step(self.V_in, self.T_in, self.Q, self.density, self.cp, self.dt)

        self.steps[self.step_idx] = step_0
        self.step_idx += 1

    def compute_next_step(self):
        step = ITR_step(V_ITR_in[self.step_idx], T_ITR_in[self.step_idx], self.Q, self.density, self.cp, self.dt)

        self.steps[self.step_idx] = step
        self.step_idx += 1

    def export_data(self):
        self.data = pd.DataFrame([vars(step) for step in self.steps.values()])
        self.data.to_excel("ThermalData.xlsx", index = False, sheet_name = "AC IT Racks")

    def run(self):
        while self.step_idx < 2000:
            self.compute_next_step()
        
        self.export_data()

# HX_sim = HeatExchanger(
#         6804.7,            # in kW
#         "Counter-current",
#         1.163,
#         997.3,
#         1.006,
#         4.16,

#         519.293952535859,
#         0.205021707649114,

#         24.0,
#         32.0,
#         30.5,
#         41.7,

#         608.024253447501,
#         0.240053191694498,
#         1.0
# )
# HX_sim.run()
# print(HX_sim.print_properties())
# print(HX_sim.data)

# HA_sim = Aisle(3784.125, 1.0, 41.7, 41.7, 519.29, T_HA_in, V_HA_in, "HA")
# HA_sim.run()
# print(HA_sim.data)

# CA_sim = Aisle(3123.645, 1.0, 30.5, 30.5, 519.29, T_CA_in, V_CA_in, "CA")
# CA_sim.run()
# print(CA_sim.data)

ITR_sim = ITRacks(1.0, 6804.7, 30.5, 519.29, 1.163, 1.006)
ITR_sim.run()