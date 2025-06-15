from pyModbusTCP.client import ModbusClient
import pyomo.environ as pyo
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pandas as pd
import time
import requests
import sys
import os
import numpy as np

sys.path.append(r"C:\Users\nnamdi\Desktop\EMS4HP")
import Load_Predictt_function
import PV_Predictt_function
import Heat_Prediction2


###################################Temperature###################################
def helper(x):
    '''
    this helper returns the earliest ref_datetime to  a given valid_datetime
    '''

    # convert string formatted datetime to numeric date time format, thus allowing mathematical arithmetic

    x['valid_datetime'] = pd.to_datetime(x['valid_datetime'], format="%Y-%m-%dT%H:%M:%SZ")
    x['ref_datetime'] = pd.to_datetime(x['ref_datetime'], format="%Y-%m-%dT%H:%M:%SZ")

    # find the index of the ref_datetime that is closest to the given valid_datetime
    indx = (x['valid_datetime'].values[0] - x['ref_datetime'].values).argmin(0)

    # ---------------------
    # for hours 6,12,18 24 when the weather model runs the last reftime should be used and in case it is the first hour we should keep it
    t1 = pd.Timestamp(x['valid_datetime'].values[0])
    if t1.hour == 0 or t1.hour == 6 or t1.hour == 12 or t1.hour == 18:
        indx = indx - 1
        if indx == -1:
            indx = 0
    # convert back the  numeric datetime to string format and return the latest ref_datetime
    out = x['ref_datetime'].values[indx]
    out = pd.to_datetime(str(out))
    out = out.strftime("%Y-%m-%dT%H:%M:%SZ")

    return pd.Series({'latest_ref_time': out})


###################################Heat price data###################################
HPrice_energy = [0.521, 0.521, 0.521, 0.359, 0.164, 0.100, 0.100, 0.100, 0.145, 0.359, 0.414, 0.521]  # SEK/kWh
# HPrice_effect = {'f':{9510, 14560, 27863}, 'v':{911, 861, 808}} #f: SEK/year, v: #SEK/kWh year, 0-100, 101-250, 251-500
HPrice_effect = {'f': [9510], 'v': [911]}  # f: SEK/year, v: #SEK/kWh year for 0-100 kWh thermal

# HPrice_Efficiency = 0.007 #SEK/kWh Degree of celsius for (Oct-April only)

###################################Electricity price data###################################

Subscription_fee = 147.5 / 30  # Subscription fee SEK/day
Transmission_fee = 0.255  # Electricity transmission fee SEK/kWh
Tax_fee = 0.49  # Tax fee SEK/kWh
Effect_fee = 36.25 / 30  # Effect fee SEK/kW
pv_sell_price = 0.5  # Example value for PV sell to grid price in SEK/kWh

###########################Heat system initialization##################################
ControlHPpercentage = [100]
ControlDHpercentage = [100]
ControlPHstatus = [0]


# save Control percentage
def store_control_percentages(ControlHPpercentage, ControlDHpercentage, timestamp):
    data = {
        'timestamp': [timestamp.strftime('%Y-%m-%d %H:%M:%S')],
        'ControlHPpercentage': [ControlHPpercentage],
        'ControlDHpercentage': [ControlDHpercentage]
    }
    df = pd.DataFrame(data)

    file_path = 'Control_percentages1.csv'
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


# Define maximum number of retries and delay between retries
MAX_RETRIES = 20
RETRY_DELAY = 30  # seconds
start_time = datetime.now()
end_time = start_time + timedelta(hours=24)


def fetch_temperature_data():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            Startdate = datetime.now().strftime('%Y-%m-%d')
            Enddate = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

            url = "https://api.rebase.energy/weather/v2/query"
            headers = {"Authorization": 'YCERPUQ7VUj0mPmM1reB6PrVBhORfP6iKAvpaZUDmDQ'}
            params = {
                'model': 'DWD_ICON-EU',
                'start-date': Startdate,
                'end-date': Enddate,
                'reference-time-freq': '6H',
                'forecast-horizon': 'full',
                'latitude': '57.688684',
                'longitude': '11.977383',
                'variables': 'Temperature'
            }
            response = requests.get(url, headers=headers, params=params, verify=False)
            response.raise_for_status()  # Raises HTTPError for bad responses
            result = response.json()
            df = pd.DataFrame.from_dict(result)
            latest_ref_time = df.groupby(['valid_datetime']).apply(lambda x: helper(x))
            Temperature = latest_ref_time.reset_index().merge(df, left_on=['valid_datetime', 'latest_ref_time'],
                                                              right_on=['valid_datetime', 'ref_datetime'], how='left')
            Temperature = Temperature[Temperature['valid_datetime'].apply(
                lambda x: datetime.now().strftime('%Y-%m-%dT%H:00:00Z') <= x <= (
                        datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%dT%H:00:00Z'))]['Temperature']
            Temperature.reset_index(drop=True, inplace=True)
            return Temperature
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Failed to fetch temperature data. Attempt {retries} of {MAX_RETRIES}. Error: {e}")
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def fetch_vp_data(client, register):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            data = client.read_input_registers(register, 1)
            if data:
                return data[0] / 10
            else:
                raise ValueError(f"Failed to fetch data from register {register}")
        except (ValueError, Exception) as e:
            retries += 1
            print(f"Failed to fetch data from register {register}. Attempt {retries} of {MAX_RETRIES}. Error: {e}")
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


from pytz import timezone


def fetch_spot_price_data(dt_start, dt_end):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Convert start and end times to timezone-aware pandas objects
            cet = timezone('CET')
            dt_start = pd.Timestamp(dt_start)
            dt_end = pd.Timestamp(dt_end)

            # Check if the datetime objects are timezone-aware
            if dt_start.tzinfo is None:
                dt_start = dt_start.tz_localize(cet)
            else:
                dt_start = dt_start.tz_convert(cet)

            if dt_end.tzinfo is None:
                dt_end = dt_end.tz_localize(cet)
            else:
                dt_end = dt_end.tz_convert(cet)

            api_key = 'bbb72679-072b-4616-9fe3-15eda0051f1f'
            client = EntsoePandasClient(api_key)
            DA_prices = client.query_day_ahead_prices('SE_3', start=dt_start, end=dt_end)

            eur_to_sek = 10.86  # EUR to SEK conversion rate on 2022-09-19
            mwh_to_kwh = 1000
            DA_prices = DA_prices * eur_to_sek / mwh_to_kwh
            return DA_prices
        except Exception as e:
            retries += 1
            print(f"Failed to fetch spot price data. Attempt {retries} of {MAX_RETRIES}. Error: {e}")
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


###########################Data reading initialization##################################
flag = 0  # Set this to an initial value, e.g., 0 for success or 1 for failure


# Function to expand hourly PV values into 5-minute intervals for 168 hours
def update_hourly_PV(hourly_pv, start_time):
    total_intervals = len(hourly_pv) * 12  # 168 hours * 12 intervals per hour

    # Create timestamps for 5-minute intervals over 168 hours
    timestamps_5min = pd.date_range(start=start_time, periods=total_intervals, freq='5T')

    # Repeat each hourly value exactly 12 times (one hour's worth of 5-minute intervals)
    pv_5min = np.repeat(hourly_pv, 12)

    # Create DataFrame with timestamps and PV values
    DataAsli = pd.DataFrame({
        'timestamp': timestamps_5min,
        'predictedValue': pv_5min
    })

    return DataAsli


# Example usage
if __name__ == "__main__":
    start_time = datetime.now()  # Start time for the simulation
    end_time = start_time + timedelta(hours=24)  # Define simulation duration (168 hours)

    # Hourly PV data spanning 168 hours
    hourly_pv = [
        0, 0, 0, 0, 0, 0, 0, 0, 6.032210445, 6.205327901, 6.303559402, 6.579425162, 6.714481832, 6.206474592, 6.083393445, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]

    # Ensure we have exactly 168 values
    if len(hourly_pv) != 24:
        raise ValueError(f"Expected 168 hourly values, but got {len(hourly_pv)}")


# Function to expand hourly PV values into 5-minute intervals for 168 hours
def update_hourly_load(hourly_load, start_time):
    total_intervals = len(hourly_load) * 12  # 168 hours * 12 intervals per hour

    # Create timestamps for 5-minute intervals over 168 hours
    timestamps_5min = pd.date_range(start=start_time, periods=total_intervals, freq='5T')

    # Repeat each hourly value exactly 12 times (one hour's worth of 5-minute intervals)
    load_5min = np.repeat(hourly_load, 12)

    # Create DataFrame with timestamps and PV values
    DataAsli1 = pd.DataFrame({
        'timestamp': timestamps_5min,
        'predictedValue': load_5min
    })

    return DataAsli1


# Example usage
if __name__ == "__main__":
    start_time = datetime.now()  # Start time for the simulation
    end_time = start_time + timedelta(hours=24)  # Define simulation duration (168 hours)

    # Hourly PV data spanning 168 hours
    hourly_load = [
        15.24780571, 13.59931549, 12.04811558, 11.56048062, 11.29668141, 10.40088915, 10.89595563, 12.64637674, 11.71651681, 15.95085111, 15.45798309, 21.73132253, 19.15784517, 18.63192767, 15.99398049, 12.23204608, 13.67431385, 15.11488006, 14.72023594, 13.50335915, 13.64267402, 13.96484351, 14.52849709, 16.10987106
    ]

    # Ensure we have exactly 168 values
    if len(hourly_load) != 24:
        raise ValueError(f"Expected 168 hourly values, but got {len(hourly_load)}")


# Function to expand hourly PV values into 5-minute intervals for 168 hours
def update_hourly_heat(hourly_heat, start_time):
    total_intervals = len(hourly_heat) * 12  # 168 hours * 12 intervals per hour

    # Create timestamps for 5-minute intervals over 168 hours
    timestamps_5min = pd.date_range(start=start_time, periods=total_intervals, freq='5T')

    # Repeat each hourly value exactly 12 times (one hour's worth of 5-minute intervals)
    heat_5min = np.repeat(hourly_heat, 12)

    # Create DataFrame with timestamps and PV values
    DataAsli2 = pd.DataFrame({
        'timestamp': timestamps_5min,
        'predictedValue': heat_5min
    })

    return DataAsli2


# Example usage
if __name__ == "__main__":
    start_time = datetime.now()  # Start time for the simulation
    end_time = start_time + timedelta(hours=24)  # Define simulation duration (168 hours)

    # Hourly PV data spanning 168 hours
    hourly_heat = [
        25, 19, 17, 13, 25, 17, 20, 16, 16, 15, 17, 7, 19, 18, 21, 12, 14, 9, 10, 18, 17, 11, 15, 14
    ]

    # Ensure we have exactly 168 values
    if len(hourly_heat) != 24:
        raise ValueError(f"Expected 168 hourly values, but got {len(hourly_heat)}")

# Main loop (every 5 minutes for 24 hours)
while datetime.now() < end_time:
    start_time_iteration = datetime.now()
    StartTimeControl = start_time_iteration.replace(second=0, microsecond=0)
    EndTimeControl = StartTimeControl + timedelta(minutes=5)
    print(
        f'Start to make decision for time interval between {StartTimeControl} and {EndTimeControl}')  # print the time slot for scheduling
    # Fetch data for this interval
    try:
        Temperature = fetch_temperature_data()
    except Exception as e:
        print(f"Failed to fetch temperature data. Error: {e}")
        flag = 1
        continue  # Proceed to the next iteration

    # ModbusTCP client interaction with retry
    client = ModbusClient(host="81.236.14.11", port=1606, unit_id=127)
    client.open()

    try:
        VP1_GT1_PV = fetch_vp_data(client, 2001)
        VP2_GT1_PV = fetch_vp_data(client, 2037)
    finally:
        client.close()
    # Write the values to the heat system
    # client = ModbusClient(host="81.236.14.11", port=1606, unit_id=127)
    # client.open()
    # write the maximum for the heat pump
    # client.write_single_register(3818, int(ControlHPpercentage[0]))
    # write the maximum for the district heating system
    # client.write_single_register(3817, int(ControlDHpercentage[0]))
    # write the preheat ON/OFF status
    # client.write_single_register(3819, int(ControlPHstatus[0]))
    # client.close()

    print(ControlHPpercentage[0])
    print(ControlDHpercentage[0])
    print(ControlPHstatus[0])

    # Read historical Electrical load data
    tag = 'HSBLL_Load'
    Startdate = datetime.now().strftime('%Y-%m-%d %H')
    Enddate = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H')

    # Generate the full 168-hour PV, Load, and Heat output at 5-minute intervals
    HSBpvOutput_df = update_hourly_PV(hourly_pv, start_time)
    HSBloadOutput_df = update_hourly_load(hourly_load, start_time)
    HSBheatOutput_df = update_hourly_heat(hourly_heat, start_time)

    # Expand hourly data into 5-minute intervals for the next 24 hours
    Temperature5min = []
    HSBelecLoad5min = []
    HSBpvOutput5min = []
    HSBheatLoad5min = []

    # Determine the current hour index
    current_hour_index = (datetime.now() - start_time).seconds // 3600

    for i in range(24):
        temp = Temperature.iloc[i] if i < len(Temperature) else np.nan

        # Ensure the current hour index is within bounds
        if 0 <= current_hour_index + i < len(hourly_pv):
            pv_output = hourly_pv[current_hour_index + i]
            load_output = hourly_load[current_hour_index + i]
            heat_output = hourly_heat[current_hour_index + i]
        else:
            pv_output = 0
            load_output = 0
            heat_output = 0

        # Expand into 5-minute intervals
        for j in range(12):
            Temperature5min.append(temp)
            HSBelecLoad5min.append(load_output)
            HSBpvOutput5min.append(pv_output)
            HSBheatLoad5min.append(heat_output)

    # Ensure lists are exactly 288 entries (24 hours * 12 intervals per hour)
    while len(Temperature5min) < 288:
        Temperature5min.append(np.nan)
    while len(HSBelecLoad5min) < 288:
        HSBelecLoad5min.append(np.nan)
    while len(HSBpvOutput5min) < 288:
        HSBpvOutput5min.append(np.nan)
    while len(HSBheatLoad5min) < 288:
        HSBheatLoad5min.append(np.nan)

    # save varaibles
    Temperature5min = pd.Series(Temperature5min)
    # spot_price5min = pd.Series(spot_price5min)
    HSBelecLoad5min = pd.Series(HSBelecLoad5min)
    HSBpvOutput5min = pd.Series(HSBpvOutput5min)
    HSBheatLoad5min = pd.Series(HSBheatLoad5min)

    Temperature5min.to_csv("Temperature5min.csv", index=False)
    # spot_price5min.to_csv("spot_price5min.csv", index=False)
    HSBelecLoad5min.to_csv("HSBelecLoad5min.csv", index=False)
    HSBpvOutput5min.to_csv("HSBpvOutput5min.csv", index=False)
    HSBheatLoad5min.to_csv("HSBheatLoad5min.csv", index=False)

    print('24-hour forecasts are completed')
    # COP of HPs
    HP1_COP = [
        8.909 + 0.2122 * Temperature5min[t] - 0.1823 * VP1_GT1_PV + 0.0009874 * Temperature5min[t] * Temperature5min[
            t] - 0.002603 * Temperature5min[t] * VP1_GT1_PV + 0.001055 * VP1_GT1_PV * VP1_GT1_PV for t in range(288)]
    HP2_COP = [
        8.909 + 0.2122 * Temperature5min[t] - 0.1823 * VP2_GT1_PV + 0.0009874 * Temperature5min[t] * Temperature5min[
            t] - 0.002603 * Temperature5min[t] * VP2_GT1_PV + 0.001055 * VP2_GT1_PV * VP2_GT1_PV for t in range(288)]

    HP_COP = [(HP1_COP[t] + HP2_COP[t]) / 2 for t in range(288)]

    ############################# Optimization Model ###################################
    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.Set(initialize=list(range(0, 288)))  # 288 time periods (5 minutes each)

    # Parameters
    model.pHPmax = 5  # HP's maximum electricity limit (kW)
    model.hHPmax = 18  # HP's maximum heat limit (kW)
    model.hDHmax = 32  # DH's maximum heat limit (kW)
    model.Pgrid_sellmax = 18  # PV's maximum sell to the grid limit (kW)
    max_boiler_capacity = 3  # kW
    boiler_efficiency = 0.9  # Efficiency (assumed)
    P_batterymax = 7.2  # Battery capacity

    # Variables
    model.hDH = pyo.Var(model.T, bounds=(0, model.hDHmax), within=pyo.NonNegativeReals, initialize=0)
    model.hHP = pyo.Var(model.T, bounds=(0, model.hHPmax), within=pyo.NonNegativeReals, initialize=0)
    model.pHP = pyo.Var(model.T, bounds=(0, model.pHPmax), within=pyo.NonNegativeReals, initialize=0)
    model.P_battery = pyo.Var(model.T, bounds=(0, P_batterymax), within=pyo.NonNegativeReals, initialize=0)
    model.Ppv_use = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0)
    model.pGrid = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
    model.Pgrid_sell = pyo.Var(model.T, bounds=(0, model.Pgrid_sellmax), within=pyo.NonNegativeReals, initialize=0)
    model.boiler_heat = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0, max_boiler_capacity))
    model.SOC_battery = pyo.Var(model.T, bounds=(0, P_batterymax), within=pyo.NonNegativeReals, initialize=P_batterymax) # Start Full
    model.P_battery_discharge = pyo.Var(model.T, bounds=(0, P_batterymax), within=pyo.NonNegativeReals, initialize=0)


    # Objective Function: Maximize Self-Consumption
    def rule_O1_multi(model):
        return sum(model.Ppv_use[t] + model.P_battery[t] for t in model.T) - 0.2 * sum(model.pGrid[t] for t in model.T)


    model.obj = pyo.Objective(rule=rule_O1_multi, sense=pyo.maximize)


    # Constraints
    def rule_C1(model, t):
        return model.hHP[t] + model.hDH[t] + model.boiler_heat[t] == HSBheatLoad5min[t]


    model.C1 = pyo.Constraint(model.T, rule=rule_C1)


    # Ensures PV is used first for electrical load
    def rule_C2(model, t):
        return model.Ppv_use[t] == min(HSBelecLoad5min[t], HSBpvOutput5min[t])


    model.C2 = pyo.Constraint(model.T, rule=rule_C2)

    # Introduce an auxiliary variable to avoid max() function
    model.pGrid_aux = pyo.Var(model.T, within=pyo.Reals)


    # Constraint to define pGrid_aux
    def rule_C3_aux(model, t):
        return model.pGrid_aux[t] == HSBelecLoad5min[t] - (model.Ppv_use[t] + model.P_battery_discharge[t])


    model.C3_aux = pyo.Constraint(model.T, rule=rule_C3_aux)


    # Ensure that pGrid is non-negative
    def rule_C3_final(model, t):
        return model.pGrid[t] >= model.pGrid_aux[t]


    model.C3_final = pyo.Constraint(model.T, rule=rule_C3_final)


    def rule_C4(model, t):
        return model.hDH[t] <= model.hDHmax


    model.C4 = pyo.Constraint(model.T, rule=rule_C4)


    # Heat Pump Efficiency
    def rule_C8(model, t):
        return model.hHP[t] == model.pHP[t] * HP_COP[t]


    model.C8 = pyo.Constraint(model.T, rule=rule_C8)


    # Ensures SOC updates based on battery power use
    def rule_battery_SOC(model, t):
        if t == 0:
            return model.SOC_battery[t] == 0  # Assume battery starts empty
        return model.SOC_battery[t] == model.SOC_battery[t - 1] + model.P_battery_discharge[t] * 0.9 - (
                    model.P_battery_discharge[
                        t] / 0.9)  # Discharging with efficiency loss# Battery efficiency assumed at 90%


    model.battery_SOC = pyo.Constraint(model.T, rule=rule_battery_SOC)


    # Charge battery when PV > Load and battery is not full
    def rule_battery_charge(model, t):
        return model.P_battery[t] <= max(0, HSBpvOutput5min[t] - HSBelecLoad5min[t])


    model.battery_charge = pyo.Constraint(model.T, rule=rule_battery_charge)


    # Ensure the battery discharges when load exceeds PV generation
    def rule_battery_discharge(model, t):
        return model.P_battery_discharge[t] <= HSBelecLoad5min[t] - model.Ppv_use[t]


    model.battery_discharge = pyo.Constraint(model.T, rule=rule_battery_discharge)


    # Ensure battery discharge does not exceed available SOC
    def rule_battery_SOC_limit(model, t):
        return model.P_battery_discharge[t] <= model.SOC_battery[t]  # Battery cannot discharge more than stored energy


    model.battery_SOC_limit = pyo.Constraint(model.T, rule=rule_battery_SOC_limit)


    # Ensures the battery does not overcharge
    def rule_battery_capacity(model, t):
        return model.SOC_battery[t] <= P_batterymax


    model.battery_capacity = pyo.Constraint(model.T, rule=rule_battery_capacity)

    # Solving the model
    opt = pyo.SolverFactory('glpk')
    opt.options = {'mipgap': 0.001}  # Optional: Can set the gap tolerance for MIP problems
    results = opt.solve(model)

    # Check if the model was solved successfully
    if (results.solver.status == pyo.SolverStatus.ok) and (
            results.solver.termination_condition == pyo.TerminationCondition.optimal):
        ControlHPpercentage = [
            max(0, round(pyo.value(model.pHP[t]) / model.pHPmax * 100, 2)) for t in model.T
        ]
        ControlDHpercentage = [
            max(0, round(pyo.value(model.hDH[t]) / model.hDHmax * 100, 2)) for t in model.T
        ]

        print(f'Solution status: {results.solver.status}')
        print(f'Objective value: {pyo.value(model.obj)}')
    else:
        print(f'Solver status: {results.solver.status}')
        print(f'Termination condition: {results.solver.termination_condition}')
        print('Model could not be solved correctly.')


    # Save optimization results
    def save_optimization_results(model, output_file="optimization_resultsWINTER.csv"):
        """
        Save only the first element of the optimized variables (Ppv_use, pGrid, P_battery, HSBelecLoad5min) to a CSV file with a timestamp.
        """
        # Validate model attributes
        if not hasattr(model, 'T') or not hasattr(model, 'Ppv_use') or not hasattr(model, 'pGrid'):
            raise ValueError("The provided model does not have the required attributes 'T', 'Ppv_use', or 'pGrid'.")

        try:
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Extract the first time step
            first_time = next(iter(model.T))  # Get the first time step in the model

            # Retrieve the values for the first time step
            first_pv_use = pyo.value(model.Ppv_use[first_time])
            first_pgrid = pyo.value(model.pGrid[first_time])
            first_hsbelec_load = HSBelecLoad5min[first_time]  # Assuming HSBelecLoad5min is a pre-defined variable
            first_hsbheat_load = HSBheatLoad5min[first_time]
            first_pv_output = HSBpvOutput5min[first_time]
            first_P_battery = pyo.value(model.P_battery[first_time])
            # Prepare the data for saving
            result = {
                "Timestamp": [timestamp],
                "Ppv_use": [first_pv_use],
                "pGrid": [first_pgrid],
                "P_battery": [first_P_battery],
                "HSBpvOutput5min": [first_pv_output],
                "HSBelecLoad5min": [first_hsbelec_load],
                "HSBheatLoad5min": [first_hsbheat_load]
            }

            # Convert to a DataFrame
            df_result = pd.DataFrame(result)

            # Check if the file exists and append to it, otherwise create a new file
            if os.path.exists(output_file):
                df_result.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_result.to_csv(output_file, index=False)

            print(f"First element of optimization results saved to: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"Error saving optimization results: {e}")


    # Call the save function
    save_optimization_results(model)

    # Store control percentages
    store_control_percentages(ControlHPpercentage[-1], ControlDHpercentage[-1], datetime.now())

    # Sleep before the next iteration (if running in a loop)
    time.sleep(300)
