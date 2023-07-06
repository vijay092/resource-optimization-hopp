import os
import sys
import pandas as pd
import numpy as np
import warnings
from params import *  # import all the parameters needed.
from pyomo.environ import *
import random
import numpy as np
import matplotlib.pyplot as plt
import time


def optimize(
    T,
    renewable_powers,
    battery_storage_hours,
    stack_life_hrs,
    cost_year,
    max_values=None,
    pem_cost_case="Mod 18",
):
    model = ConcreteModel()

    if max_values is not None:
        max_solar = max_values["solar"]
        max_wind = max_values["wind"]
        max_battery = max_values["battery"]

    else:
        max_solar = renewable_powers["solar"].max()
        max_wind = renewable_powers["solar"].max()
        max_battery = renewable_powers["solar"].max()

    solar_power = renewable_powers["solar"]
    wind_power = renewable_powers["wind"]
    battery_power = renewable_powers["battery"]

    # Of the privided renewable generation, solve for the optimal powers
    # Powers that you need to solve for:
    model.used_wind_power_mw = Var(
        [i for i in range(T)],
        bounds=(6, max_wind),
    )

    model.used_solar_power_mw = Var(
        [i for i in range(T)],
        bounds=(6, max_solar),
    )

    model.used_battery_power_mw = Var(
        [i for i in range(T)],
        bounds=(6, max_solar),
    )

    model.solar_size_mw = Var(
        [0],
        bounds=(6, max_solar),
    )

    model.wind_size_mw = Var(
        [0],
        bounds=(6, max_wind),
    )

    model.battery_size_mw = Var(
        [0],
        bounds=(6, max_battery),
    )

    # h2 flow rate form the linear model
    model.h2_flow_rate_kg_per_hr = Var(
        [i for i in range(T)],
        bounds=(0, 1e8),
    )

    # Aggregate cost
    model.annualized_cost_dlr = Var([0], bounds=(0, 1e8))

    # Aggregate h2 production
    model.h2_kg = Var([0], bounds=(0, 1e8))

    # Parameter for linearization
    model.eps = Param(initialize=1, mutable=True)

    def obj(model):
        return model.annualized_cost[0] - model.eps * model.h2_kg[0]

    def physical_constraint_AC(model):
        # Need to find aggregate values for h2:
        h2_prod_one_yr = model.h2_kg

        # LCOH calculaitons
        water_feedstock_per_year = (
            water_cost * water_usage_gal_pr_kg_h2 * h2_prod_one_yr
        )
        electrolyzer_refurbishment_cost_USD = np.zeros(plant_life)
        elec_avg_consumption_kWhprkg = eol_eff_kWh_pr_kg[-4]

        # degradation only impacts stack replacement cost if time between replacement is different by a year!
        refturb_period = int(np.floor(stack_life_hrs / (24 * 365)))

        year_costs = cost_df.loc[cost_year]
        pem_capex_kW = year_costs[pem_cost_case + ": PEM CapEx [$/kW]"]

        # Renewable Energy Plant CapEx
        wind_CapEx_USD = (model.wind_size_mw[0] * 1000) * year_costs[
            "Wind CapEx [$/kW]"
        ]  # wind_capex_kW
        solar_CapEx_USD = (model.solar_size_mw[0] * 1000) * year_costs[
            "Solar CapEx [$/kW]"
        ]  # solar_capex_kW
        # doube check below
        battery_CapEx_kW = (
            year_costs["Battery CapEx [$/kWh]"] * battery_storage_hours
        ) + year_costs["Battery CapEx [$/kW]"]
        battery_CapEx_USD = (model.battery_size_mw[0] * 1000) * battery_CapEx_kW

        # Electrolyzer CapEx - only changes on cost case and cost year!
        electrolyzer_CapEx_USD = (
            (electrolyzer_size_MW * 1000)
            * pem_capex_kW
            * (1 + indirect_electrolyzer_costs_percent)
        )

        # Renewable Energy Plant Fixed OpEx (has to be paid annually)
        wind_OpEx_USD = (model.wind_size_mw[0] * 1000) * year_costs[
            "Wind OpEx [$/kW-year]"
        ]  # *wind_opex_kWyr
        solar_OpEx_USD = (model.solar_size_mw[0] * 1000) * year_costs[
            "Solar OpEx [$/kW-year]"
        ]  #
        battery_OpEx_USD = battery_opex_perc * battery_CapEx_USD

        # Electrolyzed Variable OpEx costs [$/kg-H2] (treated like a feedstock)
        electrolyzer_VOpEx_USD = (
            pem_VOM * elec_avg_consumption_kWhprkg / 1000
        )  # [$/kg-H2]

        # Stack replacement costs - only paid in year of replacement
        electrolyzer_refurbishment_cost_USD[
            refturb_period:plant_life:refturb_period
        ] = (stack_rep_perc * electrolyzer_CapEx_USD)

        # Hydrogen Supplemental Costs - depend on electrolyzer performance!
        # 1) Hydrogen Storage:
        hydrogen_storage_CapEx_kg = (
            17.164317691925053  # [$/kg] TODO: update with cost scaling
        )
        hydrogen_storage_CapEx_USD = hydrogen_storage_CapEx_kg * model.h2_kg[0]

        # # 2) Hydrogen Transport: NOTE: not included right now
        # elec_price = (
        #     price_inc_per_year * (cost_year - ref_year)
        # ) + average_grid_retail_rate  # [$/kWh]
        # elec_usage_for_h2_transmission = 0.5892
        # h2_trans_CapEx_USD = 0  # NOTE: not included right now
        # h2_trans_OpEx_USD = 0

        # CapEx costs in [$]
        hybrid_plant_CapEx_USD = wind_CapEx_USD + solar_CapEx_USD + battery_CapEx_USD
        electrolyzer_and_BOS_CapEx_USD = (
            electrolyzer_CapEx_USD
            + desal_CapEx_USD
            + compressor_capex_USD
            + hydrogen_storage_CapEx_USD
        )
        total_CapEx = (
            hybrid_plant_CapEx_USD + electrolyzer_and_BOS_CapEx_USD
        )  # + h2_trans_CapEx_USD

        # OpEx Costs
        hybrid_plant_OpEx_USD = wind_OpEx_USD + solar_OpEx_USD + battery_OpEx_USD
        electrolyzer_and_BOS_OpEx_USD = electrolyzer_FOpEx_USD + desal_OpEx_USD

        feedstock_costs_USD = (
            water_feedstock_per_year + electrolyzer_VOpEx_USD
        )  # Add h2 transmission electrical costs eventually
        annual_OpEx = (
            hybrid_plant_OpEx_USD + electrolyzer_and_BOS_OpEx_USD + feedstock_costs_USD
        )

        y = np.arange(0, plant_life, 1)
        denom = (1 + discount_rate) ** y
        OpEx = 0
        for i, d in enumerate(denom):
            OpEx += (annual_OpEx / d) + (electrolyzer_refurbishment_cost_USD[i] / d)
        num = total_CapEx + OpEx

        return num

    def physical_constraint_F_tot(model):
        """Denominator"""
        F_tot = 0
        for t in range(T):
            F_tot = (
                F_tot
                + 0.0145 * (model.electrolyzer_input_mw[t])
                + 0.3874 * electrolyzer_size_MW / 0.5
            )

        return model.h2_kg[0] == F_tot

    def load_balance_constraint(model, t):
        """Input to the electrolyzer"""
        return (
            model.used_wind_power_mw[t]
            + model.used_solar_power_mw[t]
            + model.used_battery_power_mw[t]
        )  == model.electrolyzer_input_mw[t]

    def solar_constraint(model,t):
        """Used power should be less than the given power. """
        return model.used_solar_power_mw[t] <= solar_power[t]

    def wind_constraint(model,t):
        """Used power should be less than the given power. """
        return model.used_wind_power_mw[t] <= wind_power[t]

    def battery_constraint(model,t):
        """Used power should be less than the given power. """
        return model.used_battery_power_mw[t] <= battery_power[t]


    model.pwr_constraints = ConstraintList()
    model.safety_constraints = ConstraintList()
    model.switching_constraints = ConstraintList()
    model.physical_constraints = ConstraintList()

    for t in range(T):
        model.pwr_constraints.add(power_constraint(model, t))
    model.physical_constraints.add(physical_constraint_F_tot(model))
    model.physical_constraints.add(physical_constraint_AC(model))
    model.objective = Objective(expr=obj(model), sense=minimize)
    eps = 10
    solver = SolverFactory(
        "cbc", executable="/Users/svijaysh/CCTA_2023/cbc.exe", solver_io="python"
    )
    j = 1
    while eps > 1e-3:
        start = time.process_time()
        results = solver.solve(model)
        print("time to solve", time.process_time() - start)
        model.eps = value(model.AC[0] / model.F_tot[0])  # optimal value
        eps = model.AC[0].value - model.eps.value * model.F_tot[0].value
        j = j + 1

    P = np.array([model.p[i].value for i in range(T)])

    return (
        P_tot_opt,
        P_,
        H2f,
        I_,
        Tr,
        P_wind_t,
        model.AC[0].value,
        model.F_tot[0].value,
    )
