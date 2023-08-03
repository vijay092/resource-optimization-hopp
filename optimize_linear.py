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
        max_wind = renewable_powers["wind"].max()
        max_battery = renewable_powers["battery"].max()

    solar_power = renewable_powers["solar"]
    wind_power = renewable_powers["wind"]

    # Of the provided renewable generation, solve for the optimal powers
    model.used_wind_power_mw = Var(
        [i for i in range(T)],
        bounds=(0, max_wind),
        initialize=0.2 * max_wind,
    )

    model.used_solar_power_mw = Var(
        [i for i in range(T)],
        bounds=(0, max_solar),
        initialize=0.2 * max_solar,
    )

    model.used_battery_power_mw = Var(
        [i for i in range(T)],
        bounds=(0, max_battery),
        initialize=0.2 * max_battery,
    )

    model.solar_size_mw = Var(
        [0],
        bounds=(6, max_solar),
        initialize=0.5 * max_solar,
    )

    model.wind_size_mw = Var(
        [0],
        bounds=(6, max_wind),
        initialize=0.5 * max_wind,
    )

    model.battery_size_mw = Var(
        [0],
        bounds=(6, max_battery),
        initialize=0.5 * max_battery,
    )

    model.electrolyzer_input_mw = Var(
        [i for i in range(T)],
        bounds=(0, max_battery + max_solar + max_wind),
        initialize=10,
    )

    # Aggregate cost
    model.annualized_cost_dlr = Var(
        [0],
        bounds=(0, 1e15),
        initialize=10,
    )

    # Aggregate h2 production
    model.h2_kg = Var(
        [0],
        bounds=(1e-3, 1e15),  # just so denom doesnt go to 0
        initialize=1,
    )

    model.electrolyzer_size_mw = Var(
        [0],
        bounds=(6, max_battery + max_solar + max_wind),
        initialize=0.2 * (max_battery + max_solar + max_wind),
    )

    model.eps = Param(initialize=100, mutable=True)

    def obj(model):
        return model.annualized_cost_dlr[0] - model.eps * model.h2_kg[0]

    def battery_charge_limit(model, t):
        return (
            model.used_battery_power_mw[t] - model.used_battery_power_mw[t - 1] <= alpha
        )

    def physical_constraint_AC(model):
        # Need to find aggregate values for h2:
        h2_prod_one_yr = model.h2_kg[0]

        # LCOH calculations
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

        battery_CapEx_kW = (
            year_costs["Battery CapEx [$/kWh]"] * battery_storage_hours
        ) + year_costs["Battery CapEx [$/kW]"]
        battery_CapEx_USD = (model.battery_size_mw[0] * 1000) * battery_CapEx_kW

        # Electrolyzer CapEx - only changes on cost case and cost year!
        electrolyzer_FOpEx_USD = (
            model.electrolyzer_size_mw[0] * 1000
        ) * pem_opex_kWyr  # [$/year]
        # Electrolyzer BOS Component costs - depend on electrolyzer installed capacity
        desal_CapEx_USD = desal_capex_MW * model.electrolyzer_size_mw[0]
        desal_OpEx_USD = desal_opex_MWyr * model.electrolyzer_size_mw[0]
        compressor_capex_USD = compressor_capex_kW * (
            model.electrolyzer_size_mw[0] * 1000
        )
        electrolyzer_CapEx_USD = (
            (model.electrolyzer_size_mw[0] * 1000)
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
        )  # * model.t[
        # 0
        # ]  # [$/kg-H2]

        # Hydrogen Supplemental Costs - depend on electrolyzer performance!
        # 1) Hydrogen Storage:
        hydrogen_storage_CapEx_kg = (
            17.164317691925053  # [$/kg] TODO: update with cost scaling
        )
        hydrogen_storage_CapEx_USD = hydrogen_storage_CapEx_kg * model.h2_kg[0]

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
            if (i + 1) % (refturb_period) == 0:
                refurb = stack_rep_perc * electrolyzer_CapEx_USD
            else:
                refurb = 0

            OpEx += (annual_OpEx / d) + (refurb / d)
        num = total_CapEx + OpEx

        return model.annualized_cost_dlr[0] == num

    def physical_constraint_F_tot(model):
        """Denominator"""
        F_tot = 0
        for t in range(T):
            F_tot = F_tot + 20 * (
                model.electrolyzer_input_mw[t]
            )  # this is the model Elenya sent

        return model.h2_kg[0] == F_tot

    def load_balance_constraint_pos(model, t):
        """Input to the electrolyzer"""
        return (
            model.used_wind_power_mw[t]
            + model.used_solar_power_mw[t]
            + model.used_battery_power_mw[t]
        ) == model.electrolyzer_input_mw[t]

    # min elec power should be 4MW (1/17 of 10% min)

    def solar_power_constraint(model, t):
        """Used power should be less than the given power."""
        return model.used_solar_power_mw[t] <= solar_power[t]  # * model.t[0]

    def wind_power_constraint(model, t):
        """Used power should be less than the given power."""
        return model.used_wind_power_mw[t] <= wind_power[t]  # * model.t[0]

    def battery_power_constraint(model, t):
        """Used power should be less than the given power."""
        return model.used_battery_power_mw[t] <= max_battery  # * model.t[0]

    def solar_size_constraint(model, t):
        """Instead of imposing a max constraint, use this trick."""
        return model.solar_size_mw[0] >= model.used_solar_power_mw[t]

    def wind_size_constraint(model, t):
        """Instead of imposing a max constarint, use this trick."""
        return model.wind_size_mw[0] >= model.used_wind_power_mw[t]

    def battery_size_constraint(model, t):
        """Instead of imposing a max constarint, use this trick."""
        return model.battery_size_mw[0] >= model.used_battery_power_mw[t]

    def electrolyzer_size_constraint(model, t):
        """Instead of imposing a max constarint, use this trick."""
        return model.electrolyzer_size_mw[0] >= model.electrolyzer_input_mw[t]

    model.pwr_constraints = ConstraintList()
    model.physical_constraints = ConstraintList()

    for t in range(T):
        model.pwr_constraints.add(load_balance_constraint_pos(model, t))
        model.pwr_constraints.add(solar_power_constraint(model, t))
        model.pwr_constraints.add(battery_power_constraint(model, t))
        model.pwr_constraints.add(wind_power_constraint(model, t))
        model.pwr_constraints.add(solar_size_constraint(model, t))
        model.pwr_constraints.add(wind_size_constraint(model, t))
        model.pwr_constraints.add(electrolyzer_size_constraint(model, t))
        model.pwr_constraints.add(battery_size_constraint(model, t))

    model.physical_constraints.add(physical_constraint_F_tot(model))
    model.physical_constraints.add(physical_constraint_AC(model))

    model.objective = Objective(expr=obj(model), sense=minimize)

    solver = SolverFactory("cbc")
    res = []

    def get_useful_results(model):
        res = {}
        res["used_wind"] = [model.used_wind_power_mw[t].value for t in range(T)]
        res["used_solar"] = [model.used_solar_power_mw[t].value for t in range(T)]
        res["used_bat"] = [model.used_battery_power_mw[t].value for t in range(T)]
        res["used_elec"] = [model.electrolyzer_input_mw[t].value for t in range(T)]
        res["size_wind"] = model.wind_size_mw[0].value
        res["size_solar"] = model.solar_size_mw[0].value
        res["size_elec"] = model.electrolyzer_size_mw[0].value
        res["size_battery"] = model.battery_size_mw[0].value
        res["given_wind"] = wind_power
        res["given_solar"] = solar_power
        res["LCOH"] = value(model.annualized_cost_dlr[0] / model.h2_kg[0])
        return res

    res = []
    lcoh = []

    paramspace = np.logspace(1, 2, num=10)
    # paramspace = [paramspace[9]]
    for i in range(len(paramspace)):
        model.eps = paramspace[i]
        results = solver.solve(model)
        res.append(get_useful_results(model))
        print(
            f"Optimized for eps = {model.eps.value} and iter ={i} and got LCOH = {res[-1]['LCOH']}"
        )
        lcoh.append(res[-1]["LCOH"])
    idx = lcoh.index(min(lcoh))

    plt.loglog(paramspace, lcoh, "r.-")
    plt.ylabel("LCOH")
    plt.xlabel("eps")
    plt.grid()

    return res[idx]
