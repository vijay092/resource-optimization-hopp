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


# Technique:

# whenever constants come - add t 

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
    battery_power = renewable_powers["battery"]

    # Of the provided renewable generation, solve for the optimal powers
    model.used_wind_power_mw = Var(
        [i for i in range(T)],
        bounds=(1e-2, 1e6),
        initialize=10,
    )

    model.used_solar_power_mw = Var(
        [i for i in range(T)],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    model.used_battery_power_mw = Var(
        [i for i in range(T)],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    model.solar_size_mw = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    model.wind_size_mw = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    model.battery_size_mw = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    model.electrolyzer_input_mw = Var(
        [i for i in range(T)],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    # Aggregate cost
    model.annualized_cost_dlr = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    # Aggregate h2 production
    model.h2_kg = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    model.electrolyzer_size_mw = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )

    # Parameter for linearization

    model.eps = Param(initialize=1, mutable=True)


    # Charnes cooper:

    model.t = Var(
        [0],
        bounds=(1e-2, 1e8),
        initialize=10,
    )




    def obj(model):
        return model.annualized_cost_dlr[0] + model.eps * model.h2_kg[0]

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
        )  # [$/kg-H2]

        # Stack replacement costs - only paid in year of replacement

        # for i in range(len(electrolyzer_refurbishment_cost_USD)):
        #     electrolyzer_refurbishment_cost_USD[refturb_period + i * refturb_period] = (
        #         stack_rep_perc * electrolyzer_CapEx_USD
        #     )

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

        num = num - 1557220007051.7997997

        return model.annualized_cost_dlr[0] == num

    def physical_constraint_F_tot(model):
        """Denominator"""
        F_tot = 0
        for t in range(T):
            F_tot = (
                F_tot
                + 0.0145 * (model.electrolyzer_input_mw[t])
                + 0.3874 * model.electrolyzer_size_mw[0] / 0.5
            )

        return model.h2_kg[0] == F_tot / (16555294.323648022)

    def load_balance_constraint(model, t):
        """Input to the electrolyzer"""
        return (
            model.used_wind_power_mw[t]
            + model.used_solar_power_mw[t]
            + model.used_battery_power_mw[t]
        ) == model.electrolyzer_input_mw[t]

    def solar_power_constraint(model, t):
        """Used power should be less than the given power."""
        return model.used_solar_power_mw[t] <= solar_power[t]

    def wind_power_constraint(model, t):
        """Used power should be less than the given power."""
        return model.used_wind_power_mw[t] <= wind_power[t]

    def battery_power_constraint(model, t):
        """Used power should be less than the given power."""
        return model.used_battery_power_mw[t] <= battery_power[t]

    def solar_size_constraint(model, t):
        """Instead of imposing a max constarint, use this trick."""
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
        model.pwr_constraints.add(load_balance_constraint(model, t))
        model.pwr_constraints.add(solar_power_constraint(model, t))
        model.pwr_constraints.add(wind_power_constraint(model, t))
        model.pwr_constraints.add(battery_power_constraint(model, t))
        model.pwr_constraints.add(solar_size_constraint(model, t))
        model.pwr_constraints.add(wind_size_constraint(model, t))
        model.pwr_constraints.add(battery_size_constraint(model, t))
        model.pwr_constraints.add(electrolyzer_size_constraint(model, t))

    model.physical_constraints.add(physical_constraint_F_tot(model))
    model.physical_constraints.add(physical_constraint_AC(model))

    model.objective = Objective(expr=obj(model), sense=minimize)
    eps = 10
    solver = SolverFactory("ipopt")
    j = 1
    while eps > 1e-5:
        start = time.process_time()
        results = solver.solve(model)
        model.eps = value(
            model.annualized_cost_dlr[0] / model.h2_kg[0]
        )  # optimal value

        eps = (
            model.annualized_cost_dlr[0].value - model.eps.value * model.h2_kg[0].value
        )
        print("epsilon", eps)
        j = j + 1

    return model
