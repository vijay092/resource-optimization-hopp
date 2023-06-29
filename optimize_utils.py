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
    max_wind,
    max_solar,
    max_battery,
    max_electrolyzer,
    battery_storage_hours,
    stack_life_hrs,
    cost_year,
    elec_avg_consumption_kWhprkg,
    pem_cost_case="Mod 18",
):
    """This function finds the best sizing for wind,
    solar, battery, electrolyzer."""

    model = ConcreteModel()

    # =========================== Wind stuff ===========================
    model.wind_size = Var(
        [i for i in range(T)],
        bounds=(1e-2, max_wind),
    )

    # =========================== Solar stuff ===========================
    model.solar_size = Var(
        [i for i in range(T)],
        bounds=(1e-2, max_solar),
    )

    # =========================== Battery stuff ===========================

    model.battery_size = Var(
        [i for i in range(T)],
        bounds=(1e-2, max_battery),
    )

    # =========================== Electrolyzer stuff ===========================
    model.electrolyzer_size = Var(
        [i for i in range(T)],
        bounds=(1e-2, max_electrolyzer),
    )

    model.h2_flow_rate = Var(
        [i for i in range(T)],
        bounds=(1e-2, 1e8),
    )

    model.num = Var([0], bounds=(1e-2, 1e8))
    model.den = Var([0], bounds=(1e-2, 1e8))

    def physical_constraint_AC(model):

        
        water_feedstock_per_year = (
            water_cost * water_usage_gal_pr_kg_h2 * sum(model.h2_flow_rate)
        )
        electrolyzer_refurbishment_cost_USD = np.zeros(plant_life)

        # degradation only impacts stack replacement cost if time between replacement is different by a year!
        refturb_period = int(np.floor(stack_life_hrs / (24 * 365)))

        year_costs = cost_df.loc[cost_year]
        pem_capex_kW = year_costs[pem_cost_case + ": PEM CapEx [$/kW]"]

        # Renewable Energy Plant CapEx
        wind_CapEx_USD = (model.wind_size * 1000) * year_costs[
            "Wind CapEx [$/kW]"
        ]  # wind_capex_kW
        solar_CapEx_USD = (model.solar_size * 1000) * year_costs[
            "Solar CapEx [$/kW]"
        ]  # solar_capex_kW
        # doube check below
        battery_CapEx_kW = (
            year_costs["Battery CapEx [$/kWh]"] * battery_storage_hours
        ) + year_costs["Battery CapEx [$/kW]"]
        battery_CapEx_USD = (model.battery_size * 1000) * battery_CapEx_kW

        # Electrolyzer CapEx - only changes on cost case and cost year!
        electrolyzer_CapEx_USD = (
            (electrolyzer_size_MW * 1000)
            * pem_capex_kW
            * (1 + indirect_electrolyzer_costs_percent)
        )

        # Renewable Energy Plant Fixed OpEx (has to be paid annually)
        wind_OpEx_USD = (model.wind_size * 1000) * year_costs[
            "Wind OpEx [$/kW-year]"
        ]  # *wind_opex_kWyr
        solar_OpEx_USD = (model.solar_size * 1000) * year_costs[
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
        hydrogen_storage_CapEx_USD = hydrogen_storage_CapEx_kg * sum(model.h2_flow_rate)

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
        OpEx = (annual_OpEx / denom) + (electrolyzer_refurbishment_cost_USD / denom)
        num = total_CapEx + np.sum(OpEx)
        return model.num[0] == num

    def physical_constraint_F_tot(model):
        """Denominator function"""
        F_tot = 0
        for t in range(T):
            F_tot = F_tot + (0.0145 * model.p_el[t] + 0.3874 * max_electrolyzer / 500)
        return model.den[0] == F_tot

    def obj(model):
        return model.num[0] / model.den[0]

    model.physical_constraints = ConstraintList()
    model.physical_constraints.add(physical_constraint_F_tot(model))
    model.physical_constraints.add(physical_constraint_AC(model))

    model.objective = Objective(expr=obj(model), sense=minimize)
    solver = SolverFactory("cplex")
    results = solver.solve(model)

    return results
