import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
# import yaml
parent_path = os.path.abspath("") + "/"

cost_years = [2025, 2030, 2035]

bol_eff_curve = pd.read_pickle(parent_path + "pem_1mw_bol_eff_pickle")
cost_df = pd.read_csv(parent_path + "costs_per_year.csv", index_col="Unnamed: 0")
# below are unused right now
pipeline_compressor_cost_data = pd.read_csv(
    parent_path + "Pipeline and compressor sv 01.csv", index_col=None, header=0
)
compressor_cost_data = pipeline_compressor_cost_data.loc[
    pipeline_compressor_cost_data["Technology"] == "GH2 Pipeline Compressor"
].drop(labels=["Index"], axis=1)
pipeline_cost_data = pipeline_compressor_cost_data.loc[
    pipeline_compressor_cost_data["Technology"] == "GH2 Pipeline (transmission)"
].drop(labels=["Index"], axis=1)


bol_eff_kWh_pr_kg = bol_eff_curve["Efficiency [kWh/kg]"].values
h2_bol = bol_eff_curve["H2 Produced"].values
bol_power_input_kWh = bol_eff_curve["Power Sent [kWh]"].values
bol_load_perc = bol_power_input_kWh / 1000
eol_eff_drop_perc = 0.1
eol_eff_kWh_pr_kg = bol_eff_kWh_pr_kg * (1 + eol_eff_drop_perc)

h2_eol = bol_power_input_kWh / (eol_eff_kWh_pr_kg)

##### Constants #####
battery_opex_perc = 0.025  # [% of CapEx]
pem_opex_kWyr = 12.8  # [$/kW-year]
pem_VOM = (
    1.3  # [$/MWh] - depends on average efficiency [kWh/kg-H2], like a feedstock cost!
)
indirect_electrolyzer_costs_percent = 0.42
# pem_overnight_cost_mult = 1.42 #multiply by capex to get overnight _cost_kW
# Other financial params
# desal cost depends on installed electrolyzer capacity
desal_opex_MWyr = 1340.6880555555556 * (10 / 55.5)  # *electrolyzer_size_mw
desal_capex_MW = 9109.810555555556 * (10 / 55.5)  # *electrolyzer_size_mw

# compressor cost depends on installed electrolyzer capacity
compressor_capex_kW = 39  # electrolyzer size

# hydrogen storage capacity depends on max deviation from mean hydrogen production
storage_cost_USDprkg = 17.164317691925053  # [$/kg-H2 for storage]

# stack replacement depends on degradation
stack_rep_perc = 0.15
discount_rate = 0.0824
plant_life = 30
water_cost = 0.004  # [$/gal]
water_usage_gal_pr_kg_h2 = 10 / 3.79  # gal H2O/kg-H2
# wind_losses = (100-12.83)/100 #[%]: PySAM default wind losses
# below are for grid prices, used for h2 transmission, unused right now!
ref_year = 2021
average_grid_retail_rate = 11.1 / 100  # $/kWh for 2021
price_inc_per_year = 1.2 / 1000  # $[$/kWh/year] from $1/MWh/year average
# elec_price=(price_inc_per_year*(cost_year-ref_year)) + average_grid_retail_rate

turb_size_mw = 6
max_wind_farm_capac_MW = turb_size_mw * 300
max_battery_storage_hours = 4

target_annual_hydrogen_kg = 66000 * 1000
cluster_size_MW = 40
estimated_pem_CF = 0.9
rated_h2_per_hour_1MWstack = h2_eol[-1]  # 18.2
hourly_kgh2_target = target_annual_hydrogen_kg / (estimated_pem_CF * 8760)
required_electrolyzer_capacity_MW = (
    hourly_kgh2_target / rated_h2_per_hour_1MWstack
)  # ~502
num_clusters = np.ceil(required_electrolyzer_capacity_MW / cluster_size_MW)  # 13
electrolyzer_size_MW = cluster_size_MW * num_clusters  # 520 MW
# ^good oversizing for eol

# Electrolyzer Fixed OpEx costs (has to be paid annually)
electrolyzer_FOpEx_USD = (electrolyzer_size_MW * 1000) * pem_opex_kWyr  # [$/year]
# Electrolyzer BOS Component costs - depend on electrolyzer installed capacity
desal_CapEx_USD = desal_capex_MW * electrolyzer_size_MW
desal_OpEx_USD = desal_opex_MWyr * electrolyzer_size_MW
compressor_capex_USD = compressor_capex_kW * (electrolyzer_size_MW * 1000)
