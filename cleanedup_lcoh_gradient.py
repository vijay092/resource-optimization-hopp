import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# import yaml
parent_path = os.path.abspath('') + '/'

cost_years=[2025,2030,2035]

bol_eff_curve = pd.read_pickle(parent_path + 'pem_1mw_bol_eff_pickle')
cost_df=pd.read_csv(parent_path + 'costs_per_year.csv',index_col='Unnamed: 0')
#below are unused right now
pipeline_compressor_cost_data = pd.read_csv(parent_path +'Pipeline and compressor sv 01.csv',index_col = None,header = 0)
compressor_cost_data = pipeline_compressor_cost_data.loc[pipeline_compressor_cost_data['Technology'] == 'GH2 Pipeline Compressor'].drop(labels = ['Index'],axis=1)
pipeline_cost_data = pipeline_compressor_cost_data.loc[pipeline_compressor_cost_data['Technology'] == 'GH2 Pipeline (transmission)'].drop(labels = ['Index'],axis=1)


bol_eff_kWh_pr_kg = bol_eff_curve['Efficiency [kWh/kg]'].values
h2_bol = bol_eff_curve['H2 Produced'].values
bol_power_input_kWh = bol_eff_curve['Power Sent [kWh]'].values
bol_load_perc=bol_power_input_kWh/1000
eol_eff_drop_perc = 0.1
eol_eff_kWh_pr_kg = bol_eff_kWh_pr_kg*(1+eol_eff_drop_perc)

h2_eol = bol_power_input_kWh/(eol_eff_kWh_pr_kg)

##### Constants #####
battery_opex_perc = 0.025 #[% of CapEx]
pem_opex_kWyr = 12.8 #[$/kW-year]
pem_VOM = 1.3 #[$/MWh] - depends on average efficiency [kWh/kg-H2], like a feedstock cost!
indirect_electrolyzer_costs_percent = 0.42
# pem_overnight_cost_mult = 1.42 #multiply by capex to get overnight _cost_kW
#Other financial params
#desal cost depends on installed electrolyzer capacity
desal_opex_MWyr = 1340.6880555555556 * (10/55.5) #*electrolyzer_size_mw
desal_capex_MW = 9109.810555555556 * (10/55.5) #*electrolyzer_size_mw

#compressor cost depends on installed electrolyzer capacity
compressor_capex_kW = 39 #electrolyzer size

#hydrogen storage capacity depends on max deviation from mean hydrogen production
storage_cost_USDprkg = 17.164317691925053 #[$/kg-H2 for storage]

#stack replacement depends on degradation
stack_rep_perc=0.15
discount_rate = 0.0824
plant_life = 30
water_cost = 0.004 #[$/gal]
water_usage_gal_pr_kg_h2 = 10/3.79 #gal H2O/kg-H2
# wind_losses = (100-12.83)/100 #[%]: PySAM default wind losses
#below are for grid prices, used for h2 transmission, unused right now!
ref_year = 2021
average_grid_retail_rate = 11.1/100 #$/kWh for 2021
price_inc_per_year = 1.2/1000 # $[$/kWh/year] from $1/MWh/year average
# elec_price=(price_inc_per_year*(cost_year-ref_year)) + average_grid_retail_rate

turb_size_mw=6
max_wind_farm_capac_MW = turb_size_mw*300
max_battery_storage_hours = 4

target_annual_hydrogen_kg = 66000*1000
cluster_size_MW = 40
estimated_pem_CF = 0.9
rated_h2_per_hour_1MWstack = h2_eol[-1] #18.2
hourly_kgh2_target = target_annual_hydrogen_kg/(estimated_pem_CF*8760)
required_electrolyzer_capacity_MW = hourly_kgh2_target/rated_h2_per_hour_1MWstack #~502
num_clusters = np.ceil(required_electrolyzer_capacity_MW/cluster_size_MW) #13
electrolyzer_size_MW = cluster_size_MW*num_clusters #520 MW
#^good oversizing for eol

#Electrolyzer Fixed OpEx costs (has to be paid annually)
electrolyzer_FOpEx_USD = (electrolyzer_size_MW*1000)*pem_opex_kWyr #[$/year]
#Electrolyzer BOS Component costs - depend on electrolyzer installed capacity
desal_CapEx_USD = desal_capex_MW*electrolyzer_size_MW
desal_OpEx_USD = desal_opex_MWyr*electrolyzer_size_MW
compressor_capex_USD = compressor_capex_kW*(electrolyzer_size_MW*1000)

def quick_lcoh_approx(wind_size_MW,solar_size_MW,battery_size_MW,battery_storage_hours,\
    annual_H2,hydrogen_storage_capacity_kg,stack_life_hrs,elec_avg_consumption_kWhprkg,\
    cost_year,pem_cost_case='Mod 18'):

    water_feedstock_per_year = (water_cost*water_usage_gal_pr_kg_h2*annual_H2)
    electrolyzer_refurbishment_cost_USD = np.zeros(plant_life)
    
    #degradation only impacts stack replacement cost if time between replacement is different by a year!
    refturb_period =int(np.floor(stack_life_hrs/(24*365)))
    
    year_costs = cost_df.loc[cost_year]
    pem_capex_kW = year_costs[pem_cost_case + ': PEM CapEx [$/kW]']

    #Renewable Energy Plant CapEx
    wind_CapEx_USD = (wind_size_MW*1000)*year_costs['Wind CapEx [$/kW]']#wind_capex_kW
    solar_CapEx_USD = (solar_size_MW*1000)*year_costs['Solar CapEx [$/kW]']#solar_capex_kW
    #doube check below
    battery_CapEx_kW = ((year_costs['Battery CapEx [$/kWh]']*battery_storage_hours)+year_costs['Battery CapEx [$/kW]'])
    battery_CapEx_USD = (battery_size_MW*1000)*battery_CapEx_kW

    #Electrolyzer CapEx - only changes on cost case and cost year!
    electrolyzer_CapEx_USD = (electrolyzer_size_MW*1000)*pem_capex_kW*(1+indirect_electrolyzer_costs_percent)

    #Renewable Energy Plant Fixed OpEx (has to be paid annually)
    wind_OpEx_USD = (wind_size_MW*1000)*year_costs['Wind OpEx [$/kW-year]']#*wind_opex_kWyr
    solar_OpEx_USD = (solar_size_MW*1000)*year_costs['Solar OpEx [$/kW-year]']#
    battery_OpEx_USD = battery_opex_perc*battery_CapEx_USD

    #Electrolyzed Variable OpEx costs [$/kg-H2] (treated like a feedstock)
    electrolyzer_VOpEx_USD =pem_VOM*elec_avg_consumption_kWhprkg/1000 #[$/kg-H2]
    
    #Stack replacement costs - only paid in year of replacement
    electrolyzer_refurbishment_cost_USD[refturb_period:plant_life:refturb_period]=stack_rep_perc*electrolyzer_CapEx_USD
    

    #Hydrogen Supplemental Costs - depend on electrolyzer performance!
    #1) Hydrogen Storage: 
    hydrogen_storage_CapEx_kg = 17.164317691925053 #[$/kg] TODO: update with cost scaling
    hydrogen_storage_CapEx_USD = hydrogen_storage_CapEx_kg*hydrogen_storage_capacity_kg
    
    #2) Hydrogen Transport: NOTE: not included right now
    elec_price=(price_inc_per_year*(cost_year-ref_year)) + average_grid_retail_rate #[$/kWh]
    elec_usage_for_h2_transmission = 0.5892
    h2_trans_CapEx_USD = 0 #NOTE: not included right now
    h2_trans_OpEx_USD = 0

    #CapEx costs in [$]
    hybrid_plant_CapEx_USD = wind_CapEx_USD + solar_CapEx_USD + battery_CapEx_USD
    electrolyzer_and_BOS_CapEx_USD = electrolyzer_CapEx_USD + desal_CapEx_USD + compressor_capex_USD + hydrogen_storage_CapEx_USD
    total_CapEx = hybrid_plant_CapEx_USD + electrolyzer_and_BOS_CapEx_USD #+ h2_trans_CapEx_USD
    
    #OpEx Costs
    hybrid_plant_OpEx_USD = wind_OpEx_USD + solar_OpEx_USD + battery_OpEx_USD
    electrolyzer_and_BOS_OpEx_USD = electrolyzer_FOpEx_USD + desal_OpEx_USD
    
    feedstock_costs_USD = water_feedstock_per_year + electrolyzer_VOpEx_USD #Add h2 transmission electrical costs eventually
    annual_OpEx = hybrid_plant_OpEx_USD + electrolyzer_and_BOS_OpEx_USD + feedstock_costs_USD

    y=np.arange(0,plant_life,1)
    denom = (1+discount_rate)**y
    OpEx = (annual_OpEx/denom) + (electrolyzer_refurbishment_cost_USD/denom)

    hydrogen = annual_H2/denom
    lcoh = (total_CapEx + np.sum(OpEx))/np.sum(hydrogen) #[$/kg-H2]
    return lcoh


hourly_power_needed_max_MWh =hourly_kgh2_target*eol_eff_kWh_pr_kg[-1]/1000
wind_cf = 0.3
pv_cf = 0.2
perc_wind=0.5
nturbs=np.ceil(((hourly_power_needed_max_MWh*perc_wind)/wind_cf)/turb_size_mw)
power_from_solar = hourly_power_needed_max_MWh-(nturbs*turb_size_mw*wind_cf)
#below are actual inputs needed
wind_size_mw = nturbs*turb_size_mw
solar_size_mw = np.ceil(power_from_solar/pv_cf)
battery_size_mw = 100
battery_hrs = 2
avg_time_between_replacement = 80000
h2_storage_capacity_kg=electrolyzer_size_MW*h2_bol[-1]
kWh_pr_kg_avg = eol_eff_kWh_pr_kg[-4]

#typically h2_storage_capacity_kg is sized in a similar fashiong as size_hydrogen_storage,
#but the hourly hydrogen profile is needed. Generally, its auto-sized

#pem_cost_case has options of 'Mod 18' and 'Mod 19'
lcoh_per_year = []
for cost_yr in cost_years:
    
#h2 tranmission sizes and calcs are based either on
    lcoh=quick_lcoh_approx(wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,\
        target_annual_hydrogen_kg,h2_storage_capacity_kg,avg_time_between_replacement,kWh_pr_kg_avg,\
        cost_yr)
    lcoh_per_year.append(lcoh)
    print('Year {} has LCOH of {} $/kg-H2'.format(cost_yr,lcoh))




#NOTE: the below two functions are to show how these components are sized, but not used right now
def size_hydrogen_storage(hourly_h2_production):
    baseline_hydrogen_production = np.mean(hourly_h2_production)
    h2_surp_def = hourly_h2_production-baseline_hydrogen_production
    h2_SOC = np.cumsum(h2_surp_def)
    hydrogen_storage_size = np.max(h2_SOC) - np.min(h2_SOC)
def calc_hydrogen_storage_size(hourly_h2_production):
    #if storage is before transmission, then delivery rate and CF are found like below:
    before_storage_h2_delivery_rate = np.max(hourly_h2_production)*24
    before_storage_CF = np.sum(hourly_h2_production)/(electrolyzer_size_MW*h2_bol[-1]*8760)

    #if storage is AFTER transmission, then delivery rate and CF are found like below:
    #note: this is the default
    after_storage_h2_delivery_rate = np.mean(hourly_h2_production)*24
    after_storage_CF = 0.9 #end-use CF
    #below is the default assumption for GS
    hydrogen_flow_capacity_kg_day = after_storage_h2_delivery_rate 
    pipeline_length_km=50

    compressor_cost_data
    pipeline_cost_data
    compressor_capex = np.interp(hydrogen_flow_capacity_kg_day,compressor_cost_data['Nameplate kg/d'].to_numpy(),compressor_cost_data['Capital Cost [$]'].to_numpy())
    compressor_FOM_frac = np.interp(hydrogen_flow_capacity_kg_day,compressor_cost_data['Nameplate kg/d'].to_numpy(),compressor_cost_data['Fixed Operating Cost [fraction of OvernightCapCost/y]'].to_numpy())
    compressor_FOM_USD_yr = compressor_FOM_frac*compressor_capex

    pipeline_capex_perkm = np.interp(hydrogen_flow_capacity_kg_day,pipeline_cost_data['Nameplate kg/d'].to_numpy(),pipeline_cost_data['Capital Cost [$/km]'].to_numpy())
    pipeline_capex = pipeline_capex_perkm*pipeline_length_km
    pipeline_FOM_frac = np.interp(hydrogen_flow_capacity_kg_day,pipeline_cost_data['Nameplate kg/d'].to_numpy(),pipeline_cost_data['Fixed Operating Cost [fraction of OvernightCapCost/y]'].to_numpy())
    pipeline_FOM_USD_yr = pipeline_FOM_frac*pipeline_capex

    h2_trans_CapEx_USD = compressor_capex + pipeline_capex
    h2_trans_OpEx_USD = compressor_FOM_USD_yr + pipeline_FOM_USD_yr