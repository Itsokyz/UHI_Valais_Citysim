# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:18:03 2021

@author: Roberto


Modified on Sat Jan 20 11:17:47 2024
@author: Zetong Liu
"""

from xml.etree.ElementTree import Element, SubElement, tostring, parse
import math

from xml.dom.minidom import parseString
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as sg
import joblib
import re
import os
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, MultiPoint
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint
from shapely.ops import triangulate
from shapely.plotting import plot_polygon, plot_points
import matplotlib.pyplot as plt
from shapely.ops import unary_union

'''
TO DO:
    
    - thank Pierre
    - unify egid, ssid

'''

def prettify(element):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(element, 'utf-8')
    reparsed = parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def write_xml_file(root, filename):
    text = prettify(root)
    with open(filename, "w") as f:
        f.write(text)
    # Remove empty lines.
    # filedata = ""
    # with open(filename, "r") as infile:
    #     for line in infile.readlines():
    #         if line.strip():  # if striped line is non empty
    #             filedata += line
    # with open(filename, "w") as outfile:
    #     outfile.write(filedata)

def get_district_child_from_xml(xml_file, tag):
    tree = parse(xml_file)
    root = tree.getroot()
    file_district = root.find('District')
    elements = file_district.findall(tag)
    return elements


def add_child_from_xml_to_district(district, xml_file, tag):
    elements = get_district_child_from_xml(xml_file, tag)
    for element in elements:
        district.append(element)


def add_composites_from_xml(district, xml_file):
    add_child_from_xml_to_district(district, xml_file, 'Composite')


def add_far_fields_obstructions_from_xml(district, xml_file):
    add_child_from_xml_to_district(district, xml_file, 'FarFieldObstructions')


def add_profiles_from_xml(district, xml_file):
    add_child_from_xml_to_district(district, xml_file, 'Composite')
    add_child_from_xml_to_district(district, xml_file, 'OccupancyDayProfile')
    add_child_from_xml_to_district(district, xml_file, 'OccupancyYearProfile')
    add_child_from_xml_to_district(district, xml_file, 'DeviceType')
    add_child_from_xml_to_district(district, xml_file, 'ActivityType')
    add_child_from_xml_to_district(district, xml_file, 'DHWDayProfile')
    add_child_from_xml_to_district(district, xml_file, 'DHWYearProfile')


def add_root():
    citysim = Element("CitySim")
    return citysim


def add_simulation_days(citysim, begin_month=1, begin_day=1,
                        end_month=12, end_day=31):

    dict_simulation = {"beginMonth": str(begin_month),
                       "beginDay": str(begin_day),
                       "endMonth": str(end_month),
                       "endDay": str(end_day)}
    simulation = SubElement(citysim, "Simulation", dict_simulation)
    return simulation


def add_climate(citysim, filename):
    dict_climate = {"location": filename}
    climate = SubElement(citysim, "Climate", dict_climate)


def add_district(citysim):
    district = SubElement(citysim, "District")
    return district

def add_far_field_obstructions(district, horizon_df):
    far_field_obstructions = SubElement(district, "FarFieldObstructions")
    for i in horizon_df.index:
        row = horizon_df.loc[i]
        phi = row['phi']
        theta = row['theta']
        dict_horizon = {"phi": str(phi), "theta": str(theta)}
        horizon = SubElement(far_field_obstructions, "Point", dict_horizon)


def add_composite(district, composite_id, composite_name, layers_df):
    dict_composite = {"id": str(composite_id), "name": composite_name}
    composite = SubElement(district, "Composite", dict_composite)
    for i in layers_df.index:
        row = layers_df.loc[i]
        name = row['name']
        thickness = row['thickness']
        conductivity = row['conductivity']
        cp = row['cp']
        density = row['density']
        dict_layer = {"Name": name, "Thickness": str(thickness),
                      "Conductivity": str(conductivity), "Cp": str(cp),
                      "Density": str(density)}
        layer = SubElement(composite, "Layer", dict_layer)


def add_building(district, row, volume, simulate=True, tmin=21, tmax=26, 
                 blinds_lambda=0.2,blinds_irradiance_cutoff=150):
    # Building characteristics
    egid = row['egid']
    bid = row['bid']
    ventilation_rate = row['Ninf']

    dict_building = {"id": str(bid), "key": str(egid), "Ninf": str(ventilation_rate),
                     "Vi": str(volume), "Tmin": str(tmin), "Tmax": str(tmax),
                     "BlindsLambda": str(blinds_lambda),
                     "BlindsIrradianceCutOff": str(blinds_irradiance_cutoff),
                     "Simulate": str(simulate).lower()}

    building = SubElement(district, "Building", dict_building)
    return building


def add_heat_tank(building, v=50*1e-3, phi=200, rho=1000, cp=4180, tmin=20,
                  tmax=35, tcrit=90):

    dict_tank = {"V": str(v), "phi": str(phi), "rho": str(rho),
                 "Cp": str(cp), "Tmin": str(tmin), "Tmax": str(tmax),
                 "Tcritical": str(tcrit)}

    heat_tank = SubElement(building, "HeatTank", dict_tank)
    return heat_tank


def add_dhw_tank(building, v=0.2, phi=2.5, rho=1000, cp=4180, tmin=60,
                 tmax=70, tcrit=90, tinlet=5):

    dict_tank = {"V": str(v), "phi": str(phi), "rho": str(rho),
                 "Cp": str(cp), "Tmin": str(tmin), "Tmax": str(tmax),
                 "Tcritical": str(tcrit), "Tinlet": str(tinlet)}

    dhw_tank = SubElement(building, "DHWTank", dict_tank)
    return dhw_tank


def add_cool_tank(building, v=20, phi=20, rho=1000, cp=4180, tmin=5,
                  tmax=20):

    dict_tank = {"V": str(v), "phi": str(phi), "rho": str(rho),
                 "Cp": str(cp), "Tmin": str(tmin), "Tmax": str(tmax)}

    cool_tank = SubElement(building, "CoolTank", dict_tank)
    return cool_tank


def add_heat_source(building, begin_day=1, end_day=365):
    '''
    begin_day : The default is January 1st.
    end_day : The default is December 31st.
    '''
    
    dict_heat_source = {"beginDay": str(begin_day), "endDay": str(end_day)}
    heat_source = SubElement(building, "HeatSource", dict_heat_source)
    return heat_source


def add_boiler(heat_source, pmax=50000, eta_th=0.96):
    dict_boiler = {"name": "Boiler", "Pmax": str(pmax), "eta_th": str(eta_th)}
    boiler = SubElement(heat_source, "Boiler", dict_boiler)
    return boiler


def add_zone(building, net_volume, zone_id=0, psi=0.2, ground_floor=True):

    dict_zone = {"id": str(zone_id), "volume": str(net_volume),
                 "Psi": str(psi), "groundFloor": str(ground_floor).lower()}

    zone = SubElement(building, "Zone", dict_zone)
    return zone


def add_imposed_heat_demand(building, values, start_day=1, start_hour=1,
                            end_day=365, end_hour=24):

    count = 0
    demand_dict = {}
    for i in range(start_day, end_day + 1):
        if start_day == end_day:
            daily_hours_range = range(start_hour, end_hour + 1)
        elif i == start_day and start_day != end_day:
            daily_hours_range = range(start_hour, 25)
        elif i == end_day and start_day != end_day:
            daily_hours_range = range(1, end_hour + 1)
        else:
            daily_hours_range = range(1, 25)
        for j in daily_hours_range:
            key = 'd{}h{}'.format(i, j)
            value = values.loc[count]
            demand_dict[key] = str(value)
            count += 1
    heat_demand = SubElement(building, "ImposedHeatDemand", demand_dict)
    return heat_demand


def add_occupants(zone, number_of_occupants=5, building_type=1, activity_type=None,
                  dhw_type=None, stochastic=False):

    dict_occupants = {"n": str(number_of_occupants),
                      "type": str(building_type), "Stochastic": str(stochastic).lower(),
                      # "activityType":  str(activity_type),
                      # "DHWType": str(dhw_type)
                      }
    if activity_type:
        dict_occupants["activityType"] = str(activity_type)
    if dhw_type:
        dict_occupants["DHWType"] = str(dhw_type)
    occupants = SubElement(zone, "Occupants", dict_occupants)
    return occupants

def add_surfaces(zone, envelope, center_coordinates):
    x_center = center_coordinates[0]
    y_center = center_coordinates[1]
    for r in envelope.index:
        row = envelope.loc[r]
        geometry = row['geometry']
        class_id = row['class_id']
        glazing_ratio = row["glazing_ratio"]
        glazing_u_value = row["glazing_u_value"]
        glazing_g_value = row["glazing_g_value"]
        openable_ratio = row["openable_ratio"]
        shortwave_reflectance = row["shortwave_reflectance"]
        surface_type = int(row["surface_type"])

        if class_id == 33:
            dict_surface = {"id": str(r), "type": str(surface_type)}
            surface = SubElement(zone, "Floor", dict_surface)

        elif class_id == 34:
            dict_surface = {"id": str(r), "type": str(surface_type),
                            "ShortWaveReflectance": str(shortwave_reflectance),
                            "GlazingRatio": str(glazing_ratio),
                            "GlazingUValue": str(glazing_u_value),
                            "GlazingGValue": str(glazing_g_value),
                            "OpenableRatio": str(openable_ratio)}
            surface = SubElement(zone, "Wall", dict_surface)

        elif class_id == 35:
            dict_surface = {"id": str(r), "type": str(surface_type  ),
                            "ShortWaveReflectance": str(shortwave_reflectance),
                            "GlazingRatio": str(glazing_ratio),
                            "GlazingUValue": str(glazing_u_value),
                            "GlazingGValue": str(glazing_g_value),
                            "OpenableRatio": str(openable_ratio)}
            surface = SubElement(zone, "Roof", dict_surface)
            
        else:
            raise ValueError("Surface class not understood.")

        # Add points translated to center of the scene
        for p in range(len(geometry.exterior.coords)-1):
            point_name = "V{}".format(p)
            coordinates = geometry.exterior.coords[p]
            x = str(coordinates[0]-x_center)
            y = str(coordinates[1]-y_center)
            z = str(coordinates[2])

            dict_point = {"x": x, "y": y, "z": z}
            point = SubElement(surface, point_name, dict_point)
            
def add_surfaces_pedestrian(zone, pedestrian_envelope, center_coordinates):
    x_center = center_coordinates[0]
    y_center = center_coordinates[1]
    for r in pedestrian_envelope.index:
        row = pedestrian_envelope.loc[r]
        geometry = row['geometry']
        class_id = row['class_id']
        
        glazing_ratio = 0
        glazing_u_value = 0
        glazing_g_value = 0
        openable_ratio = 0
        k_factor = 0
        shortwave_reflectance = 0.37
        long_wave_emissivity = 0.95
        surface_type = 9 # pedestrian surface in composite

        if class_id == 1:
            dict_surface = {"id": str(r), "type": str(surface_type),
                            "ShortWaveReflectance": str(shortwave_reflectance),
                            "LongWaveEmissivity": str(long_wave_emissivity),
                            "GlazingRatio": str(glazing_ratio),
                            "GlazingUValue": str(glazing_u_value),
                            "GlazingGValue": str(glazing_g_value),
                            "OpenableRatio": str(openable_ratio)}
            surface = SubElement(zone, "Floor", dict_surface)

        elif class_id == 3:
            dict_surface = {"id": str(r), "type": str(surface_type),
                            "ShortWaveReflectance": str(shortwave_reflectance),
                            "LongWaveEmissivity": str(long_wave_emissivity),
                            "GlazingRatio": str(glazing_ratio),
                            "GlazingUValue": str(glazing_u_value),
                            "GlazingGValue": str(glazing_g_value),
                            "OpenableRatio": str(openable_ratio)}
            surface = SubElement(zone, "Wall", dict_surface)

        elif class_id == 2:
            dict_surface = {"id": str(r), "type": str(surface_type),
                            "ShortWaveReflectance": str(shortwave_reflectance),
                            "LongWaveEmissivity": str(long_wave_emissivity),
                            "GlazingRatio": str(glazing_ratio),
                            "GlazingUValue": str(glazing_u_value),
                            "GlazingGValue": str(glazing_g_value),
                            "OpenableRatio": str(openable_ratio),
                            "kFactor": str(k_factor)}
            surface = SubElement(zone, "Roof", dict_surface)
            
        else:
            raise ValueError("Surface class not understood.")

        # Add points translated to center of the scene
        for p in range(len(geometry.exterior.coords)-1):
            point_name = "V{}".format(p)
            coordinates = geometry.exterior.coords[p]
            x = str(coordinates[0]-x_center)
            y = str(coordinates[1]-y_center)
            z = str(coordinates[2])

            dict_point = {"x": x, "y": y, "z": z}
            point = SubElement(surface, point_name, dict_point)

            



def AST_3(envelope, TS_grounds, TS_buildings, ground_data, zone_geom):

    AVG_T = TS_grounds.mean()
    AVG_T = pd.DataFrame({'T': AVG_T.values}, index=AVG_T.index)
    AVG_T['surface_id'] = AVG_T.index
    AVG_T['surface_id'] = AVG_T['surface_id'].astype(str)

    merged_T_grounds = pd.merge(AVG_T , ground_data[['gid', 'geometry']], left_on='surface_id', right_on='gid', how='left')
    merged_T_grounds = merged_T_grounds.drop(columns='surface_id')
    merged_T_grounds['bid'] = np.nan
    grounds_AST = gpd.GeoDataFrame(merged_T_grounds, geometry='geometry', crs='EPSG:2056')

    weighted_avg_T_list = []
    bid_list = [] 
    TS_buildings = TS_buildings.mean()
    column_series = pd.Series(TS_buildings.index).astype(str)
    pattern = re.compile(r'^(\d+)\(\d+\)')
    building_id = column_series.apply(lambda x: pattern.match(x).group(1) if pattern.match(x) else None).dropna().unique()
    z_g_copy = zone_geom.copy()
    z_g_copy['bid'] = z_g_copy['bid'].astype(str)
    
    for r in building_id:
        single_building_cols = TS_buildings.index[TS_buildings.index.str.extract(r'^(\d+)\(\d+\):(\d+)\(\)')[0] == str(r)]
        single_building_temps = TS_buildings[single_building_cols]
        surface_ids = single_building_cols.str.extract(r':(\d+)\(')[0].astype(int)
        valid_surface_ids = surface_ids[surface_ids.isin(envelope.index)]
        
        if not valid_surface_ids.empty:
            m2 = envelope.loc[valid_surface_ids]['geometry']
            s1 = m2.area
            valid_temps = [single_building_temps[single_building_cols[surface_ids == sid].values[0]] for sid in valid_surface_ids]
            T = pd.Series(valid_temps)
            mul = s1.values * T
            n = mul.sum()
            m = s1.sum()
            weighted_avg_T = n / m
            weighted_avg_T_list.append(weighted_avg_T)
            bid_list.append(r)  # Append building id
        else:
            print(f"No valid surface_ids found for building id {r}")
            weighted_avg_T_list.append(np.nan)
            bid_list.append(r)

    buildings_AST = pd.DataFrame({'T': weighted_avg_T_list, 'bid': bid_list})
    buildings_AST['bid'] = buildings_AST['bid'].astype(str)
    buildings_AST = buildings_AST.merge(z_g_copy[['bid', 'geometry']], on='bid', how='left')
    buildings_AST['gid'] = np.nan
    buildings_AST = gpd.GeoDataFrame(buildings_AST, geometry='geometry', crs='EPSG:2056')
    concatenated_TS_df = pd.concat([buildings_AST, grounds_AST], ignore_index=True)
    result_gdf = gpd.GeoDataFrame(concatenated_TS_df, geometry='geometry')

    return grounds_AST, buildings_AST, result_gdf


def AST_baseline_3(envelope, TS_grounds, ground_data):

    AVG_T = TS_grounds.mean()
    AVG_T = pd.DataFrame({'T': AVG_T.values}, index=AVG_T.index)
    AVG_T['surface_id'] = AVG_T.index
    AVG_T['surface_id'] = AVG_T['surface_id'].astype(str)

    merged_T_grounds = pd.merge(AVG_T , ground_data[['gid', 'geometry']], left_on='surface_id', right_on='gid', how='left')
    merged_T_grounds = merged_T_grounds.drop(columns='surface_id')
    merged_T_grounds['bid'] = np.nan
    grounds_AST = gpd.GeoDataFrame(merged_T_grounds, geometry='geometry', crs='EPSG:2056')

    return grounds_AST


def T_sol_air_2(TS_grounds, LW_grounds, SW_grounds, SWA_df, hc_df):
    SWA_df['ID'] = SWA_df['ID'].astype('int64')
    
    Tsat_df = pd.DataFrame(index=TS_grounds.index, columns=TS_grounds.columns, dtype=float)
    for gid in TS_grounds.columns:
        Ts = TS_grounds[gid]
        SW = SW_grounds[gid]
        LW = LW_grounds[gid]
        hc = hc_df[gid]
        alpha_value = SWA_df[SWA_df['ID'] == gid].iloc[0, 1]  # Accès à la valeur alpha pour cette colonne
    
        # Application de la formule ligne par ligne
        for i in range(len(TS_grounds)):
            Tsat_df.at[i, gid] = Ts[i] + (LW[i] + alpha_value * SW[i]) / hc[i]

    return Tsat_df


def clean_sort_OUT_files(subdirectory_path, xml_name, file_name):
    data_file = os.path.join(subdirectory_path, xml_name+ file_name)
    data_df = pd.read_csv(data_file, delimiter='\t', encoding='latin1')
    data_df = data_df.dropna(axis=1)
    columns_without_ke = [col for col in data_df.columns if 'Ke' not in col]
    filtered_columns = data_df[columns_without_ke]

    # TS for building
    filtered_columns_no_NA = filtered_columns.loc[:, ~filtered_columns.columns.str.contains('NA')]
    data_buildings = filtered_columns_no_NA.loc[:, filtered_columns_no_NA.columns.str.contains(r'\(\d+\):')]

    # TS for ground
    data_grounds = filtered_columns.loc[:, filtered_columns.columns.str.contains('NA')]
    surface_id = data_grounds.columns.to_series().str.split(':', expand=True)[1].str.split('(', expand=True)[0].astype(int)
    data_grounds.columns = surface_id
    
    return data_grounds, data_buildings


def hottest_day(climate_file):
    
    Climatic_df = pd.read_csv(climate_file, delimiter='\t', skiprows=3)
    Climatic_df = Climatic_df.dropna(axis=1)
    Climatic_df.columns = Climatic_df.columns.str.strip()
    
    # Hottest day in year
    daily_TA_mean_df = Climatic_df.groupby(['m', 'dm'])['Ta'].mean().reset_index()
    daily_TA_mean_df = daily_TA_mean_df.rename(columns={'Ta': 'Ta_mean'})
    TA_mean_max_index = daily_TA_mean_df['Ta_mean'].idxmax()
    
    # Tmax in hottest day
    max_temp_month = daily_TA_mean_df.loc[TA_mean_max_index, 'm']
    max_temp_day = daily_TA_mean_df.loc[TA_mean_max_index, 'dm']
    max_temp_day_data = Climatic_df[(Climatic_df['m'] == max_temp_month) & (Climatic_df['dm'] == max_temp_day)]
    max_temp_index = max_temp_day_data['Ta'].idxmax()
    max_temp_values = max_temp_day_data.loc[max_temp_day_data['Ta'].idxmax()]
    
    return Climatic_df, max_temp_values, max_temp_index 

def add_hd_data_gpkg_2(max_temp_index, Ta_max_values, ground_data, Tsat_df, TS_wb_df, SW_wb_df, LW_wb_df, hc_wb_df, SWA_wb_df, VF_wb_df):
    
    # Select Tsat line to be represented in QGIS
    Tsat_ground_1hour = ground_data
    
    Tsat_ground_1hour['gid'] = Tsat_ground_1hour['gid'].astype('int64')
    
    Tsat_ground_1hour['bid'] = np.nan
    Tsat_ground_1hour['Tsat'] = np.nan
    Tsat_ground_1hour['Ts'] = np.nan
    Tsat_ground_1hour['SW'] = np.nan
    Tsat_ground_1hour['LW'] = np.nan
    Tsat_ground_1hour['SWA'] = np.nan
    Tsat_ground_1hour['hc'] = np.nan
    #Tsat_ground_1hour['Elevation [°]'] = np.nan
    #Tsat_ground_1hour['Azimuth [°]'] = np.nan
    
    # Select row corresponding to TAIR max
    row_Tsatm_max = Tsat_df.iloc[max_temp_index]
    row_Ts_max = TS_wb_df.iloc[max_temp_index]
    row_SW_max = SW_wb_df.iloc[max_temp_index]
    row_LW_max = LW_wb_df.iloc[max_temp_index]
    row_hc_max = hc_wb_df.iloc[max_temp_index]    
    
    for gid in Tsat_ground_1hour['gid']:
        if gid in row_Tsatm_max.index:
            #print('correspondance)
            Tsat_ground_1hour.loc[Tsat_ground_1hour['gid'] == gid, 'Tsat'] = row_Tsatm_max[gid]
            Tsat_ground_1hour.loc[Tsat_ground_1hour['gid'] == gid, 'Ts'] = row_Ts_max[gid]
            Tsat_ground_1hour.loc[Tsat_ground_1hour['gid'] == gid, 'SW'] = row_SW_max[gid]
            Tsat_ground_1hour.loc[Tsat_ground_1hour['gid'] == gid, 'LW'] = row_LW_max[gid]
            Tsat_ground_1hour.loc[Tsat_ground_1hour['gid'] == gid, 'hc'] = row_hc_max[gid]

    swa_values = SWA_wb_df.set_index('ID')['ShortWaveAbsorptance']
    Tsat_ground_1hour['SWA'] = Tsat_ground_1hour['gid'].map(swa_values)
    
    # Meteorological data
    Tsat_ground_1hour['max_temp_index'] = max_temp_index
    Tsat_ground_1hour['Ta_max [°C]'] = Ta_max_values['Ta']
    Tsat_ground_1hour['Tg [°C]'] = Ta_max_values['Ts']
    Tsat_ground_1hour['G_DH [W/m2]'] = Ta_max_values['G_Dh']
    Tsat_ground_1hour['G_BN [W/m2]'] = Ta_max_values['G_Bn']
    Tsat_ground_1hour['Wind speed [m/s]'] = Ta_max_values['FF']
    Tsat_ground_1hour['Wind direction [°]'] = Ta_max_values['DD']
    Tsat_ground_1hour['RH [%]'] = Ta_max_values['RH']
    Tsat_ground_1hour['Precipitation [mm]'] = Ta_max_values['RR']
    Tsat_ground_1hour['Nebulosity [octa]'] = Ta_max_values['N']
    
    month_str = Ta_max_values['m'].astype(int).astype(str)
    day_str = Ta_max_values['dm'].astype(int).astype(str)
    hour_str = Ta_max_values['h'].astype(int).astype(str)
    Tsat_ground_1hour['month.day.hour'] = month_str + ':' + day_str + ':' + hour_str
    
    # sun position
    VF_wb_selected = VF_wb_df[['gid', 'Altitude(°)', 'Azimuth(°)']]
    VF_wb_selected['gid'] = VF_wb_selected['gid'].astype(str)
    Tsat_ground_1hour['gid'] = Tsat_ground_1hour['gid'].astype(str)
        
    merged_df = Tsat_ground_1hour.merge(VF_wb_selected, on='gid', how='left')
    merged_df.rename(columns={'Altitude(°)': 'Elevation [°]', 'Azimuth(°)': 'Azimuth [°]'}, inplace=True)
        
    SAT_results_gdf = gpd.GeoDataFrame(merged_df[['geometry', 'gid', 'bid','max_temp_index', 'month.day.hour', 'Tsat', 'Ts', 'SWA', 'SW', 'LW', 'hc','Ta_max [°C]', 'Tg [°C]','G_DH [W/m2]', 'G_BN [W/m2]', 'Wind speed [m/s]', 'Wind direction [°]', 'RH [%]', 'Precipitation [mm]', 'Nebulosity [octa]', 'Elevation [°]', 'Azimuth [°]']], geometry='geometry')
    
    return SAT_results_gdf


def add_hd_data_roof_gpkg(max_temp_index, Ta_max_values, zone_geom_qgis, Tsat_r_df, TS_r_df, SW_r_df, LW_r_df, hc_r_df, SWA_r_df):
    
    # Select Tsat line to be represented in QGIS
    Tsat_roof_1hour = zone_geom_qgis
    Tsat_roof_1hour['gid'] = np.nan
    Tsat_roof_1hour['Tsat'] = np.nan
    Tsat_roof_1hour['Ts'] = np.nan
    Tsat_roof_1hour['SW'] = np.nan
    Tsat_roof_1hour['LW'] = np.nan
    Tsat_roof_1hour['SWA'] = np.nan
    Tsat_roof_1hour['hc'] = np.nan
    Tsat_roof_1hour['Elevation [°]'] = np.nan
    Tsat_roof_1hour['Azimuth [°]'] = np.nan
    
    row_Tsatm_max = Tsat_r_df.iloc[max_temp_index]
    row_Ts_max = TS_r_df.iloc[max_temp_index]
    row_SW_max = SW_r_df.iloc[max_temp_index]
    row_LW_max = LW_r_df.iloc[max_temp_index]
    row_hc_max = hc_r_df.iloc[max_temp_index]
    
    for bid in Tsat_roof_1hour['bid']:
        if bid in row_Tsatm_max.index:
            Tsat_roof_1hour.loc[Tsat_roof_1hour['bid'] == bid, 'Ts'] = row_Ts_max[bid]
            Tsat_roof_1hour.loc[Tsat_roof_1hour['bid'] == bid, 'SW'] = row_SW_max[bid]
            Tsat_roof_1hour.loc[Tsat_roof_1hour['bid'] == bid, 'LW'] = row_LW_max[bid]
        
    # Meteorological data
    Tsat_roof_1hour['max_temp_index'] = max_temp_index
    Tsat_roof_1hour['Ta_max [°C]'] = Ta_max_values['Ta']
    Tsat_roof_1hour['Tg [°C]'] = Ta_max_values['Ts']
    Tsat_roof_1hour['G_DH [W/m2]'] = Ta_max_values['G_Dh']
    Tsat_roof_1hour['G_BN [W/m2]'] = Ta_max_values['G_Bn']
    Tsat_roof_1hour['Wind speed [m/s]'] = Ta_max_values['FF']
    Tsat_roof_1hour['Wind direction [°]'] = Ta_max_values['DD']
    Tsat_roof_1hour['RH [%]'] = Ta_max_values['RH']
    Tsat_roof_1hour['Precipitation [mm]'] = Ta_max_values['RR']
    Tsat_roof_1hour['Nebulosity [octa]'] = Ta_max_values['N']
    
    month_str = Ta_max_values['m'].astype(int).astype(str)
    day_str = Ta_max_values['dm'].astype(int).astype(str)
    hour_str = Ta_max_values['h'].astype(int).astype(str)
    Tsat_roof_1hour['month.day.hour'] = month_str + ':' + day_str + ':' + hour_str
        
    SAT_results_roof_gdf = gpd.GeoDataFrame(Tsat_roof_1hour[['geometry', 'gid', 'bid','max_temp_index', 'month.day.hour', 'Tsat', 'Ts', 'SWA', 'SW', 'LW', 'hc','Ta_max [°C]', 'Tg [°C]','G_DH [W/m2]', 'G_BN [W/m2]', 'Wind speed [m/s]', 'Wind direction [°]', 'RH [%]', 'Precipitation [mm]', 'Nebulosity [octa]', 'Elevation [°]', 'Azimuth [°]']], geometry='geometry')
    
    return SAT_results_roof_gdf

def extract_roof_output_2(TS_buildings, LW_buildings, SW_buildings, envelope):
        
    # Appeler la fonction pour chaque DataFrame
    T_buildings_roof_mean = calculate_roof_average_output(TS_buildings, envelope)
    LW_buildings_roof_mean = calculate_roof_average_output(LW_buildings, envelope)
    SW_buildings_roof_mean = calculate_roof_average_output(SW_buildings, envelope)


    """Creation de faux data frame pour tsolair"""
    SWA_r_df = pd.DataFrame(0, index=T_buildings_roof_mean.index, columns=T_buildings_roof_mean.columns)
    hc_r_df = pd.DataFrame(0, index=T_buildings_roof_mean.index, columns=T_buildings_roof_mean.columns)
    Tsat_r_df = pd.DataFrame(0, index=T_buildings_roof_mean.index, columns=T_buildings_roof_mean.columns)
    
    return T_buildings_roof_mean, LW_buildings_roof_mean, SW_buildings_roof_mean, SWA_r_df, hc_r_df, Tsat_r_df


def calculate_roof_average_output(buildings_df, envelope_df, class_id_filter=35):
    # Enregistrer uniquement roof buildings correspondant au class_id_filter
    results = []

    # Itérer sur les colonnes de buildings_df
    for col in buildings_df.columns:
        # Extraire bid et sid du nom de la colonne
        match = re.match(r'(\d+)\((\d+)\):(\d+)', col)
        if match:
            bid_b = int(match.group(1))
            sid_b = int(match.group(3))

            # Itérer sur les lignes de enveloppe_df
            for sid_e, row in envelope_df.iterrows():
                bid_e = row['bid']
                class_id = row['class_id']

                # Comparer les bid et sid et vérifier class_id
                if bid_b == bid_e and sid_b == sid_e and class_id == class_id_filter:
                    # Récupérer les valeurs de la colonne de buildings_df
                    values = buildings_df[col].values
                    # Stocker les valeurs avec l'identifiant du nom de la colonne
                    results.append((col, values))

    # Convertir les résultats en DataFrame
    buildings_roof = pd.DataFrame({col: values for col, values in results})

    # Préparer un dictionnaire pour stocker les résultats
    results_mean = {}

    # Itérer sur les colonnes de buildings_roof
    for col in buildings_roof.columns:
        # Extraire bid et sid du nom de la colonne
        match = re.match(r'(\d+)\((\d+)\):(\d+)', col)
        if match:
            bid = int(match.group(1))
            sid = int(match.group(3))

            # Trouver l'aire correspondante dans envelope_df
            if bid in envelope_df['bid'].values and sid in envelope_df.index:
                area = envelope_df.loc[sid, 'geometry'].area

                # Multiplier les valeurs de la colonne par l'aire
                weighted_temp = buildings_roof[col] * area

                # Ajouter les résultats au dictionnaire
                if bid not in results_mean:
                    results_mean[bid] = {'weighted_temp': pd.Series(0, index=buildings_roof.index), 'total_area': 0}

                results_mean[bid]['weighted_temp'] += weighted_temp
                results_mean[bid]['total_area'] += area

    # Calculer la température moyenne pour chaque bid et chaque heure
    result_data = {}
    for bid in results_mean:
        weighted_temp = results_mean[bid]['weighted_temp']
        total_area = results_mean[bid]['total_area']
        mean_temp = weighted_temp / total_area
        result_data[bid] = mean_temp

    # Convertir les résultats en DataFrame
    buildings_roof_mean = pd.DataFrame(result_data)

    return buildings_roof_mean


def SWR(district):
    shortwave_dict = {}
    grounds = district.find("GroundSurface")
    for surface in grounds.iter("Ground"):
        current_id = surface.attrib["id"]
        shortwave_value = surface.attrib["ShortWaveReflectance"]
        shortwave_dict[current_id] = shortwave_value
    shortwave_df = pd.DataFrame(shortwave_dict.items(), columns=['ID', 'ShortWaveReflectance'])
    shortwave_df['ShortWaveReflectance'] = pd.to_numeric(shortwave_df['ShortWaveReflectance'], errors='coerce')
    # Absorbtance
    a_shortwave_df = shortwave_df.copy()
    a_shortwave_df.rename(columns={'ShortWaveReflectance': 'ShortWaveAbsorptance'}, inplace=True)
    a_shortwave_df['ShortWaveAbsorptance'] = 1 - a_shortwave_df['ShortWaveAbsorptance']
    return shortwave_df, a_shortwave_df

def height_meteo(climate_file):
    with open(climate_file, 'r') as file:
        lines = file.readlines()
    header = lines[1].strip()
    values = header.split(',')
    h_meteo_station = int(values[2])
    
    return h_meteo_station

def concevtion_coefficient(climate_df, ground_data, h_meteo_station):
    # Mcadams linearized
    # Windspeed pre-processing, H.B Awbi (1991)
    # prime = data for the meteorologiacl station
    
    # Urban. industrial or forest area
    alpha_prime = 0.67
    gama_prime = 0.25
    v_prime = climate_df['FF']
    h_prime = h_meteo_station + 10
    # Urban. industrial or forest area
    alpha = 0.67
    gama = 0.25

    hc_df = pd.DataFrame(columns=ground_data['gid'])
    
    for index, row in ground_data.iterrows():
        current_id = row['gid']
        
        # pre-processing wind speed
        triangle = row.geometry
        z_coords = [point[2] for point in triangle.exterior.coords]
        mean_z = sum(z_coords) / 3
        h_surface = mean_z + 2
        #hc_df[current_id] = (3*(v_prime *(alpha*(h_surface/10)**gama)/(alpha_prime*(h_prime/10)**gama_prime)).astype('float32') + 2.8).astype('float32')
        hc_values = []
        for v in v_prime:
            hc_value = (3 * (v * (alpha * (h_surface / 10) ** gama) / (alpha_prime * (h_prime / 10) ** gama_prime)) + 2.8)
            hc_values.append(hc_value)
        
        # Ajouter les valeurs calculées dans le DataFrame
        hc_df[current_id] = hc_values
        
    
    return hc_df

    

def add_ground(district, terrain_df, groundtype=1, detailedSimulation=False, ShortWaveReflectance=0.3):
    groundsurface = SubElement(district, "GroundSurface")
    for r in terrain_df.index:
        geometry = terrain_df["geometry"].loc[r]
        dict_surface = {"id": str(r),
                        "ShortWaveReflectance":str(ShortWaveReflectance),
                        "type":str(groundtype),
                        "detailedSimulation":str(detailedSimulation).lower()}

        surface = SubElement(groundsurface, "Ground", dict_surface)
        # Add points
        for p in range(len(geometry.exterior.coords)-1):
            point_name = "V{}".format(p)
            coordinates = geometry.exterior.coords[p]
            x = str(coordinates[0])
            y = str(coordinates[1])
            z = str(coordinates[2])
            
            dict_point = {"x": x, "y": y, "z": z}
            point = SubElement(surface, point_name, dict_point)
            
         


def add_ground_from_XYZ(MO_dhn, district, terrain_df, zone_box, center_coordinates=(0,0), kFactor=0.1, groundtype=2, detailedSimulation=False, ShortWaveReflectance=0.14):  
    x_center = center_coordinates[0]
    y_center = center_coordinates[1]
    groundsurface = SubElement(district, "GroundSurface")
    tri_id=0 
    n=int((len(terrain_df))**0.5)
    row_diff=[[0, n, n+1], [0, n+1, 1]] 
    max_x=terrain_df['X'].max()
    min_y=terrain_df['Y'].min()
    geometry_list = []
    id_list = []
    for r in terrain_df.index:
        if terrain_df.loc[r,'X'] < max_x and terrain_df.loc[r,'Y'] > min_y:
            for _ in range(2):
                dict_surface = {"id": str(tri_id),
                                "ShortWaveReflectance":str(ShortWaveReflectance),
                                "type":str(groundtype),
                                "kFactor": str(kFactor),
                                "detailedSimulation":str(detailedSimulation).lower()}
            # Add points
                x=[0]*3
                y=[0]*3
                z=[0]*3
                for p in range(3):
                    coordinates = terrain_df.loc[r+row_diff[tri_id%2][p]]
                    x[p] = coordinates['X']-x_center
                    y[p] = coordinates['Y']-y_center
                    z[p] = coordinates['Z']
                triangle = Polygon([(x[0]+x_center,y[0]+y_center),(x[1]+x_center,y[1]+y_center),(x[2]+x_center,y[2]+y_center)])
                if triangle.within(zone_box) and not triangle.intersects(MO_dhn.geometry.unary_union):
                    surface = SubElement(groundsurface, "Ground", dict_surface)
                    for p in range(3):
                        point_name = "V{}".format(p)
                        dict_point = {"x": str(x[p]), "y": str(y[p]), "z": str(z[p])}
                        point = SubElement(surface, point_name, dict_point)
                    coords = ((x[0]+x_center,y[0]+y_center,z[0]),(x[1]+x_center,y[1]+y_center,z[1]),(x[2]+x_center,y[2]+y_center,z[2]))
                    geometry_list.append(Polygon(coords))
                    id_list.append(tri_id)
                tri_id+=1
    gdf = gpd.GeoDataFrame(geometry=geometry_list, crs='EPSG:2056')
    gdf['gid'] = id_list
    # out_buildings = ~gdf.geometry.intersects(MO_dhn.geometry.unary_union)
    # gdf=gdf[out_buildings]
    return gdf



def add_z_to_mo(gdf_intersection, zone_geom, buildings, altitude_default=0):
    zone_geom['alt'] = int
    for i in range(len(zone_geom)):
        # Selection of the ground geometry to intersect with triangles
        if zone_geom.at[i,'envelope_type'] == '2.5D':
            current_egid = zone_geom.at[i,'egid'] 
            z_coords_list = []
            for j in range(len(gdf_intersection)):
                intersection = gdf_intersection['geometry'][j].intersection(zone_geom['geometry'][i])
                intersection_area = (gdf_intersection['geometry'][j].intersection(zone_geom['geometry'][i])).area
                if not gdf_intersection['geometry'][j].within(zone_geom['geometry'][i]):
                #if intersection_area < 2:   
                    for l in range(3):
                        point = Point(gdf_intersection['geometry'][j].exterior.coords[l])
                        if point.intersects(zone_geom['geometry'][i]):
                            z_coords_list.append(point.z)
            
            if not z_coords_list:
                print(f"Warning: The building with egid {current_egid} did not find ground for intersection. Default height {altitude_default}.")
                zmin = altitude_default
            else:
                zmin = min(z_coords_list)                
                            
            #zmin = min(z_coords_list)
            zone_geom.at[i, 'alt'] = zmin
            for s in range(len(buildings)):
                b_egid = buildings.loc[s, 'egid']
                if str(current_egid) == str(b_egid):
                    buildings.loc[s, 'altitude'] = zmin
                     
            


def reverse_polygon_coordinates(geometry):
    exterior_coords = list(geometry.exterior.coords)[::-1]
    return Polygon(exterior_coords)

# Fonction pour enlever la coordonnée z d'une géométrie
def remove_z(geometry):
    if isinstance(geometry, Polygon):
        new_exterior = [(coord[0], coord[1]) if len(coord) == 3 else (coord[0], coord[1]) for coord in geometry.exterior.coords]
        return Polygon(new_exterior)
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([remove_z(p) for p in geometry.geoms])
    else:
        return geometry
    
def remove_duplicate_coords_simple(coords):
    """Remove duplicate coordinates based on x and y values."""
    seen = set()
    unique_coords = []
    for coord in coords:
        # Check if (x, y) has been seen
        if (coord[0], coord[1]) not in seen:
            unique_coords.append(coord)
            seen.add((coord[0], coord[1]))
    return unique_coords


def remove_duplicate_coords(coords):
    seen = set()
    unique_coords = []
    for coord in coords:
        rounded_coord = (round(coord[0], 6), round(coord[1], 6))
        if rounded_coord not in seen:
            unique_coords.append(coord)
            seen.add(rounded_coord)
    return unique_coords

def clean_polygon(geometry):
    if isinstance(geometry, Polygon):
        # Remove duplicate coordinates
        new_coords = remove_duplicate_coords(list(geometry.exterior.coords))
        if len(new_coords) >= 3:
            return Polygon(new_coords)
    return None


def close_ground(ground_data, gdf_intersection, zone_geom):
    
    zone_geom_copy = zone_geom.copy()
    zone_geom_copy['geometry'] = zone_geom_copy['geometry'].apply(remove_z)
    union_geometry = unary_union(zone_geom_copy.geometry)

    results = []

    # Itérer sur chaque ligne de gdf_intersection
    for idx, intersection_row in gdf_intersection.iterrows():
        triangle = intersection_row['geometry']
        gid = intersection_row['gid']

        if triangle.intersects(union_geometry):
            intersection = triangle.intersection(union_geometry)
            if isinstance(intersection, (Polygon, MultiPolygon, Point, MultiPoint, LineString)) and not intersection.is_empty:
                difference = triangle.difference(union_geometry)
                if not difference.is_empty:
                    # Polygon area bigger than 1mm square
                    if difference.area>1e-6: 
                        # Ajouter la géométrie résultante et son identifiant à la liste des résultats
                        results.append({'gid': gid, 'geometry': difference})

    # Convertir la liste des résultats en GeoDataFrame
    fill_ground = gpd.GeoDataFrame(results, columns=['gid', 'geometry'], geometry='geometry')

    #joblib.dump(fill_ground, r'C:\Users\Giuliano\Documents\EPFL\Master\Projet de master\UHI_CH_sp-main\Fichiers_joblibs_sols\Geneve\fill_ground_3D.pkl')

    max_gid = ground_data['gid'].max()
    additional_rows = []

    for index, row in fill_ground.iterrows():
        geom = row['geometry']
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                if not poly.is_empty:
                    max_gid += 1
                    new_row = row.copy()
                    new_row['geometry'] = poly
                    new_row['gid'] = max_gid
                    additional_rows.append(new_row)
        else:
            additional_rows.append(row)

    check_fill_ground = gpd.GeoDataFrame(additional_rows, columns=fill_ground.columns, crs=fill_ground.crs)
    
    #joblib.dump(check_fill_ground, r'C:\Users\Giuliano\Documents\EPFL\Master\Projet de master\UHI_CH_sp-main\Fichiers_joblibs_sols\Geneve\check_fill_ground_3D.pkl')
    
    # supprimer les polygones avec les mêmes coordonnées
    check_fill_ground['geometry'] = check_fill_ground['geometry'].apply(clean_polygon)
    
    #joblib.dump(check_fill_ground, r'C:\Users\Giuliano\Documents\EPFL\Master\Projet de master\UHI_CH_sp-main\Fichiers_joblibs_sols\Geneve\check_fill_ground_not_same_coord_3D.pkl')
    
    # Supprimer les polygones qui sont devenus None
    check_fill_ground = check_fill_ground[check_fill_ground['geometry'].notnull()]
    
    #joblib.dump(check_fill_ground, r'C:\Users\Giuliano\Documents\EPFL\Master\Projet de master\UHI_CH_sp-main\Fichiers_joblibs_sols\Geneve\check_fill_ground_wo_none_3D.pkl')
    
    check_fill_ground['geometry'] = check_fill_ground['geometry'].apply(reverse_polygon_coordinates)

    ground_data['gid'] = ground_data['gid'].astype(int)
    check_fill_ground['gid'] = check_fill_ground['gid'].astype(int)
    check_fill_ground = check_fill_ground.rename(columns={'geometry': 'fill_geometry'})

    #MERGE explicite et propre
    gdf1 = pd.merge(ground_data, check_fill_ground, how='outer', on='gid')
    #gdf1 = pd.merge(ground_data, check_fill_ground, how='outer')
    gdf1['geometry'] = gdf1['fill_geometry']

    gdf1 = gpd.GeoDataFrame(gdf1, geometry='geometry', crs=ground_data.crs)

    return gdf1

def add_ground_in_xml(gdf, district, center_coordinates, groundtype, kFactor, ShortWaveReflectance, detailedSimulation=False, LongWaveEmissivity=None):
    x_center = center_coordinates[0]
    y_center = center_coordinates[1]
    gdf.reset_index(drop=True, inplace=True)
    groundsurface = SubElement(district, "GroundSurface")
    
    for i in range(len(gdf)):
        polygon_len = (len(gdf.at[i, 'geometry'].exterior.coords)-1)
        x=[0]*polygon_len
        y=[0]*polygon_len
        z=[0]*polygon_len
        tri_id = gdf['gid'][i]
        for p in range(polygon_len):
            x[p] = gdf['geometry'][i].exterior.coords[p][0]-x_center
            y[p] = gdf['geometry'][i].exterior.coords[p][1]-y_center
            z[p] = gdf['geometry'][i].exterior.coords[p][2]
        
        dict_surface = {"id": str(tri_id),
                        "ShortWaveReflectance":str(ShortWaveReflectance),
                        "type":str(groundtype),
                        "kFactor": str(kFactor),
                        "detailedSimulation":str(detailedSimulation).lower()} 
        
        if LongWaveEmissivity is not None:
            dict_surface["LongWaveEmissivity"] = str(LongWaveEmissivity)

        surface = SubElement(groundsurface, "Ground", dict_surface)
        for p in range(polygon_len):
            point_name = "V{}".format(p)
            dict_point = {"x": str(x[p]), "y": str(y[p]), "z": str(z[p])}
            point = SubElement(surface, point_name, dict_point)           
            
def fusion_triangles(gdf1):

    n = len(gdf1)
    a = 0
    
    for i in range(n):
        # Allows iterating over a union of polygons
        if a != 0:              
            i = i - 1
        a = 0
        for j in range(i+1, n):
            j = j - a
            trianglea = gdf1.iloc[i]['geometry']
            triangleb = gdf1.iloc[j]['geometry']
            intersection = trianglea.intersection(triangleb)
            if intersection.geom_type == 'LineString':
                normala = normal_comp(trianglea)
                normalb = normal_comp(triangleb) 
                #Normal vector parallel? 
                cross_product = np.cross(normala,normalb)
                if np.all(cross_product == 0):
                    # building the new polygon
                    polygons=[trianglea, triangleb]
                    u = unary_union(polygons)
                    u = Polygon(u.exterior.coords[::-1])
                    # modify geodataframe
                    gdf1.at[i, 'geometry'] = u
                    gdf1.drop(j, inplace=True)
                    gdf1.reset_index(drop=True, inplace=True)
                    n = len(gdf1)
                    a = a + 1 
                   
    return gdf1


def modify_type_road(district, ground_data, road_groundtype, road_kfactor, road_SWR, modify_data=None):
    
    geometry_list = []
    road_width={'Autobahn': 7, # E41 typical width of lanes is 3.5m, two lanes=7m
                'Autostrasse': 7,
                '10m Strasse': 10,
                '8m Strasse': 8,
                '6m Strasse': 6,
                '3m Strasse': 3,
                '2m Weg': 2,
                '4m Strasse': 4,
                '1m Weg': 1,
                'Provisorium': 3.0
                }
    road_index_list=[]
    str_error = 0
    if modify_data is not None:
        for index, row in modify_data.iterrows():
            line = row['geometry']
            road_type=row['objektart'] 
            buffered_line = line.buffer(road_width[road_type]/2, cap_style='flat')
            #geometry_list.append(buffered_line)
            road_ground = ground_data[ground_data['geometry'].intersects(buffered_line)]
            if road_ground is not None:
                geometry_list.append(buffered_line)
            road_indices = road_ground['gid'].values
            road_index_list += road_indices.tolist()
        road_index_list = list(set(road_index_list))
        buffered_streets = gpd.GeoDataFrame(geometry=geometry_list, crs='EPSG:2056')

    grounds = district.find("GroundSurface")
    for surface in grounds.iter("Ground"):
        current_id = surface.attrib["id"]
        
        if int(current_id) in road_index_list:
            surface.set("type", str(road_groundtype))
            surface.set("kFactor", str(road_kfactor))
            surface.set("ShortWaveReflectance", str(road_SWR))

    return road_index_list, buffered_streets



def modify_type(district, ground_data, groundtype, kfactor, SWR, LongWaveEmissivity=None, modif_data=None):
    
    geometry_list2 = []
    index_list = []
    itsctd = gpd.GeoDataFrame(geometry=[], crs='EPSG:2056')

    total_intersections = 0  # compteur d'intersections

    if modif_data is not None:
        for index, row in modif_data.iterrows():
            single_block = row['geometry']
            ground = ground_data[ground_data['geometry'].intersects(single_block)]
            n = len(ground)
            total_intersections += n  # on compte
            if n > 0:
                indices = ground['gid'].values
                index_list += indices.tolist()
                geometry_list2.append(single_block)

        index_list = list(set(index_list))
        itsctd = gpd.GeoDataFrame(geometry=geometry_list2, crs='EPSG:2056')

    # Modification du XML
    grounds = district.find("GroundSurface")
    for surface in grounds.iter("Ground"):
        current_id = surface.attrib["id"]
        if int(current_id) in index_list:
            surface.set("type", str(groundtype))
            surface.set("kFactor", str(kfactor))
            surface.set("ShortWaveReflectance", str(SWR))
            if LongWaveEmissivity is not None:
                surface.set("LongWaveEmissivity", str(LongWaveEmissivity))

    print(f"[INFO] Total intersecting ground surfaces: {total_intersections}")
    print(f"[INFO] Unique ground IDs modified in XML: {len(index_list)}")

    return index_list, itsctd

def keep_soil(district):
    grounds = district.find("GroundSurface")
    to_remove = []

    for surface in grounds.iter("Ground"):
        current_type = surface.attrib["type"]
        if int(current_type) != 37:
            to_remove.append(surface)
    
    for surface in to_remove:
        grounds.remove(surface)

def keep_green(district):
    grounds = district.find("GroundSurface")
    to_remove = []

    for surface in grounds.iter("Ground"):
        current_type = surface.attrib["type"]
        if int(current_type) != 3:
            to_remove.append(surface)
    
    for surface in to_remove:
        grounds.remove(surface)

def keep_road(district):
    grounds = district.find("GroundSurface")
    to_remove = []

    for surface in grounds.iter("Ground"):
        current_type = surface.attrib["type"]
        if int(current_type) != 2:
            to_remove.append(surface)
    
    for surface in to_remove:
        grounds.remove(surface)


def cut(district, ground_data, MO_dhn, footprints, kept_range=15):
    buffered_geometries = MO_dhn.geometry.buffer(kept_range)
    convex_hull = buffered_geometries.unary_union.convex_hull
    #remove distant grounds
    grounds = district.find("GroundSurface")
    to_remove = []
    
    not_in_convex_hull_mask = ~ground_data.geometry.within(convex_hull)
    remove_list = ground_data.loc[not_in_convex_hull_mask, 'gid'].tolist()
    #remove grounds under buildings
    # under_buildings = ground_data.geometry.intersects(MO_dhn.geometry.unary_union)
    # remove_list += ground_data.loc[under_buildings, 'gid'].tolist()

    for surface in grounds.iter("Ground"):
        current_id = surface.attrib["id"]
        if int(current_id) in remove_list:
            to_remove.append(surface)
    
    for surface in to_remove:
        grounds.remove(surface)
    #remove distant buildings
    to_remove_b=[]
    buffered_geometries_2 = MO_dhn.geometry.buffer(kept_range)
    convex_hull_2 = buffered_geometries_2.unary_union.convex_hull
    not_in_convex_hull_mask_2 = ~footprints.geometry.within(convex_hull_2)
    remove_list_2 = footprints.loc[not_in_convex_hull_mask_2, 'bid'].tolist()
    for building in district.iter("Building"):
        current_id = building.attrib["id"]
        if int(current_id) in remove_list_2:
            to_remove_b.append(building)
    
    for building in to_remove_b:
        district.remove(building)
        
def cut_baseline(district, ground_data_baseline, MO_dhn, footprints, kept_range=15):
    #remove grounds

    for ground_surface in district.findall('GroundSurface'):
        district.remove(ground_surface)
    


def add_all_buildings(district, buildings, envelope, center_coordinates=(0,0)):
    for i in buildings.index:
        row = buildings.loc[i]
        
        # Get volume from Swissbuildings3D if provided
        volume_MO = row['volume_MO']
        volume_3D = row['volume_3D']
        if volume_3D == 0:
            volume = volume_MO
        else:
            volume = volume_3D
            
        # Add building with according simulation status
        Simulate_status = row['Simulate_status']
        tmin = row['Tmin']
        if Simulate_status == True:
            building = add_building(district, row, volume, tmin=tmin, tmax=26, blinds_lambda=0.2,
                     blinds_irradiance_cutoff=150)
        else:
            building = add_building(district, row, volume, simulate=False)
        zone = add_zone(building, volume)
        
        # Activity profile according to building type (no profile for 11 sports installations)
        building_type = row['building_type']
        if building_type == 11:
            activity_type = None
        else:
            activity_type = building_type
        
        # DHW profile according to building type
        dhw_type = row['building_type']

        # Add heat tank with temperature setpoint of heat supplier depending on year of construction
        year = row['year']
        # Radiator
        if year < 1990:
            add_heat_tank(building, tmin=50, tmax=60, tcrit=90) #TODO
        # Underfloor heating
        else:
            add_heat_tank(building, tmin=35, tmax=40, tcrit=90)        
        
        # Add DHW tank according to number of occupants (0.05 m3/person, 3 m3 max.)
        n_occupants = row['n_occupants']
        if n_occupants != 0:
            vol_DHW = 50*1e-3*n_occupants
            if vol_DHW > 3:
                vol_DHW = 3   
            add_dhw_tank(building, v=vol_DHW)
        add_occupants(zone, n_occupants, building_type, activity_type, dhw_type, stochastic=False)
        
        # Add boiler as heat source of 10 MW
        heat_source = add_heat_source(building)
        add_boiler(heat_source, pmax=10e6)

        # Add building's envelope surfaces 
        bid = row['bid']
        e = envelope[envelope['bid'] == bid]
        add_surfaces(zone, e, center_coordinates)
        
        
def add_pedestrians(district, pedestrian_data, pedestrian_envelope, center_coordinates):
    for i in pedestrian_data.index:
        row = pedestrian_data.loc[i]
        
        egid = row['egid']
        bid = row['bid']
        volume = row['volume']
        
        # dict building
        ninf = 0.100000001
        tmin = 35
        tmax = 37
        blinds_lambda = 0.200000003
        blinds_irradiance_cutoff = 100
        simulate = True
        MRT = True
        MRTEPSILON = 0.95
        
        
        dict_building = {"id": str(bid), "key": str(egid), "Ninf": str(ninf), "Vi": str(volume),
                         "Tmin": str(tmin), "Tmax": str(tmax), "BlindsLambda": str(blinds_lambda),
                         "BlindsIrradianceCutOff": str(blinds_irradiance_cutoff),"Simulate": str(simulate).lower(),
                         "mrt": str(MRT).lower(), "mrtEpsilon": str(MRTEPSILON)}

        building = SubElement(district, "Building", dict_building)
    
        zone = add_zone(building, volume)
    
        # dict occupants
        number_of_occupants = 0
        activity_type = 1
        dhw_type = 1
        stochastic = False
        building_type = 0
    
        dict_occupants = {"n": str(number_of_occupants),
                          "type": str(building_type), "Stochastic": str(stochastic).lower(),
                          "activityType":  str(activity_type),
                          "DHWType": str(dhw_type)
                          }
    
        occupants = SubElement(zone, "Occupants", dict_occupants)
        pde = pedestrian_envelope[pedestrian_envelope['bid'] == bid]
        add_surfaces_pedestrian(zone, pde, center_coordinates)
    
    
    return 


def is_ccw(polygon):
    coords = list(polygon.exterior.coords)
    area = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i][:2]
        x2, y2 = coords[i + 1][:2]
        area += (x2 - x1) * (y2 + y1)
    return area < 0

def add_shading_surfaces(district, tarps_df, tarps_envelope, center_coordinates):
    """
    Ajoute chaque bâche comme <Surface> dans <ShadingSurface>,
    et duplique chaque surface avec l’ordre des sommets inversé (pour ombre dans les deux sens).
    """
    x0, y0 = center_coordinates[:2]
    SWR = 0.14  # bâche sombre

    shading_node = district.find("ShadingSurface")
    if shading_node is None:
        shading_node = SubElement(district, "ShadingSurface")

    def add_one_surface(surface_id, coords):
        surface = SubElement(shading_node, "Surface", {
            "id": str(surface_id),
            "ShortWaveReflectance": str(SWR)
        })
        for i, (x, y, z) in enumerate(coords):
            SubElement(surface, f"V{i}", {
                "x": f"{x - x0}", "y": f"{y - y0}", "z": f"{z}"
            })

    surface_id = 0
    for r in tarps_envelope.index:
        row = tarps_envelope.loc[r]
        geom = row['geometry']

        # Corriger le sens si besoin (ordre antihoraire)
        coords = list(geom.exterior.coords[:-1])
        if not is_ccw(geom):
            coords = coords[::-1]

        # Surface dans le sens normal
        add_one_surface(surface_id, coords)
        surface_id += 1

        # Surface en sens inverse (ombre dans l’autre direction)
        add_one_surface(surface_id, coords[::-1])
        surface_id += 1



def add_surfaces_parapluie(parapluie_node, surfaces_df, center_coordinates):
    """
    Ajoute les disques (parapluies) à un nœud XML donné.

    Parameters:
    - parapluie_node: l’élément XML parent
    - surfaces_df: GeoDataFrame avec 'geometry', 'class_id', 'tid'
    - center_coordinates: tuple (x0, y0) pour recaler les coordonnées
    """
    x0, y0 = center_coordinates[:2]

    SWR = 0.14  # Short Wave Reflectance
    EPS = 0.95  # Long Wave Emissivity

    for r in surfaces_df.index:
        row = surfaces_df.loc[r]
        geom = row['geometry']
        class_id = int(row['class_id'])

        if class_id != 23:
            raise ValueError(f"Unexpected class_id {class_id} in parapluie surface")

        surf = SubElement(
            parapluie_node,
            "Disk",
            {
                "id": str(row['tid']),
                "ShortWaveReflectance": str(SWR),
                "LongWaveEmissivity": str(EPS)
            }
        )

        # Ajout des sommets du disque (ordre ccw)
        for i_pt, (x, y, z) in enumerate(geom.exterior.coords[:-1]):
            SubElement(
                surf,
                f"V{i_pt}",
                {
                    "x": f"{x - x0:.3f}",
                    "y": f"{y - y0:.3f}",
                    "z": f"{z:.3f}"
                }
            )



def add_pedestrian_GPKG_Tmax_2(CM_df, Year_RPB_df, pedestrian_data, max_temp_index, Ta_max_values):
    # Separate different indicators
    MRT_df = CM_df.filter(like='MRT')
    COMFA_df = CM_df.filter(like='COMFA*')
    ITS_df = CM_df.filter(like='ITS')
    UTCI_df = CM_df.filter(like='UTCI')

    # Rename columns with building identificator (bid)
    MRT_df.columns = [extract_first_number(col) for col in MRT_df.columns]
    COMFA_df.columns = [extract_first_number(col) for col in COMFA_df.columns]
    ITS_df.columns = [extract_first_number(col) for col in ITS_df.columns]
    UTCI_df.columns = [extract_first_number(col) for col in UTCI_df.columns]

    # value @ Tair max:
    UTCI_max_index = UTCI_df.iloc[max_temp_index]
    MRT_max_index = MRT_df.iloc[max_temp_index]
    ITS_max_index = ITS_df.iloc[max_temp_index]
    COMFA_max_index = COMFA_df.iloc[max_temp_index]
    
    # UTCI annual thermal comfort
    UTCI_h_PB_df = Year_RPB_df[['buildingId(key)', 'UTCI(h)', 'COMFA(h)', 'ITS(h)', 'MRT(celsius)']]
    UTCI_h_PB_df['buildingId(key)'] = UTCI_h_PB_df['buildingId(key)'].apply(extract_first_number_row)
    UTCI_comfort_percentages = UTCI_h_PB_df[UTCI_h_PB_df['UTCI(h)'] != "-"]
    UTCI_comfort_percentages['buildingId(key)'] = UTCI_comfort_percentages['buildingId(key)'].astype(str)
    
    UTCI_comfort_percentages['MRT(celsius)'] = pd.to_numeric(UTCI_comfort_percentages['MRT(celsius)'], errors='coerce').astype(float)
    UTCI_comfort_percentages['ITS(h)'] = pd.to_numeric(UTCI_comfort_percentages['ITS(h)'], errors='coerce').astype(int)
    UTCI_comfort_percentages['COMFA(h)'] = pd.to_numeric(UTCI_comfort_percentages['COMFA(h)'], errors='coerce').astype(int)
    UTCI_comfort_percentages['UTCI(h)'] = pd.to_numeric(UTCI_comfort_percentages['UTCI(h)'], errors='coerce').astype(int)

    # Match pedestrian_data and indicators
    pedestrian_data['bid'] = pedestrian_data['bid'].astype(str)

    pedestrian_data['UTCI [°C]'] = pedestrian_data['bid'].map(UTCI_max_index)
    pedestrian_data['MRT [°C]'] = pedestrian_data['bid'].map(MRT_max_index)
    pedestrian_data['ITS [°C]'] = pedestrian_data['bid'].map(ITS_max_index)
    pedestrian_data['COMFA [W/m²]'] = pedestrian_data['bid'].map(COMFA_max_index)
    
    # Add UTCI comfort hours to pedestrian_data
    pedestrian_data = pedestrian_data.merge(UTCI_comfort_percentages[['buildingId(key)', 'UTCI(h)', 'COMFA(h)', 'ITS(h)', 'MRT(celsius)']], left_on='bid', right_on='buildingId(key)', how='left')
    pedestrian_data.rename(columns={'UTCI(h)': 'UTCI Annual Thermal Comfort [h]'}, inplace=True)
    pedestrian_data.rename(columns={'COMFA(h)': 'COMFA Annual Thermal Comfort [h]'}, inplace=True)
    pedestrian_data.rename(columns={'ITS(h)': 'ITS Annual Thermal Comfort [h]'}, inplace=True)
    pedestrian_data.rename(columns={'MRT(celsius)': 'MRT Annual Average [Celsius]'}, inplace=True)
    pedestrian_data.drop(columns=['buildingId(key)'], inplace=True)
    
    # Meteorological data
    pedestrian_data['max_temp_index'] = max_temp_index
    pedestrian_data['Ta_max [°C]'] = Ta_max_values['Ta']

    pedestrian_data['Wind speed [m/s]'] = Ta_max_values['FF']
    pedestrian_data['RH [%]'] = Ta_max_values['RH']
    pedestrian_data['Precipitation [mm]'] = Ta_max_values['RR']
    pedestrian_data['Nebulosity [octa]'] = Ta_max_values['N']
    
    month_str = Ta_max_values['m'].astype(int).astype(str)
    day_str = Ta_max_values['dm'].astype(int).astype(str)
    hour_str = Ta_max_values['h'].astype(int).astype(str)
    pedestrian_data['month.day.hour'] = month_str + ':' + day_str + ':' + hour_str
    
    return pedestrian_data

def extract_first_number_row(value):
    match = re.match(r"(\d+)", str(value))
    if match:
        return match.group(1)
    else:
        return value

    
# Function to extract first number (bid)
def extract_first_number(col_name):
    match = re.match(r"(\d+)", col_name)
    if match:
        return match.group(1)
    else:
        return col_name
    
    
def add_trees(district, trees_data, trees_envelope, center_coordinates):
    
    trees = district.find("Trees")
    if trees is None:
        trees = SubElement(district, "Trees")
    
    for i in trees_data.index:
        row = trees_data.loc[i]
        
        tid = row['tid']
        name = row['NOM_COMPL']
        h_tot = row['H_TOTALE'] #[m]
        h_trunc = row['H_TRONC'] #[m]
        
        # dict trees
        leaf_area_index = 3
        leaf_width = 0.01 
        leaf_distance = (h_tot - h_trunc)/(leaf_area_index - 1)
        decidious = True
        class_tree = "C3"

    
        dict_trees = {"id": str(tid), "key": str(name), "leafAreaIndex": str(leaf_area_index),
                      "leafWidth": str(leaf_width), "leafDistance": str(leaf_distance),
                      "decidious": str(decidious).lower(), "class": class_tree}
        
        tree = SubElement(trees, "Tree", dict_trees)
        
        te = trees_envelope[trees_envelope['tid'] == tid]
        add_surfaces_trees(tree, te, center_coordinates)
    
    return



def add_surfaces_trees(tree, trees_envelope, center_coordinates):
    x_center = center_coordinates[0]
    y_center = center_coordinates[1]

    for r in trees_envelope.index:
        row = trees_envelope.loc[r]
        geometry = row['geometry']
        class_id = row['class_id']
        
        shortwave_reflectance = 0.3
        long_wave_emissivity = 0.95

        if class_id == 1:
            dict_surface = {"id": str(r), "key": "trunc", "ShortWaveReflectance": str(shortwave_reflectance),
                            "LongWaveEmissivity": str(long_wave_emissivity)}
            surface = SubElement(tree, "Trunc", dict_surface)


        elif class_id == 2:
            dict_surface = {"id": str(r), "key": "leaf", "ShortWaveReflectance": str(shortwave_reflectance),
                            "LongWaveEmissivity": str(long_wave_emissivity)}
            surface = SubElement(tree, "Leaf", dict_surface)
    

        # Add points translated to center of the scene
        for p in range(len(geometry.exterior.coords)-1):
            point_name = "V{}".format(p)
            coordinates = geometry.exterior.coords[p]
            x = str(coordinates[0]-x_center)
            y = str(coordinates[1]-y_center)
            z = str(coordinates[2])

            dict_point = {"x": x, "y": y, "z": z}
            point = SubElement(surface, point_name, dict_point)
    
    
    return

    


def normal_comp(triangle):
    P1=triangle.exterior.coords[0]
    P2=triangle.exterior.coords[1]
    P3=triangle.exterior.coords[2] 
    P1P2 = [P2[0]-P1[0],P2[1]-P1[1],P2[2]-P1[2]]
    P1P3 = [P3[0]-P1[0],P3[1]-P1[1],P3[2]-P1[2]]
    cross_product_x = P1P2[1] * P1P3[2] - P1P2[2] * P1P3[1]
    cross_product_y = P1P2[2] * P1P3[0] - P1P2[0] * P1P3[2]
    cross_product_z = P1P2[0] * P1P3[1] - P1P2[1] * P1P3[0]
    normal = (cross_product_x, cross_product_y, cross_product_z)
    
    return normal
    
        


# DHN ########################################################################

def add_district_heating_center(district, cp_water=4180, rho=990,
                                mu=0.0004):

    d = {"id": str(0), "Cp": str(cp_water), "rho": str(rho), "mu": str(mu)}
    district_heating_center = SubElement(district, 'DistrictEnergyCenter', d)
    return district_heating_center

def add_thermal_station(district_heating_center, node_id,
                        rho=999, n0=5000, a0_n0=1247180, a1_n0=-1640.236,
                        a2_n0=-0.00016031, e_pump=0.6,  begin_day=1, end_day=365,
                        low_supply_temp=90, high_supply_temp=75,
                        low_ext_temp_limit=-10, high_ext_temp_limit=15,
                        high_temperature_drop=20, start_summer_threshold=16,
                        end_summer_threshold=5, p_max=900000, delta_p=200000,
                        dp_type='constant', storage_type='seasonalStorageHeating',
                        kvMax=1000, temp_storage=75, c_storage=1e7,
                        efficiencies=None, stages=None):
    
    print("Preparing Thermal Station...")
    
    # Set thermal station parameters
    d_thermal_station = {"linkedNodeId": str(node_id), "beginDay": str(begin_day), 
                         "endDay": str(end_day), "type": str(storage_type), "kvMax": str(kvMax)}
    thermal_station = SubElement(district_heating_center, "ThermalStation",
                                 d_thermal_station)
    
    # Set temperature parameters
    d_temp_setpoint = {"type": "affineWinterConstantSummer",    
                        "lowExtTemp": str(low_ext_temp_limit),
                        "highExtTemp": str(high_ext_temp_limit),
                        "lowExtTempSupplyTemp": str(low_supply_temp),
                        "highExtTempSupplyTemp": str(high_supply_temp),
                        "startSummerTempThreshold": str(start_summer_threshold),
                        "endSummerTempThreshold": str(end_summer_threshold)}
    temp_setpoint = SubElement(thermal_station, "TemperatureSetpoint",
                                d_temp_setpoint)
    
    # Set pressure parameters
    if dp_type == 'affine': 
        mass_flows = np.array([5, 10, 64, 118])*rho/3600
        pressure_diffs = [180000, 250000, 390000, 480000]
        d_pressure_setpoint = {"type": "affine", "massFlow1": str(mass_flows[0]),
                                "pressureDiff1": str(pressure_diffs[0]),
                                "massFlow2": str(mass_flows[1]),
                                "pressureDiff2": str(pressure_diffs[1]),
                                "massFlow3": str(mass_flows[2]),
                                "pressureDiff3": str(pressure_diffs[2]),
                                "massFlow4": str(mass_flows[3]),
                                "pressureDiff4": str(pressure_diffs[3])}
        pressure_setpoint = SubElement(thermal_station, "PressureSetpoint",
                                        d_pressure_setpoint)
    elif dp_type == 'constant':
        d_pressure_setpoint = {"type": "constant",
                                "targetPressureDiff": str(delta_p)}
        pressure_setpoint = SubElement(thermal_station, "PressureSetpoint",
                                        d_pressure_setpoint)
    else:
        raise ValueError('dp_type must be either "constant" or "affine"')

    # Set pump parameters #TODO
    d_pump = {"n0": str(n0), "a0": str(a0_n0),
              "a1": str(a1_n0), "a2": str(a2_n0)}
    ts_pump = SubElement(thermal_station, "Pump", d_pump)

    d_epump = {"type": "constant", "efficiencyPump": str(e_pump)}
    epump = SubElement(ts_pump, "EfficiencyPump", d_epump)

    # Set storage parameters #TODO
    d_storage = {"type": "simple", "initialTemperature": str(high_supply_temp),
                  "heatCapacity": str(c_storage)}
    ts_storage = SubElement(thermal_station, "Storage", d_storage)

    # d_boiler = {"Pmax": str(p_max/2), "eta_th": "0.95"}
    # ts_boiler = SubElement(thermal_station, "Boiler", d_boiler)
    
    # d_chp = {"Pmax": str(p_max/2), "eta_th": "0.35", "eta_el": "0.6", "minPartLoadCoeff": "0.2"}
    # ts_chp = SubElement(thermal_station, "CHP", d_chp)

    # Add heat production             
    if stages == None:
        d_boiler = {"Pmax": str(p_max), "eta_th": "0.95"}
        ts_boiler = SubElement(thermal_station, "Boiler", d_boiler)
    
    else:
        # Set production units parameters
        for i in range(len(stages[0])):
            unit_power = stages[0][i]
            unit_type = stages[1][i]
            
            if unit_type == "CHP":
                eta_th = efficiencies[efficiencies['T_eff']=='CHP_th']['efficiency'].iloc[0]
                eta_el = efficiencies[efficiencies['T_eff']=='CHP_el']['efficiency'].iloc[0]
                d_CHPHP = {"Pmax": str(unit_power), "eta_th": str(eta_th), "eta_el": str(eta_el), "minPartLoadCoeff":"0.0"} #TODO
                ts_CHPHP = SubElement(thermal_station, "CHP", d_CHPHP)
    
            elif unit_type == "Heat_Pump_Water":
                eta_tech = efficiencies[efficiencies['T_eff']=='Heat_Pump_Water_eta_tech']['efficiency'].iloc[0]
                COP = efficiencies[efficiencies['T_eff']=='Heat_Pump_Water_COP']['efficiency'].iloc[0]
                P_el = unit_power/COP
                d_HeatPump = {"Pmax": str(P_el), "eta_tech": str(eta_tech), "Ttarget":"80", "Tsource":"20"} 
                ts_HeatPump = SubElement(thermal_station, "HeatPump", d_HeatPump)
        
            elif unit_type == "Wood_boiler":
                eta_th = efficiencies[efficiencies['T_eff']=='Wood_boiler']['efficiency'].iloc[0]
                d_WoodBoiler = {"Pmax": str(unit_power), "eta_th": str(eta_th), "name":"Wood Boiler"} #TODO
                ts_WoodBoiler = SubElement(thermal_station, "Boiler", d_WoodBoiler)   
    
            elif unit_type == "Gas_boiler":
                eta_th = efficiencies[efficiencies['T_eff']=='Gas_boiler']['efficiency'].iloc[0]
                d_GasBoiler = {"Pmax": str(unit_power), "eta_th": str(eta_th),"name":"Gas Boiler"} #TODO
                ts_GasBoiler = SubElement(thermal_station, "Boiler", d_GasBoiler)  


def add_network(district_heating_center, points, pipes):
    '''

    '''
    print("Preparing Network...")

    network = SubElement(district_heating_center,
                         "Network", {"soilkValue": "0.5"})

    # Set nodes parameters
    points = points.sort_values(['npid']).reset_index(drop=True)
    for index_node in points.index:
        current_row = points.loc[index_node]
        node_id = current_row['npid']
        node_coordinates_x = current_row['coordinates_x']
        node_coordinates_y = current_row['coordinates_y']
        node_EGID = current_row['EGID']
        node_type = current_row['Type']
        # Node connected to Thermal station
        if node_type == 'start heating station':
            pair_type = "ThermalStationNodePair"
        # Node connected to Substation
        elif node_type == 'HX':
            pair_type = "SubstationNodePair"
        # Node connecting pipes without heat exchange
        else:
            pair_type = "NodePair"
        d_point = {"id": str(node_id), "key": str(node_EGID), 
                   "x": str(node_coordinates_x), "y": str(node_coordinates_y), "z": str(0)}
        point_pair = SubElement(network, pair_type, d_point)

    # Set pipes parameters
    for index_pipe in pipes.index:
        current_row = pipes.loc[index_pipe]
        pipe_start_point = current_row['startpoint']
        pipe_end_point = current_row['endpoint']
        pipe_id = current_row['pid']
        pipe_length = current_row['length[m]']
        pipe_inner_radius = current_row['DN']/1000/2
        pipe_insulation_thick = current_row['insulation_thickness']
        pipe_insulation_k_value = current_row['insulation_k_value']
        
        d_pipe = {"id": str(pipe_id),
                  "node1": str(pipe_start_point),
                  "node2": str(pipe_end_point),
                  "length": str(pipe_length),
                  "innerRadius": str(pipe_inner_radius),
                  "interPipeDistance": "0.5"}
        pipe_pair = SubElement(network, "PipePair", d_pipe)

        d_pipe_properties = {
            "insulationThick": str(pipe_insulation_thick),
            "insulationkValue": str(pipe_insulation_k_value),
            "buriedDepth": "1"}
        supply_pipe = SubElement(pipe_pair, "SupplyPipe", d_pipe_properties)
        return_pipe = SubElement(pipe_pair, "ReturnPipe", d_pipe_properties)


def change_boiler_to_substation(district, substations, points, design_epsilon=0.6, design_temp_difference=20):
    print("Modifying Boilers to Substations...")

    for building in district.iter("Building"):
        # Remove boiler
        heat_source = building.find("./HeatSource")
        boiler = heat_source.find("./Boiler")
        heat_source.remove(boiler)
            
        if building.attrib["Simulate"] == "true":
            # Set Substation parameters
            building_EGID = int(building.attrib["key"])
    
            substation_row = substations.loc[substations["EGID"] == building_EGID]
            node_row = points.loc[points["EGID"] == building_EGID]

            design_factor = 115/100 # Heat exchanger sizing
            design_thermal_power = substation_row["Power[W]"].iloc[0]*design_factor   
            linked_node_id = node_row["npid"].iloc[0]

            type_substation = "simple"
            
            d_substation = {"linkedNodeId": str(linked_node_id),
                            "designThermalPower": str(design_thermal_power),
                            "designTempDifference": str(design_temp_difference),
                            "designEpsilon": str(design_epsilon),
                            "type": type_substation}
            substation = SubElement(heat_source, "Substation", d_substation)
        
        else:
            # Remove heat source
            heat_source = building.find("./HeatSource")
            building.remove(heat_source)


def add_all_dhn(district, points, pipes, substations, hs_delta_p=200000, hs_p_max = 10000000, dp_type='affine', stages = None):
    # Add District heating center
    district_heating_center = add_district_heating_center(district)
    
    # Add Network (nodes and pipes)
    add_network(district_heating_center, points=points.copy(), pipes=pipes.copy())
    
    # Add Thermal station
    ts_node_id = points.loc[points['Type']=='start heating station']['npid'].iloc[0]
    
    add_thermal_station(district_heating_center, ts_node_id, delta_p=hs_delta_p, p_max = hs_p_max, 
                        dp_type=dp_type, stages=stages, temp_storage=50, c_storage=1e7)
    
    
    change_boiler_to_substation(district, substations, points)
