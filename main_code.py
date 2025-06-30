# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:16:28 2023

@author: Olivier Chavanne
"""

import geopandas as gpd
import pandas as pd
from shapely import box
import os
import matplotlib.pyplot as plt

# Local libraries
from xml.etree.ElementTree import Element, SubElement
from enerCAD.building import generate_envelope
from enerCAD.building import generate_buildings
from enerCAD.building import generate_pedestrian
from enerCAD.building import get_scene_center
from enerCAD.building import generate_tree
from enerCAD.building import update_pedestrian_z
from enerCAD.building import update_tree_z
from enerCAD.building import generate_tarps_from_polygons
from enerCAD.building import generate_parapluies_from_polygons
import enerCAD.xml as xml
import enerCAD.result as result
import enerCAD.network as network
import enerCAD.production as prod
import enerCAD.KPI as KPI

# URL for RegBL API request
GEOADMIN_BASE_URL = "https://api.geo.admin.ch/rest/services/ech/MapServer/ch.bfs.gebaeude_wohnungs_register/"
    
##################################################
# 
#                  Functions
#
##################################################

def create_xml_root(xml_file_to_copy, climate_file, horizon_file):
    '''
    Parameters                                                          
    ----------
    xml_file_to_copy : TYPE
        DESCRIPTION.
    climate_file : TYPE
        DESCRIPTION.
    horizon_file : TYPE
        DESCRIPTION.

    Returns
    -------
    root : TYPE
        DESCRIPTION.
    district : TYPE
        DESCRIPTION.
    '''
    
    # Write XML file for CitySim :
    print("Writing XML file...")    
    # Add Root 
    root = xml.add_root()
    # Add Simulation days
    xml.add_simulation_days(root)
    # Add Climate
    xml.add_climate(root, climate_file)
    # Add District
    district = xml.add_district(root)
    
    # Horizon
    # read in the tab-separated file as a dataframe
    horizon_df = pd.read_csv(horizon_file, sep='\t', header=None)
    # assign column names to the dataframe
    horizon_df.columns = ['phi', 'theta']
    # Add Far field obstructions
    xml.add_far_field_obstructions(district, horizon_df)
    
    # Add all the composites and profiles, taken from a source XML
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'Composite')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'OccupancyDayProfile')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'OccupancyYearProfile')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'DeviceType')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'ActivityType')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'DHWDayProfile')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'DHWYearProfile')
    
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'Building')
    xml.add_child_from_xml_to_district(district, xml_file_to_copy, 'DistrictEnergyCenter')
    
    print("Xml source copied")
    
    return root, district 


def Module_1(gpkg_filepath, XYZfile, GEOADMIN_BASE_URL,
             directory_path, xml_name,
             xml_base_file, climate_file, horizon_file,
             create_geometry_3D=False, calculate_volume_3D=False,
             EGID_column='RegBL_EGID'):
    '''
    Parameters
    ----------
    gpkg_filepath : TYPE
        DESCRIPTION.
    GEOADMIN_BASE_URL : TYPE
        DESCRIPTION.
    directory_path : TYPE
        DESCRIPTION.
    xml_file_to_create : TYPE
        DESCRIPTION.
    xml_base_file : TYPE
        DESCRIPTION.
    climate_file : TYPE
        DESCRIPTION.
    horizon_file : TYPE
        DESCRIPTION.
    create_geometry_3D : TYPE, optional
        DESCRIPTION. The default is False.
    calculate_volume_3D : TYPE, optional
        DESCRIPTION. The default is False.
    EGID_column : TYPE, optional
        DESCRIPTION. The default is 'RegBL_EGID'.

    Returns
    -------
    None.
    '''
    
    ### Exctract geopackage ###
    
    print("Exctracting geopackage layers...")
    
    # MO Cadaster
    MO_all = gpd.read_file(gpkg_filepath, layer = "zone_tout")
    MO_dhn = gpd.read_file(gpkg_filepath, layer = "zone_cad")
    centrale = gpd.read_file(gpkg_filepath, layer = "centrale")
    pedestrian = gpd.read_file(gpkg_filepath, layer = "pedestrian") #ajout des pedestrians
    trees = gpd.read_file(gpkg_filepath, layer = "trees")
    new_trees = gpd.read_file(gpkg_filepath, layer = "new_trees")
    EGID_column = 'RegBL_EGID'
    
    # Split Multipolygons into Polygons
    zone_all = MO_all.explode(index_parts=False)
    zone_dhn = MO_dhn.explode(index_parts=False)
    
    # List containing EGID of buildings to simulate
    EGID_list = MO_dhn[EGID_column].tolist()
    
    # Save EGID list of buildings connected to CAD
    df_EGID = pd.DataFrame(EGID_list)
    df_EGID.columns = ['EGID']
    EGID_path = os.path.join(directory_path, 'EGID.csv')     
    df_EGID.to_csv(EGID_path, index=False)
    print("EGID.csv created")
    
    # Swissbuildings3D
    print("Swissbuildings3D processing...")
    try:
        floor_data = gpd.read_file(gpkg_filepath, layer = "floor")
        roof_data = gpd.read_file(gpkg_filepath, layer = "roof")
        wall_data = gpd.read_file(gpkg_filepath, layer = "wall")
        green_data = gpd.read_file(gpkg_filepath, layer = 'green')
        ori_street_data = gpd.read_file(gpkg_filepath, layer = 'streets')
        street_data = ori_street_data[~ori_street_data['objektart'].isin( ['Verbindung', 'Platz'])]
        
        
        # Filter on the zone with 10m buffer around surrounding square box 
        zone_bounds = MO_all.geometry.buffer(10).values.total_bounds
        zone_box = box(zone_bounds[0], zone_bounds[1], zone_bounds[2], zone_bounds[3])
        
        # Cut swissbuildings3D to zone of concern
        floor_data_intersection = floor_data[floor_data.geometry.intersects(zone_box)]
        roof_data_intersection = roof_data[roof_data.geometry.intersects(zone_box)]
        wall_data_intersection = wall_data[wall_data.geometry.intersects(zone_box)]
    
        # Split Multipolygons into Polygons
        zone_floor = floor_data_intersection.explode(index_parts=True).reset_index()
        zone_roof = roof_data_intersection.explode(index_parts=True).reset_index()
        zone_wall = wall_data_intersection.explode(index_parts=True).reset_index()
        print('Swissbuildings3D cut to zone of interest \n')
    
    except: print('Error : Swissbuildings3D not provided')

    ### Envelope processing ###
    
    try:
        # Get z coordinates of 1st vertex from 1st surface of 1st building's floor polygon as altitude by default for MO footprints
        altitude_default = zone_floor.loc[0].geometry.exterior.coords[0][2]
    except:
        altitude_default = 0
    
    # Create DataFrames containing all necessary information for each building
    print("Creating Buildings GeoDataFrame...")
    footprints, buildings = generate_buildings(zone_all, EGID_list, GEOADMIN_BASE_URL, altitude_default,
                                               create_geometry_3D, calculate_volume_3D, zone_floor, zone_roof, zone_wall)
    print("Buildings GeoDataFrame created \n") 
    center_coordinates = get_scene_center(footprints)

  
    root, district = create_xml_root(xml_base_file, climate_file, horizon_file)
    
    print("Generating ground...")
    terrain_df = pd.read_table(XYZfile, skiprows=1,  sep='\s+', names=['X', 'Y', 'Z'])
    ground_data = xml.add_ground_from_XYZ(MO_dhn, district, terrain_df, zone_box, center_coordinates)


    print("Adding actual scene ground in xml file...")
    # id, k-factor, SWR, distric, ground_data,..
    xml.add_ground_in_xml(ground_data, district, center_coordinates, groundtype=31, kFactor=0.1, ShortWaveReflectance=0.4)

    
    road_index_list, _ = xml.modify_type_road(district, ground_data, road_groundtype=2, road_kfactor=0.1, road_SWR=0.14, modify_data=street_data)
    green_index_list, green_itsctd = xml.modify_type(district, ground_data, groundtype=3, kfactor=0.7, SWR=0.22, modif_data=green_data)
    xml.cut(district, ground_data, MO_dhn, footprints)

    # Generate the envelope surfaces
    print("Generating Buildings envelope...")
    
    envelope, buildings_volume_3D = generate_envelope(footprints, buildings, calculate_volume_3D)

    print("Envelope created \n")
    
    # Merge "volume_3D" and "n_occupants" to main buildings geodataframe according to 'bid'
    merged_buildings = buildings.merge(buildings_volume_3D, left_on='bid', right_on='bid', how='left')    
    if not merged_buildings.empty:
        columns_to_add = ['volume_3D', 'n_occupants']
        for column in columns_to_add:
            buildings[column] = merged_buildings[column]
        print("Buildings 3D volume calculated and merged \n")
    
    ### Buildings XML processing ###
    
    print("Adding buildings...")
    # Add the buildings
    xml.add_all_buildings(district, buildings, envelope, center_coordinates)

   
    print(ground_data.sample(n=10, random_state=1))
    print("Creating pedestrian...")
    pedestrian = pedestrian.to_crs(2056) 
    
    pedestrian_data, pedestrian_envelope = generate_pedestrian(pedestrian, buildings, ground_data)
    print("Adding pedestrians in xml file...")
    xml.add_pedestrians(district, pedestrian_data, pedestrian_envelope, center_coordinates)
    
    
    print("Creating trees...")
    next_tid = 0
    trees = trees.to_crs(2056) 

    trees_data, trees_envelope, next_tid = generate_tree(trees, ground_data, next_tid)
    trees = update_tree_z(trees, ground_data)
    xml.add_trees(district, trees_data, trees_envelope, center_coordinates)

    
    # Write XML file of Scenario 1
    sc_id=1
    
    
    sidewalk = gpd.read_file(gpkg_filepath, layer = 'sidewalk')
    ground_green = gpd.read_file(gpkg_filepath, layer = 'soil_green')

    ground_green = ground_green.to_crs(2056)


    sidewalk_index_list, _ = xml.modify_type(district, ground_data, 
                                                    groundtype=31, kfactor=0.1, SWR=0.4, 
                                                    modif_data=sidewalk)
    ground_green_index_list, _ = xml.modify_type(district, ground_data, 
                                                 groundtype=3, kfactor=0.7, SWR=0.22, 
                                                 modif_data=ground_green)

    
  
    print("Adding extra trees..")
    new_trees = new_trees.to_crs(2056) 
    new_trees_data, new_trees_envelope, next_tid = generate_tree(new_trees, ground_data, next_tid)
    trees = update_tree_z(trees, ground_data)
    xml.add_trees(district, new_trees_data, new_trees_envelope, center_coordinates)
    
    

    #choisir entre tarps et parapluies 
    #Les 2 ont le meme objectifs mais une esthetique differente
    print("Creating tarps...")


    tarps_2d = gpd.read_file(gpkg_filepath, layer='tarps') 
    tarps_2d  = tarps_2d.to_crs(2056)
    

    tarps_df, tarps_surfaces = generate_tarps_from_polygons(tarps_2d,ground_data,
        start_tid=next_tid, height=6.0, thickness=0.8)
    

    print("Tarps surface count:", len(tarps_surfaces))
    print(tarps_surfaces.head())

    next_tid = tarps_df['tid'].max() + 1  



    print("Adding tarps in XML…")
    xml.add_shading_surfaces(district, tarps_df, tarps_surfaces, center_coordinates)

    print("Tarps added to XML\n")
    
    
    print("Generating parapluies...")

    parapluie_zones = gpd.read_file(gpkg_filepath, layer='tarps')
    parapluie_zones = parapluie_zones.to_crs(2056)

    parapluies_df, parapluies_surfaces = generate_parapluies_from_polygons(
        parapluie_zones,
        ground_data,
        start_tid=next_tid, 
        height=5.0,          
        radius=2.0,           
        spacing=4.0          
    )
    
    next_tid = parapluies_df['tid'].max() + 1

    surfaces_parapluies = parapluies_surfaces[parapluies_surfaces['class_id'] == 23]
    xml.add_shading_surfaces(district, parapluies_df, surfaces_parapluies, center_coordinates)
    
    
    #approximation comparison
    print('approximation comparison \n') 
    
    selected_grounds1=ground_data[ground_data['gid'].isin(road_index_list)]
    selected_grounds2=ground_data[ground_data['gid'].isin(green_index_list)]
    selected_grounds3=ground_data[ground_data['gid'].isin(sidewalk_index_list)]
    selected_grounds4=ground_data[ground_data['gid'].isin(ground_green_index_list)]
    
 

    # add layers of simulated green areas and streets
    selected_grounds1.to_file(gpkg_filepath, layer='street_grounds_2m', driver='GPKG')
    selected_grounds2.to_file(gpkg_filepath, layer='green_grounds_2m', driver='GPKG')
    selected_grounds3.to_file(gpkg_filepath, layer='sidewalk_2m', driver='GPKG')
    selected_grounds4.to_file(gpkg_filepath, layer='ground_green_2m', driver='GPKG')
    
    

    
    #write xml file of default case
    print('creating xml file \n')

    # Write XML file
    xml.cut(district, ground_data, MO_dhn, footprints)
    #xml.cut(district, ground_data, MO_dhn, footprints, buildings)
    xml_path = os.path.join(directory_path, xml_name+".xml")     
    xml.write_xml_file(root, xml_path)
    print(f"{xml_name}.xml file created \n")
   
    return envelope, ground_data, buildings, zone_dhn, centrale
   



def simulate_citysim(directory_path, xml_file, citysim_filepath):
    '''
    Parameters
    ----------
    xml_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    
    import subprocess
    import time
    start = time.time()
    print('Process started')
    print(f'Simulation of {xml_file}.xml...')

    #run CitySim.exe with xml file
    xml_path = os.path.join(directory_path, xml_file+".xml")
    result = subprocess.run([citysim_filepath, '-q', f"{xml_path}"])
    
    end = time.time()
    duration = end - start
    m, s = divmod(duration, 60)
    print('Simulation ended. Time :', "%.0f" %m,'min', "%.0f" %s,'s \n')
    

#------------------Part 2 iterating----------------------------------------------------------

def Module_2(directory_path, xml_name, zone_dhn, centrale,
             xml_DHN, climate_file, horizon_file,
             scenarios_list):   

    # Open the 1st iteration results file (without DHN)
    results_filepath = os.path.join(directory_path, xml_name+"_TH.out")
    results = pd.read_csv(results_filepath, delimiter="\t")
    results = results.set_index("#timeStep")
    print(f'{xml_name}_TH.out opened')
    
    # Get P_max for every building in the network
    power_EGID = result.get_Pmax_per_EGID(results)
    power_EGID_path = os.path.join(directory_path, 'Power_EGID.csv')
    power_EGID.to_csv(power_EGID_path, index=False)
        
    # Create and size network
    graph, lines_gdf, nodes_gdf, points, pipes, substations = network.get_trace(zone_dhn, centrale, power_EGID)
    
    # Save to csv file
    points_path = os.path.join(directory_path, 'Points.csv')
    pipes_path = os.path.join(directory_path, 'Pipes.csv')
    substations_path = os.path.join(directory_path, 'Substations.csv')
    points.to_csv(points_path, index=False)
    pipes.to_csv(pipes_path, index=False)
    substations.to_csv(substations_path, index=False)
    print("csv files saved")
    
    # Get Load duration curve of the network
    load_curve = result.get_thermal_load_curve(results)
    load_curve_path = os.path.join(directory_path, 'Load_curve.csv')
    load_curve.to_csv(load_curve_path, index=False)
     
    # Compute production scenarios
    scenarios = prod.get_scenarios(load_curve)
    
    # Calculate production water storage
    volume_storage, capacity_storage = prod.get_storage(load_curve)
    
    for i in range(len(scenarios)):
        scenario = scenarios[i]
        sc_id = scenario[0]
        if sc_id in scenarios_list:

            ### Scenarios XML processing ###
            
            xml_to_copy_path = os.path.join(directory_path, xml_name+'.xml' )
            root, district = create_xml_root(xml_to_copy_path, climate_file, horizon_file)
               
            # Add District heating center
            district_heating_center = xml.add_district_heating_center(district)
            
            # Add Network (nodes and pipes)
            xml.add_network(district_heating_center, points=points.copy(), pipes=pipes.copy())

            # Change Boilers into Substations
            xml.change_boiler_to_substation(district, substations=substations.copy(), points=points.copy())
            
            # Add Thermal station
            ts_node_id = points.loc[points['Type']=='start heating station']['npid'].iloc[0]
            network_p_max = pipes['power_line[W]'].max()
            production_stages = [scenario[1],scenario[2]]
            
            # Efficiency data
            technology_parameters = pd.read_csv("KPI.csv", delimiter=",")
            eff_columns = ['T_eff','efficiency']
            efficiency_parameters = technology_parameters[eff_columns]
            
            xml.add_thermal_station(district_heating_center, ts_node_id, p_max=network_p_max, 
                                    c_storage=capacity_storage, efficiencies=efficiency_parameters, stages=production_stages)

            # Write XML file
            scenario_path = os.path.join(directory_path,f"Scenario_{sc_id}")
            os.makedirs(scenario_path, exist_ok=True)
            xml_to_create_path = os.path.join(scenario_path, xml_DHN+f"_sc_{sc_id}"+".xml")
            xml.write_xml_file(root, xml_to_create_path)
            print(f'{xml_DHN}_sc_{sc_id}.xml file created \n')

    return graph, lines_gdf, nodes_gdf, results, scenarios, volume_storage

#--------------------- Part 3

def Module_results_network(scenario_path, sc_id, xml_DHN, zone_dhn, centrale, graph, lines_gdf, nodes_gdf):
    
    # Open the 2nd iteration results file (with DHN)
    results_filepath = os.path.join(scenario_path, xml_DHN+f"_sc_{sc_id}"+"_TH.out")
    results = pd.read_csv(results_filepath, delimiter="\t")
    results_final = results.set_index("#timeStep")
        
    # Get Data for every node and pipe in the network
    Pipes_mass_flow, Nodes_supply_temp, index_max = result.get_network_data_max(results_final)
    
    # Save to csv file
    Pipes_mass_flow_filepath = os.path.join(scenario_path, f'Pipes_mass_flow_sc_{sc_id}.csv')
    Nodes_supply_temp_filepath = os.path.join(scenario_path, f'Nodes_supply_temp_sc_{sc_id}.csv')
    Pipes_mass_flow.to_csv(Pipes_mass_flow_filepath, index=False) 
    Nodes_supply_temp.to_csv(Nodes_supply_temp_filepath, index=False)

    return results_final, Pipes_mass_flow, Nodes_supply_temp, index_max

#--------------------- KPI calculation

def Module_KPI(results_production, volume_storage, 
               scenarios, sc_id, scenario_path, do_plot):

    # Get production consumption in kWh
    pump_cons, fuel_cons, elec_cons, th_prod, df_th_prod, df_elec = result.get_energy_data(results_production, sc_id, scenarios)

    # Save thermal results to csv file
    th_production_results_filepath = os.path.join(scenario_path, f'Thermal_sc_{sc_id}.csv')
    df_th_prod.to_csv(th_production_results_filepath, index=True) 

    if do_plot == True:
    # Plot energy production data
        print('Energy production plot...')
        result.plot_energy_data(results_production, sc_id, scenarios, scenario_path)

    # Calculate KPI (key performance indicators)
    technology_parameters = pd.read_csv("KPI.csv", delimiter=",")

    df_KPI = KPI.calculate_KPI(sc_id, scenarios, volume_storage, technology_parameters,
                                                 pump_cons, fuel_cons, elec_cons, th_prod)
    print('KPI calculated')
    
    # Save electrical results to csv file
    electricity_results_filepath = os.path.join(scenario_path, f'Electricity_sc_{sc_id}.csv')
    df_elec.to_csv(electricity_results_filepath, index=True) 

    # Save KPI results to csv file
    KPI_results_filepath = os.path.join(scenario_path, f'KPI_results_sc_{sc_id}.csv')
    df_KPI.to_csv(KPI_results_filepath, index=False) 
    
    return df_KPI


##################################################
# 
#         Information to provide
#
##################################################

# Geopackage filepath
gpkg_filepath = r"input/Conthey_3.gpkg"                                   #TODO

# Create geometry with swissbuildings3D
create_geometry_3D = True                                    #TODO

# Calculate volume from swissbuildings3D
calculate_volume_3D = True                                   #TODO

# CitySim.exe filepath
citysim_filepath = r"input/CitySim.exe" #TODO

# XML name to export
directory_path = r"output"                                   #TODO

os.makedirs(directory_path, exist_ok=True)
                                      
xml_name = directory_path                                       
xml_DHN = "DHN_"+xml_name

# XML source files
xml_base_file = r"xml_base.xml"                                
climate_file = r"input/cli/Conthey_Contemporary_2025.cli"                                   #TODO
horizon_file = r"input/cli/Conthey.hor"                                        #TODO
XYZfile = r"input/Conthey.xyz"  #TODO    

# Scenarios to simulate
scenarios_list = [1,2,3,4,5,6]                                  #TODO

do_plot = True

def main(): 
    
    # Generate individual buildings XML
    print('***Module 1*** \n')
    envelope, ground_data, buildings, zone_dhn, centrale = Module_1(gpkg_filepath,XYZfile, GEOADMIN_BASE_URL, 
                                             directory_path, xml_name,
                                             xml_base_file, climate_file, horizon_file,
                                             create_geometry_3D, calculate_volume_3D,
                                             EGID_column='RegBL_EGID')
 
    # # 1st CitySim simulation
    # simulate_citysim(directory_path, xml_name, citysim_filepath)
    
    # # Generate DHN XML for each scenario
    # print('***Module 2*** \n')
    # graph, lines_gdf, nodes_gdf, results, scenarios, volume_storage = Module_2(directory_path, xml_name, 
                                                                        # zone_dhn, centrale, 
                                                                        # xml_DHN, climate_file, horizon_file,
                                                                        # scenarios_list)
    # # Intermediate results
    # network.plot_network_sizing(directory_path, graph, zone_dhn, lines_gdf, centrale)     
    # result.plot_load_curve(results, directory_path)
    
    # KPI_result_list = []
        
    # # DHN simulation for each scenario
    # for i in range(len(scenarios_list)):
        # sc_id = scenarios_list[i]
        # print(f'***Scenario {sc_id}*** \n')
        
        # # CitySim simulation        
        # scenario_path = os.path.join(directory_path,f"Scenario_{sc_id}")
        # simulate_citysim(scenario_path, f'{xml_DHN}_sc_{sc_id}', citysim_filepath)
                
        # # Analysing final results  
        # print('Results processing...')
        # results_prod_path = os.path.join(scenario_path,  f'{xml_DHN}_sc_{sc_id}_TH.out')
        # results_production = pd.read_csv(results_prod_path, delimiter="\t")
            
        # # KPI calculation
        # df_KPI = Module_KPI(results_production, volume_storage, 
                            # scenarios, sc_id, scenario_path, do_plot)
        
        # KPI_result_list.append(df_KPI)

        # # Network final plots
        # results_final, Pipes_mass_flow, Nodes_supply_temp, index_max = Module_results_network(scenario_path, sc_id, xml_DHN, 
                                                                                # zone_dhn, centrale, 
                                                                                # graph, lines_gdf, nodes_gdf)   
        # if do_plot == True:
            # network.plot_network_data(scenario_path, sc_id, graph, zone_dhn, 
                                      # lines_gdf, nodes_gdf, centrale, index_max)
        
        # plt.close("all")
        # print(f"Scenario {sc_id} processed \n")
    
    # if len(KPI_result_list)>1:
        # KPI.plot_KPI(KPI_result_list, scenarios_list, directory_path)
    
    # print("***Overall processing finished***")
    # print(f"Find all results and graphs in directory : {directory_path}")

if __name__ == "__main__":
    plt.close("all")
    
    import subprocess
    import time
    start_overall = time.time()
    print('Main code started')
    print('-----------------')
    
    main()    

    print('-----------------')
    print('Main code ended')
    print('-----------------')
    end_overall = time.time()
    duration_overall = end_overall - start_overall
    m, s = divmod(duration_overall, 60)
    print('Overall run time :', "%.0f" %m,'min', "%.0f" %s,'s \n')














