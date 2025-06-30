# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:14:48 2021
@author: Roberto Boghetti

Modified on August 2023
@author: Olivier Chavanne
"""
import numpy as np
import geopandas as gpd
import shapely.geometry as shp
import requests
import math
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.geometry import Point
import pandas as pd
from shapely.affinity import translate
from shapely.ops import nearest_points

import enerCAD.dictionaries as dicts

# Silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# def fix_ccw_polygons(df, geometry_column = 'geometry', inplace = False):
#     '''   
#     Parameters
#     ----------
#     df : geoDataFrame with the geometries to orient
#     inplace : bool indicating if orient as convention or contrary

#     Returns
#     -------
#     df or oriented_df : geoDataFrame with the geometries oriented as indicated
#     '''
    
#     # Orient as convention (exterior ccw, interior cw)
#     if inplace:
#         df[geometry_column] = df[geometry_column].apply(orient, sign=1.0)
#         return df
#     # Orient contrary as convention
#     else:
#         oriented_df = df.copy()
#         oriented_df[geometry_column] = oriented_df[geometry_column].apply(orient, sign=-1.0)
#         return oriented_df


def fix_ccw_polygons(df, geometry_column = 'geometry', inplace = False):
    if inplace:
        df[geometry_column] = df[geometry_column].apply(orient, sign = 1.0)
        return df
    oriented_df = df.copy()
    oriented_df[geometry_column] = df[geometry_column].apply(orient, sign = -1.0)
    return oriented_df


def get_scene_center(data):
    '''   
    Parameters
    ----------
    data : geoDataFrame with the geometries.

    Returns
    -------
    center_coordinates : tuple with the center coordinates (x,y).
    '''
    
    df = data.copy()
    city_box = df.geometry.values.total_bounds
    mean_x = (city_box[2] + city_box[0])/2
    mean_y = (city_box[3] + city_box[1])/2
    center_coordinates = (mean_x, mean_y)
    
    return center_coordinates

def surface_flipped(polygon):
    '''   
    Parameters
    ----------
    polygon : geometry as polygon.

    Returns
    -------
    flipped_polygon : geometry as polygon with coordinated order flipped
    '''
    
    # Reverse the order of the polygon's vertices
    exterior_ring = list(polygon.exterior.coords)[::-1]
    # Create a new polygon with the reversed exterior ring
    flipped_polygon = Polygon(exterior_ring)
    
    return flipped_polygon

def generate_buildings(zone_all, EGID_list, API_url, altitude_default=0,
                                 create_geometry_3D=False, calculate_volume_3D=False, 
                                 zone_floor=None, zone_roof=None, zone_wall=None):

    '''
    Parameters
    ----------
    zone_all : geoDataFrame
        geoDataFrame containing the buildings footprints from MO cadaster of the zone of interest.
    EGID_list : list
        list containing the EGID numbers of buildings connected to the DHN.
    API_url : url
        url adress for scraping processing.
    altitude_default : float, optional
        float number containing default altitude number when swissbuildings3D not provided or floor unavailable.
    create_geometry_3D : bool, optional
        bool signaling if swissbuildings3D is used to create the geometries.
    calculate_volume_3D : bool, optional
        bool signaling if swissbuildings3D is used to calculate the buildings volume.
    zone_floor, zone_roof, zone_wall : geoDataFrame, optional
        geoDataFrame containing surfaces from swissbuildings3D database.

    Returns
    -------
    all_footprints : geoDataFrame
        geoDataFrame containing all the buildings footprints from MO cadaster 
    buildings : geoDataFrame
        geoDataFrame containing the buildings informations. 
    '''
    
    floors_id = []
    roofs_id = []
    walls_id = []
    
    all_footprints = []
    buildings = []  
    EGID_column = 'RegBL_EGID'
    count = 0
    bid = 0    
    for MO_index in zone_all.index:
        if len(zone_all) > 1:
            print('GeoDataFrame progress: {:.2f}%'.format(100*count/(len(zone_all)-1)))
        else:
            print('GeoDataFrame progress: 100%')
        count += 1
        bid += 1
        
        row = zone_all.loc[MO_index]
        footprint = row.geometry
        EGID = row[EGID_column]

        buffer_distance = 1.01 #m
        footprint_buffered = footprint.buffer(buffer_distance)
        
        # Compare the Polygon's centroids with the MO footprint and extract the corresponding geometries
        floor = zone_floor[zone_floor.geometry.centroid.intersects(footprint_buffered)]
        roof = zone_roof[zone_roof.geometry.centroid.intersects(footprint_buffered)]
        wall = zone_wall[zone_wall.geometry.centroid.intersects(footprint_buffered)]
            
        # If surface component missing
        if len(floor)==0 or len(roof)==0 or len(wall)==0:
            envelope_type = '2.5D'
            floor_geometry_list = []
            roof_geometry_list = []
            wall_geometry_list = []
            altitude = altitude_default
            volume_3D_available = False

        else:
            # Set geometry type to generate envelope
            # MO footprint
            if not create_geometry_3D: 
                envelope_type = '2.5D' 
            # Swissbuildings3D
            else:
                envelope_type = '3D'
                
            volume_3D_available = True
            
            # Create 3D geometry to generate envelope or calculate 3D volume
            if create_geometry_3D or calculate_volume_3D:
                # Check if polygons already taken for another building and avoid them if the case
                # Add polygons to a list for further processing
                # Floor
                for i in floor.index:
                    if i in floors_id:
                        floor = floor.drop(index=i)
                    else: 
                        floors_id.append(i) 
                floor_geometry_list = floor.geometry.to_list()
                # Get floor altitude from 1st point of polygon list
                # If no floor found because all given to neighbor building, previous altitude passed
                try:
                    altitude = floor_geometry_list[0].exterior.coords[0][2]
                except:
                    altitude = altitude_default
                
                # Roof
                for i in roof.index:
                    if i in roofs_id:
                        roof = roof.drop(index=i)
                    else: 
                        roofs_id.append(i) 
                roof_geometry_list = roof.geometry.to_list()
                
                # Walls
                for i in wall.index:
                    if i in walls_id:
                        wall = wall.drop(index=i)
                    else: 
                        walls_id.append(i) 
                wall_geometry_list = wall.geometry.to_list()

            else:
                floor_geometry_list = []
                roof_geometry_list = []
                wall_geometry_list = []
                altitude = altitude_default
                volume_3D_available = False
                
        # Status by default for further calculations
        Simulate_status = False
        Year_default_values = False
        Type_default_values = False
        Floors_default_values = False
        Volume_default_values = False
        
        # Buildings connected to DHN, search for [construction_year, building_type, n_floors, SRE]
        if EGID in EGID_list:
            
            Simulate_status = True        
            # scraping API RegBL if building is connected to DHN
            url = API_url + str(EGID) + "_0"            
            try :
                response = requests.get(url=url)
                if response.status_code != 200: # no existing building for that EGID
                    print(f'Building n°{MO_index} EGID {EGID} not in RegBL')
                    Year_default_values = True
                    Floors_default_values = True
                    Type_default_values = True
                    Volume_default_values = True                      
                else:
                    regbl = response.json()            
                    # retrieving construction periode with RegBL code "gbaup"
                    try:
                        construction_period = int(regbl["feature"]["attributes"]["gbaup"])
                        construction_period = dicts.PERIODS[str(construction_period)]
                    except:
                        print(f'Building n°{MO_index} EGID {EGID} : error retrieving "gpaup" contruction period in RegBL')
                        Year_default_values = True                        
                    # retrieving type of building with RegBL code "gklas"    
                    try:
                        building_type_RegBL = int(regbl["feature"]["attributes"]["gklas"])
                        building_type = dicts.TYPE[str(building_type_RegBL)]
                        #no heating needs for these classes, suppose residential for simulation
                        if building_type == 3 or building_type == 10: 
                            Type_default_values = True
                    except:
                        print(f'Building n°{MO_index} EGID {EGID} : error retrieving "gklas" building class in RegBL')
                        Type_default_values = True                    
                    # retrieving number of floors with RegBL code "gastw"  
                    try:
                        n_floors = int(regbl["feature"]["attributes"]["gastw"])
                    except:
                        print(f'Building n°{MO_index} EGID {EGID} : error retrieving "gastw" number of floors in RegBL')
                        Floors_default_values = True
                    # retrieving SRE with RegBL code "gebf"  
                    try:
                        SRE = int(regbl["feature"]["attributes"]["gebf"])
                        # check SRE is a number
                        if SRE > 0:
                            pass
                        else:
                            SRE = 0   
                    except:
                        print(f'Building n°{MO_index} EGID {EGID} : error retrieving "gebf" SRE in RegBL')
                        Volume_default_values = True                                          
            except:
                print('Building n°{MO_index} EGID {EGID} : error with RegBL API request')
                Year_default_values = True
                Type_default_values = True
                Floors_default_values = True 
                Volume_default_values = True
    
        # Buildings in the scene, not simulated, search for [n_floors]        
        else: 
            Year_default_values = True
            Type_default_values = True
            Volume_default_values = True        
            try:
                EGID = int(EGID)
                # scrapping API RegBL if building is connected to DHN
                url = API_url + str(EGID) + "_0"            
                try :
                    response = requests.get(url=url)                
                    if response.status_code != 200: # no existing building for that EGID
                        Floors_default_values = True                   
                    else:
                        regbl = response.json()                 
                        # retrieving number of floors with RegBL code "gastw"  
                        try:
                            n_floors = int(regbl["feature"]["attributes"]["gastw"])
                        except:
                            Floors_default_values = True                                            
                except:
                    Floors_default_values = True            
            except:
                Floors_default_values = True              
        
        if Year_default_values == True: #by default, most recent building
            construction_year = max(dicts.THRESHOLDS.keys())
            wall_type, roof_type, floor_type, Ninf, glazing_u_value, glazing_g_value, glazing_ratio = dicts.THRESHOLDS[construction_year]          
        else:
            for year in dicts.THRESHOLDS.keys():
                if construction_period <= year:
                    construction_year = year
                    wall_type, roof_type, floor_type, Ninf, glazing_u_value, glazing_g_value, glazing_ratio = dicts.THRESHOLDS[construction_year]
                    break
        if Type_default_values == True: #by default, residential building
            building_type = 1    
        if Floors_default_values == True: #by default, 1 floor
            n_floors = 1 
    
        # Height of building
        floor_height = 2.73 #standard floor height (from Perez)
        height = n_floors*floor_height
        
        # Volume heated calculation
        t_floors = 0.2 #m thickness of building floor
        alpha_floor = 0.8 #fraction of interior surface relative to footprint
        if Volume_default_values == True :
            SRE = 0
            volume_MO = footprint.area*alpha_floor*(height-n_floors*t_floors)
        else:
            volume_MO = SRE*(floor_height-t_floors)
    
        # Area dedicated per person
        area_person = dicts.SURFACE[str(building_type)]
 
        # Minimum temperature setpoint   
        Tmin = dicts.TEMPERATURE[str(building_type)]
    
        # Create GeoDataFrames
        all_footprints.append({'bid':bid, 'egid':EGID, 
                               'geometry':footprint, 'height':height, 'altitude':altitude})
        
        buildings.append( 
            { 
            'bid':bid, 'egid':EGID, 'geometry':footprint, 'footprint':footprint, 'altitude':altitude, 'height':height,
            'floor':floor_geometry_list, 'roof':roof_geometry_list, 'wall':wall_geometry_list,
            'year':construction_year, 'area_person':area_person, 'n_floors' : n_floors,
            'wall_type':wall_type, 'roof_type':roof_type, 'floor_type':floor_type,  
            'Ninf':Ninf, 'glazing_u_value':glazing_u_value, 
            'glazing_g_value':glazing_g_value, 'glazing_ratio':glazing_ratio,
            'building_type':building_type, 'SRE_RegBL':SRE, 'Tmin':Tmin, 
            'volume_3D_available':volume_3D_available, 'volume_MO':volume_MO,
            'envelope_type':envelope_type, 'Simulate_status':Simulate_status
             }               
        )
    
    all_footprints = gpd.GeoDataFrame(all_footprints)
    buildings = gpd.GeoDataFrame(buildings)
    
    return all_footprints, buildings

def envelope_MO(all_footprints_ref, all_footprints, row, envelope_building, geometry_column='geometry'):
    '''
    Parameters
    ----------
    all_footprints_ref, all_footprints : geoDataFrame
        geoDataFrame containing all the buildings footprints from MO cadaster    
    row : Series
        Series containing all the informations necessary for the building envelope processing.
    envelope_building : geoDataFrame
        empty geoDataFrame with set up columns to be filled.

    Returns
    -------
    envelope : geoDataFrame
        geoDataFrame with the 3D envelope of walls, roof and floor processed from footprint. 
    '''
    
    # Define variables for creating the surfaces
    envelope = envelope_building.copy()
    bid = row['bid']
    footprint = row['footprint']
    glazing_ratio = row['glazing_ratio']
    glazing_g_value = row['glazing_g_value']
    glazing_u_value = row['glazing_u_value']
    floor_type = row['floor_type']
    roof_type = row['roof_type']
    wall_type = row['wall_type']       
    altitude = row['altitude']
    height = row['height']    
    
    # Get index of MO footprint corresponding to 'bid'    
    r_index = all_footprints[all_footprints["bid"]==bid].index.to_list()[0]
    
    # Create floor, class_id=33 in CitySim :
    floorpoints = list()     
    try:
        # Create list containing footprint vertices
        for pt in footprint.exterior.coords: 
            point = tuple((pt[0],  pt[1], altitude))
            
            floorpoints.append(point)
                
        # Take care of interior rings
        if len(all_footprints["interior"].loc[r_index]) == 0:
            pass
        else:
            previous_poly = shp.MultiPoint(footprint.exterior.coords)              
            for ring in all_footprints["interior"].loc[r_index]:
                # First, look for the closest points in the polygons to connect
                ring_multipoint = shp.MultiPoint(ring)
                nearest_ext, nearest_int = nearest_points(previous_poly, ring_multipoint)
                # Then put the ring's closest point as first
                nearest_int = nearest_int.coords[0] 
                int_point_index = ring.index(nearest_int)
                ring = ring[int_point_index:] + ring[:int_point_index + 1]
                # Finally insert the ring's points in the right index
                nearest_ext = tuple(nearest_ext.coords[0] + (altitude,))
                ext_point_index = floorpoints.index(nearest_ext)
                ring = [t + (altitude,) for t in ring]
                floorpoints = floorpoints[:ext_point_index + 1] + ring + floorpoints[ext_point_index:]
        floor = shp.Polygon(floorpoints)
        surface = {"bid" : bid, "geometry" : floor, "class_id" : 33, 
                   "glazing_ratio" : 0, "glazing_g_value" : glazing_g_value,
                   "glazing_u_value" : glazing_u_value, 
                   "openable_ratio" : 0, 'shortwave_reflectance' : 0.2,
                   "surface_type" : floor_type}
        df_surface = pd.DataFrame.from_dict([surface])
        envelope = pd.concat([envelope, df_surface], ignore_index=True)                            
    except:
        pass
    
    # Create roof, class_id=35 in CitySim :
    try:
        # Translate floor vertically by 'height' and flip it
        roof = translate(shp.Polygon(floorpoints[::-1]), zoff = height)
        surface = {"bid" : bid, "geometry" : roof, "class_id" : 35, 
                   "glazing_ratio" : 0, "glazing_g_value" : glazing_g_value,
                   "glazing_u_value" : glazing_u_value, 
                   "openable_ratio" : 0, "shortwave_reflectance" : 0.2,
                   "surface_type" : roof_type}
        df_surface = pd.DataFrame.from_dict([surface])
        envelope = pd.concat([envelope, df_surface], ignore_index=True)                            
    except:
        pass
    
    # Take care of overlapping walls and create patches (in CitySim common walls have to be removed)
    all_footprints["floor_union"] = all_footprints[geometry_column]
    for j in all_footprints_ref.index:
        linesect = all_footprints_ref[geometry_column].loc[j].intersection(all_footprints[geometry_column].loc[r_index])
        #case if the intersection is a single line (linestring format)
        if isinstance(linesect, shp.linestring.LineString):
                floor_union = all_footprints["floor_union"].loc[r_index].union(all_footprints_ref[geometry_column].loc[j])
                all_footprints["floor_union"].loc[r_index] = orient(floor_union, sign=1.0)
                # all_footprints["floor_union"].loc[r_index] = floor_union
                if (all_footprints["height"].loc[r_index] + all_footprints["altitude"].loc[r_index]) > (all_footprints_ref["height"].loc[j] + all_footprints_ref["altitude"].loc[j]):
                    x_one = linesect.coords[0][0]
                    x_two = linesect.coords[1][0]
                    y_one = linesect.coords[0][1]
                    y_two = linesect.coords[1][1]
                    z_one = (all_footprints_ref["height"].loc[j] + all_footprints["altitude"].loc[j])
                    z_two = (all_footprints["height"].loc[r_index] + all_footprints["altitude"].loc[r_index])
                    if abs(z_one - z_two) > 0.001: # avoid very small vertical surfaces
                        patchlist = [[x_two, y_two, z_one], [x_two, y_two, z_two], [x_one, y_one, z_two], [x_one, y_one, z_one]]
                        patchpoly = shp.Polygon(patchlist[::-1])
                        # patchpoly = shp.Polygon(patchlist) 
                        surface = {"bid" : bid, "geometry" : patchpoly, "class_id" : 34, 
                                    "glazing_ratio" : glazing_ratio, 
                                    "glazing_g_value" : glazing_g_value,
                                    "glazing_u_value" : glazing_u_value,
                                    "openable_ratio" : 0.5, 
                                    "shortwave_reflectance" : 0.2,
                                    "surface_type" : wall_type}
                        df_surface = pd.DataFrame.from_dict([surface])                        
                        envelope = pd.concat([envelope, df_surface], ignore_index=True)
        #case if the intersection is a multisegment line, in a L or U shape for example (multilinestring format)
        elif isinstance(linesect, shp.multilinestring.MultiLineString):
                floor_union = all_footprints["floor_union"].loc[r_index].union(all_footprints_ref[geometry_column].loc[j])
                all_footprints["floor_union"].loc[r_index] = orient(floor_union, sign=1.0)
                # all_footprints["floor_union"].loc[r_index] = floor_union
                if (all_footprints["height"].loc[r_index] + all_footprints["altitude"].loc[r_index]) > (all_footprints_ref["height"].loc[j] + all_footprints_ref["altitude"].loc[j]):
                    for l in linesect.geoms:                           
                        x_one = l.coords[0][0]
                        x_two = l.coords[1][0]
                        y_one = l.coords[0][1]
                        y_two = l.coords[1][1]                            
                        z_one = (all_footprints_ref["height"].loc[j] + all_footprints_ref["altitude"].loc[j])
                        z_two = (all_footprints["height"].loc[r_index] + all_footprints["altitude"].loc[r_index])
                        if abs(z_one - z_two) > 0.001: # avoid very small vertical surfaces
                            patchlist = [[x_two, y_two, z_one], [x_two, y_two, z_two], [x_one, y_one, z_two], [x_one, y_one, z_one]]
                            patchpoly = shp.Polygon(patchlist[::-1])
                            # patchpoly = shp.Polygon(patchlist)
                            surface = {"bid" : bid, "geometry" : patchpoly, "class_id" : 34, 
                                       "glazing_ratio" : glazing_ratio,
                                       "glazing_g_value" : glazing_g_value,
                                       "glazing_u_value" : glazing_u_value,
                                       "openable_ratio" : 0.5, 
                                       "shortwave_reflectance" : 0.2,
                                       "surface_type" : wall_type}
                            df_surface = pd.DataFrame.from_dict([surface])                        
                            envelope = pd.concat([envelope, df_surface], ignore_index=True)
                                    
    # Create walls, class_id=34 in CitySim : 
    splitpoints = shp.MultiPoint(list(all_footprints["floor_union"].loc[r_index].exterior.coords))
    for i in range(len(splitpoints.geoms)-1):
        x_one = splitpoints.geoms[i].x
        x_two = splitpoints.geoms[i+1].x
        y_one = splitpoints.geoms[i].y
        y_two = splitpoints.geoms[i+1].y
        line = shp.LineString([(splitpoints.geoms[i].x, splitpoints.geoms[i].y), (splitpoints.geoms[i+1].x, splitpoints.geoms[i+1].y)])
        wallpoints = [[x_one, y_one, altitude], [x_one, y_one, altitude+height], [x_two, y_two, altitude+height], [x_two, y_two, altitude]]
        wall = shp.Polygon(wallpoints[::-1])
        # wall = shp.Polygon(wallpoints)
        if all_footprints[geometry_column].loc[r_index].exterior.contains(line):
            surface = {"bid" : bid, "geometry" : wall, "class_id" : 34, 
                       "glazing_ratio" : glazing_ratio, 
                       "glazing_g_value" : glazing_g_value,
                       "glazing_u_value" : glazing_u_value,
                       "openable_ratio" : 0.5, 
                       "shortwave_reflectance" : 0.2,
                       "surface_type" : wall_type} 
            df_surface = pd.DataFrame.from_dict([surface])
            envelope = pd.concat([envelope, df_surface], ignore_index=True)
    
    # Create interior walls
    for ring in all_footprints["interior"].loc[r_index]:
        for i in range(len(ring)-1):
            x_one = ring[i][0]
            x_two = ring[i+1][0]
            y_one = ring[i][1]
            y_two = ring[i+1][1]
            wallpoints = [[x_one, y_one, altitude], [x_one, y_one, altitude+height], [x_two, y_two, altitude+height], [x_two, y_two, altitude]]
            wall = shp.Polygon(wallpoints[::-1])
            # wall = shp.Polygon(wallpoints)
            surface = {"bid" : bid, "geometry" : wall, "class_id" : 34, 
                       "glazing_ratio" : glazing_ratio, 
                       "glazing_g_value" : glazing_g_value,
                       "glazing_u_value" : glazing_u_value,
                       "openable_ratio" : 0.5, 
                       "shortwave_reflectance" : 0.2,
                       "surface_type" : wall_type}
            df_surface = pd.DataFrame.from_dict([surface])
            envelope = pd.concat([envelope, df_surface], ignore_index=True)

    return envelope
    

def envelope_3D(row, envelope_building):
    '''
    Parameters
    ----------
    row : Series
        Series containing all the informations necessary for the building envelope processing.
    envelope_building : geoDataFrame
        empty geoDataFrame with set up columns.

    Returns
    -------
    envelope : geoDataFrame
        geoDataFrame with the 3D envelope of walls, roof and floor processed from 3D surfaces.
    volume : DataFrame
        DataFrame containing the calculated 3D enclosed volume.
    '''
    
    # Define variables for creating the surfaces
    envelope = envelope_building.copy()
    bid = row['bid']    
    altitude = row['altitude']
    floor_geometry = row['floor']
    roof_geometry = row['roof']
    wall_geometry = row['wall']
    glazing_ratio = row['glazing_ratio']
    glazing_g_value = row['glazing_g_value']
    glazing_u_value = row['glazing_u_value']
    floor_type = row['floor_type']
    roof_type = row['roof_type']
    wall_type = row['wall_type'] 
    Simulate_status = row['Simulate_status']
    
    # Create floor, class_id=33 in CitySim :              
    try: 
        for i in range(len(floor_geometry)):
            surface = {"bid" : bid, "geometry" : floor_geometry[i], "class_id" : 33, 
                       "glazing_ratio" : 0, "glazing_g_value" : glazing_g_value,
                       "glazing_u_value" : glazing_u_value, 
                       "openable_ratio" : 0, 'shortwave_reflectance' : 0.2,
                       "surface_type" : floor_type}
            df_surface = pd.DataFrame.from_dict([surface])
            envelope = pd.concat([envelope, df_surface], ignore_index=True)                            
    except:
        print('error floor : bid', bid)
        pass
    
    # Create roof, class_id=35 in CitySim : 
    try:
        for i in range(len(roof_geometry)):
            roof_flipped = surface_flipped(roof_geometry[i])
            surface = {"bid" : bid, "geometry" : roof_flipped, "class_id" : 35, 
                        "glazing_ratio" : 0, "glazing_g_value" : glazing_g_value,
                        "glazing_u_value" : glazing_u_value, 
                        "openable_ratio" : 0, "shortwave_reflectance" : 0.2,
                        "surface_type" : roof_type}
            df_surface = pd.DataFrame.from_dict([surface])
            envelope = pd.concat([envelope, df_surface], ignore_index=True)                            
    except:
        print('error roof : bid', bid)
        pass
           
    # Create walls, class_id=34 in CitySim : 
    try:
        for i in range(len(wall_geometry)):
            wall_flipped = surface_flipped(wall_geometry[i])
            surface = {"bid" : bid, "geometry" : wall_flipped, "class_id" : 34,
                            "glazing_ratio" : glazing_ratio, 
                            "glazing_g_value" : glazing_g_value,
                            "glazing_u_value" : glazing_u_value,
                            "openable_ratio" : 0.5, 
                            "shortwave_reflectance" : 0.2,
                            "surface_type" : wall_type}
            df_surface = pd.DataFrame.from_dict([surface])
            envelope = pd.concat([envelope, df_surface], ignore_index=True)                            
    except:
        print('error wall : bid', bid)
        pass    

    if Simulate_status == True:
        # Calculate enclosed volume of polygon triangles     
        volume = 0.0    
        for surface in envelope.index:
            triangle = envelope['geometry'].loc[surface]
            p1,p2,p3,_ = triangle.exterior.coords
            # Calculate projected area
            mean_z = np.mean([p1[2],p2[2],p3[2]])           
            z = [0,0,1] 
            v1 = np.array(p1)
            v2 = np.array(p2)
            v3 = np.array(p3)
            n = np.cross(v2-v1, v3-v1)
            projected_triangle_area = 0.5*np.dot(n,z)
            # Calculate volume between mean z and altitude of the building's floor
            #mean_z = float(mean_z)
            #altitude = float(altitude)
            volume += projected_triangle_area*(mean_z-altitude)
    else:
        volume = 0
    
    return envelope, volume
    
def generate_envelope(footprints, buildings, calculate_volume_3D, geometry_column = 'geometry',
                       construction_year_column = 'year'):
    '''
    Parameters
    ----------
    footprint : geoDataFrame
        geoDataFrame containing all the buildings footprints from MO cadaster.
    buildings : geoDataFrame
        geoDataFrame containing the buildings informations and surfaces from swissbuildings3D if provided.
    calculate_volume_3D : bool, optional
        bool signaling if swissbuildings3D is used to calculate the buildings volume.
    geometry_column : str, optional
        name of the column containing the geometries. The default is 'geometry'.
    construction_year_column : str, optional
        name of the column containing the year of construction. The default is 'year'.

    Returns
    -------
    envelope : geoDataFrame
        geoDataFrame with the 3D geometries of walls, roof and floor created. 
    buildings_volume_3D : DataFrame
        DataFrame containing the calculated 3D enclosed volume
    center_coordinates : tuple
        tuple containing the coordinates of the center of the scene
    '''
            
    # Work on copies
    footprints_copy = footprints.copy()   
    buildings_copy = buildings.copy()
    
    
    # Simplify geometries to take care of curves, and convert to polygon, with small tolerance so that it preserves intersections 
    footprints_copy[geometry_column] = footprints_copy[geometry_column].simplify(0.05, preserve_topology=True)
    
    # Create reference geodataframe for envelope generation
    footprints_ref = footprints_copy.copy()
    
    # Orient polygons clockwise
    footprints_copy = fix_ccw_polygons(footprints_copy, inplace=True)

    # Create an empty envelope geoDataFrame to be populated
    envelope = gpd.GeoDataFrame(columns=['bid', 'geometry', 'class_id', 
                                         'glazing_ratio', 'glazing_g_value',
                                         'glazing_u_value', 'openable_ratio',
                                         'shortwave_reflectance'])
    
    # Check for footprints interior rings
    footprints_copy["interior"] = ''
    for r in footprints_copy.index:
        interiors_list = []
        for ring in footprints_copy[geometry_column].loc[r].interiors:
            ring_coords = list(ring.coords)
            interiors_list.append(ring_coords)
        footprints_copy["interior"].loc[r] = interiors_list

    # Create the surfaces: 
    count = 0
    buildings_volume_3D = []
    for r in buildings_copy.index:
        if len(buildings_copy) > 1:
            print('Surfacing progress: {:.2f}%'.format(100*count/(len(buildings_copy)-1)))
        else:
            print('Surfacing progress: 100%')
        count += 1
        volume = 0
        
        envelope_empty = gpd.GeoDataFrame(columns=['bid', 'geometry', 'class_id', 
                                             'glazing_ratio', 'glazing_g_value',
                                             'glazing_u_value', 'openable_ratio',
                                             'shortwave_reflectance'])
        
        # Define variables for creating the surfaces
        row = buildings_copy.loc[r]
        bid = row['bid']
        envelope_type = row['envelope_type']
        volume_3D_available = row['volume_3D_available']
        footprint = row['footprint']
        
        # Create the surfaces of the building's envelope
        # If use of swissbuildings3D sufaces
        if envelope_type == '3D':
            envelope_building, volume = envelope_3D(row, envelope_empty)
        # If use of MO cadaster footprint           
        elif envelope_type == '2.5D':
            # Skip buildings with footprint smaller than 1 m2 
            if footprint.area >= 1:
                envelope_building = envelope_MO(footprints_ref, footprints_copy, row, envelope_empty)
                if calculate_volume_3D and volume_3D_available:
                    _, volume = envelope_3D(row, envelope_empty)
                
        # Calculate number of occupants
        area_person = row['area_person']
        SRE = row['SRE_RegBL']
        footprint = row['footprint']
        n_floors = row['n_floors']

        floor_height = 2.73 #standard floor height (from Perez)
        t_floors = 0.2 #m thickness of building floor
        alpha_floor = 0.8 #fraction of interior surface relative to footprint

        # Take the absolute value of the volume with correction factor
        volume = abs(volume)*alpha_floor
        
        if volume == 0:
            if SRE != 0:
                total_area = SRE
            else:   
                total_area = footprint.area*alpha_floor*n_floors
        else:
            total_area = volume/(floor_height-t_floors)
        n_occupants = math.ceil(total_area/area_person) 
        
        
        # Create dataframe with 3D volume and number of occupants
        buildings_volume_3D.append({"bid" : bid, 
                                    "volume_3D" : volume, "n_occupants" : n_occupants})
        
        # Append building's envelope to general dataframe
        envelope = pd.concat([envelope, envelope_building], ignore_index=True)
        
    buildings_volume_3D = pd.DataFrame(buildings_volume_3D)
    
    return envelope, buildings_volume_3D


#jjj
def update_z_coordinate_from_ground(point, ground_polygons):
    """Update the z-coordinate of a point based on ground polygons (working in 2D)."""
    x, y = point.x, point.y
    point_2d = Point(x, y)  # version 2D du point (sans z)

    for poly in ground_polygons:
        # Transformer polygon 3D en 2D pour le test
        poly_2d = Polygon([(p[0], p[1]) for p in poly.exterior.coords])
        if poly_2d.contains(point_2d):
            # Moyenne des 3 premiers z du sol pour l'altitude
            coords = list(poly.exterior.coords)[:3]
            z = np.mean([c[2] for c in coords])
            return Point(x, y, z)

    # Si aucun sol trouvé, on garde z original
    return point


def generate_pedestrian(pedestrian, buildings, ground_data):
    # adding bid and egid columns
    max_bid = buildings['bid'].max()
    bid = range(max_bid + 1, max_bid + 1 + len(pedestrian))
    pedestrian['bid'] = bid
    pedestrian['egid'] = ''
    
    # Adding height to pedestrian center coordinate 
    #pedestrian['geometry'] = pedestrian['geometry'].apply(lambda point: update_z_coordinate_from_ground(point, ground_data['geometry']))
    
    pedestrian['geometry'] = pedestrian['geometry'].apply(lambda point: update_z_coordinate(point, ground_data['geometry']))
    pedestrian = pedestrian.to_crs(2056)
    
    """ Pedestrian creation """
    # circle radius
    radius = 0.17/2 # [m]
    hg = 1.1 # height from the ground [m]
    h = 0.4 # heigth of pedestrian [m]
    # data_frame to store polygones
    data = []

    for idx, row in pedestrian.iterrows():
        center_pedestrian = row['geometry']
        bid = row['bid']

        floor_points = create_floor_points(center_pedestrian, radius, hg)
        roof_points = [Point(point.x, point.y, point.z + h) for point in floor_points]
        floor_polygon_points = [(point.x, point.y, point.z) for point in floor_points]
        roof_polygon_points = [(point.x, point.y, point.z) for point in roof_points]
        floor_polygon = Polygon(floor_polygon_points)
        roof_polygon = Polygon(roof_polygon_points)

        walls = []
        for i in range(len(floor_points) - 1):
            wall_points = [
                (floor_points[i].x, floor_points[i].y, floor_points[i].z),
                (floor_points[i + 1].x, floor_points[i + 1].y, floor_points[i + 1].z),
                (roof_points[i + 1].x, roof_points[i + 1].y, roof_points[i + 1].z),
                (roof_points[i].x, roof_points[i].y, roof_points[i].z),
                (floor_points[i].x, floor_points[i].y, floor_points[i].z)
            ]
            walls.append(Polygon(wall_points))

        # Close last wal
        wall_points = [
            (floor_points[-1].x, floor_points[-1].y, floor_points[-1].z),
            (floor_points[0].x, floor_points[0].y, floor_points[0].z),
            (roof_points[0].x, roof_points[0].y, roof_points[0].z),
            (roof_points[-1].x, roof_points[-1].y, roof_points[-1].z),
            (floor_points[-1].x, floor_points[-1].y, floor_points[-1].z)
        ]
        walls.append(Polygon(wall_points))

        # Modify the floor polygon points
        modified_floor_points = [(point.x, point.y, point.z) for point in floor_points]
        modified_floor_points_reversed = modified_floor_points[::-1]
        modified_floor_polygon = Polygon(modified_floor_points_reversed)

        data.append({'bid': bid, 'geometry': modified_floor_polygon, 'class_id': 3})
        data.append({'bid': bid, 'geometry': roof_polygon, 'class_id': 2})
        for wall in walls:
            data.append({'bid': bid, 'geometry': wall, 'class_id': 3})

    pedestrian_envelope = gpd.GeoDataFrame(data)

    """ Calculate pedestrian volume """
    volumes = pedestrian_envelope.groupby('bid').apply(lambda g: calculate_volume_2(g, h)).reset_index(name='volume')
    pedestrian = pedestrian.merge(volumes, on='bid', how='left')
    
    
    return pedestrian, pedestrian_envelope


def interpolate_z(point, polygon):
    if polygon.geom_type == 'Polygon' and len(polygon.exterior.coords) >= 3:
        x1, y1, z1 = polygon.exterior.coords[0]
        x2, y2, z2 = polygon.exterior.coords[1]
        x3, y3, z3 = polygon.exterior.coords[2]
        z = (z1 + z2 + z3) / 3
        return Point(point.x, point.y, z)
    else:
        return point

def update_z_coordinate(point, polygons):
    """Mets à jour le Z du point à partir du polygone le plus proche dans ground_data"""
    min_dist = float('inf')
    nearest_poly = None

    for poly in polygons:
        dist = point.distance(poly)
        if dist < min_dist:
            min_dist = dist
            nearest_poly = poly

    if nearest_poly is not None:
        return interpolate_z(point, nearest_poly)
    
    # fallback: keep Z at 0
    return point

# Create 8 points on circle 
def create_floor_points(center, radius, hg):
    points = []
    for i in range(8):
        theta = i * (np.pi / 4)  # Increment angle by pi/4
        x = center.x + radius * np.cos(theta)
        y = center.y + radius * np.sin(theta)
        z = center.z + hg  # z-coordinate remains the same
        points.append(Point(x, y, z))
    return points

# Calculate pedestrian volume
def calculate_volume(group):
    floor = None
    roof = None
    walls = []

    for idx, geom in group.iterrows():
        if geom['class_id'] == 1:
            floor = geom['geometry']
        elif geom['class_id'] == 2:
            roof = geom['geometry']
        elif geom['class_id'] == 3:
            walls.append(geom['geometry'])

    if floor and roof:
        height = roof.exterior.coords[0][2] - floor.exterior.coords[0][2]
        floor_area = floor.area
        volume = floor_area * height
        return volume
    else:
        return 0
    
    
def calculate_volume_2(group, h):
    # Find the geometry corresponding to the roof (class_id == 2)
    roof_polygon = group[group['class_id'] == 2].iloc[0]['geometry']
    volume = roof_polygon.area * h
    return volume



def generate_tree(trees, ground_data, start_tid = 0):
    
    #trees['tid'] = range(len(trees))
    trees['tid'] = range(start_tid, start_tid + len(trees))
    # Apply this function to each point in pedestrian['geometry']
    trees['geometry'] = trees['geometry'].apply(lambda point: update_z_coordinate(point, ground_data['geometry']))
    # data_frame to store polygones
    data = []
    
    # Assigner des valeurs par défaut pour les colonnes vides
    trees['H_TOTALE'].fillna(7.5, inplace=True)
    trees['H_TRONC'].fillna(2.5, inplace=True)
    trees['D_1M'].fillna(14, inplace=True)  # 0.14 m * 100
    trees['D_COURONNE'].fillna(3, inplace=True)  # 3 m * 2
    trees.loc[trees['D_COURONNE'] == 0, 'D_COURONNE'] = 3 # 3m
    trees['NOM_COMPL'].fillna("arbre", inplace=True)  # désigne cas standard
    
    for idx, row in trees.iterrows():
        center_tree = row['geometry']
        tid = row['tid']
        height = row['H_TOTALE'] # [m]
        r_couronne = row['D_COURONNE']/2 # [m] crown radius
        r_trunc = row['D_1M']/200 # [m] trunc radius
        
        
        bottom_points = create_base_points_tree(center_tree, r_trunc)
        apex_points = [Point(point.x, point.y, point.z + height) for point in bottom_points]
        
        # trunc creation
        trunc_surface = []
        for i in range(len(bottom_points) - 1):
            trunc_surface_points = [
                (bottom_points[i].x, bottom_points[i].y, bottom_points[i].z),
                (bottom_points[i + 1].x, bottom_points[i + 1].y, bottom_points[i + 1].z),
                (apex_points[i + 1].x, apex_points[i + 1].y, apex_points[i + 1].z),
                (apex_points[i].x, apex_points[i].y, apex_points[i].z),
                (bottom_points[i].x, bottom_points[i].y, bottom_points[i].z)
            ]
            trunc_surface.append(Polygon(trunc_surface_points))

        # Close last wall
        trunc_surface_points = [
            (bottom_points[-1].x, bottom_points[-1].y, bottom_points[-1].z),
            (bottom_points[0].x, bottom_points[0].y, bottom_points[0].z),
            (apex_points[0].x, apex_points[0].y, apex_points[0].z),
            (apex_points[-1].x, apex_points[-1].y, apex_points[-1].z),
            (bottom_points[-1].x, bottom_points[-1].y, bottom_points[-1].z)
        ]
        trunc_surface.append(Polygon(trunc_surface_points))
        
        # leaf_creation
        leaf_surface = []
        
        x_centroid = center_tree.x
        y_centroid = center_tree.y
        z_centroid = center_tree.z 
        z = (height + z_centroid)
        
        leaf_surface_points = [
            (x_centroid + r_couronne, y_centroid, z),
            (x_centroid + r_couronne/2, y_centroid + r_couronne*0.866, z),
            (x_centroid - r_couronne/2, y_centroid + r_couronne*0.866, z),
            (x_centroid - r_couronne, y_centroid, z),
            (x_centroid - r_couronne/2, y_centroid - r_couronne*0.866, z),
            (x_centroid + r_couronne/2, y_centroid - r_couronne*0.866, z)
        ]
        leaf_surface.append(Polygon(leaf_surface_points))
        
        for trunc in trunc_surface:
            data.append({'tid': tid, 'geometry': trunc, 'class_id': 1})
            
        for leaf in leaf_surface:
            data.append({'tid': tid, 'geometry': leaf, 'class_id': 2})

    trees_envelope = gpd.GeoDataFrame(data)
    
    return trees, trees_envelope, start_tid + len(trees)
    
    
def create_base_points_tree(center, radius):
    points = []
    for i in range(6):
        theta = i * (np.pi / 3)  # Increment angle by pi/4
        x = center.x + radius * np.cos(theta)
        y = center.y + radius * np.sin(theta)
        z = center.z  # z-coordinate remains the same
        points.append(Point(x, y, z))
    return points

def update_pedestrian_z(pedestrian, ground_data):
    """ Met à jour la coordonnée Z des piétons pour qu'ils soient bien posés sur le sol. """
    updated_geometries = []

    for idx, row in pedestrian.iterrows():
        point = row['geometry']
        x, y = point.x, point.y

        # Trouver le sol sous le piéton
        matching_ground = ground_data[ground_data.geometry.contains(Point(x, y))]

        if not matching_ground.empty:
            # Moyenne des 3 premiers points du polygone (approx Z moyen)
            coords = list(matching_ground.iloc[0].geometry.exterior.coords)[:3]
            z_ground = np.mean([coord[2] for coord in coords])
        else:
            z_ground = point.z  # fallback

        updated_geometries.append(Point(x, y, z_ground))

    pedestrian['geometry'] = updated_geometries
    return pedestrian

def update_tree_z(trees, ground_data):
    """ Met à jour la coordonnée Z des arbres pour qu'ils soient bien posés sur le sol. """
    updated_geometries = []

    for idx, row in trees.iterrows():
        point = row['geometry']
        x, y = point.x, point.y

        # Trouver le sol sous l'arbre
        matching_ground = ground_data[ground_data.geometry.contains(Point(x, y))]

        if not matching_ground.empty:
            # Moyenne des 3 premiers points du polygone
            coords = list(matching_ground.iloc[0].geometry.exterior.coords)[:3]
            z_ground = np.mean([coord[2] for coord in coords])
        else:
            z_ground = point.z  # fallback

        updated_geometries.append(Point(x, y, z_ground))

    trees['geometry'] = updated_geometries
    return trees


def generate_umbrellas(umbrellas, ground_data,
                       start_uid=0,
                       radius=1.0,       # rayon du parapluie en mètres
                       thickness=0.02,   # épaisseur du disque
                       height=3.0,       # hauteur du bas du parapluie au-dessus du sol
                       n_segments=32     # nombre de points pour approximer le cercle
                       ):
    
    umbrellas = umbrellas.copy()
    umbrellas['uid'] = range(start_uid, start_uid + len(umbrellas))
    umbrellas['geometry'] = umbrellas['geometry'].apply(
        lambda pt: update_z_coordinate(pt, ground_data['geometry'])
    )
    
    envelope_records = []
    
    for idx, row in umbrellas.iterrows():
        center = row['geometry']
        uid = row['uid']
        base_z = center.z + height
        top_z = base_z + thickness
        
        # 2) Générer les cercles top et bottom
        angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
        bottom_pts = [(center.x + radius*np.cos(a),
                       center.y + radius*np.sin(a),
                       base_z) for a in angles]
        top_pts = [(x, y, top_z) for (x, y, _) in bottom_pts]
        
        bottom_poly = Polygon(bottom_pts)
        top_poly    = Polygon(top_pts[::-1])  # inversé pour orientation
        
        envelope_records.append({
            'uid': uid, 'geometry': bottom_poly, 'class_id': 10  # 10=bottom
        })
        envelope_records.append({
            'uid': uid, 'geometry': top_poly,    'class_id': 11  # 11=top
        })
        
        # 3) Générer les faces latérales
        for i in range(n_segments):
            p0b = bottom_pts[i]
            p1b = bottom_pts[(i+1) % n_segments]
            p1t = top_pts   [(i+1) % n_segments]
            p0t = top_pts   [i]
            side_poly = Polygon([p0b, p1b, p1t, p0t])
            envelope_records.append({
                'uid': uid, 'geometry': side_poly, 'class_id': 12  # 12=side
            })
    
    umbrella_envelope = gpd.GeoDataFrame(envelope_records, columns=['uid','geometry','class_id'])
    return umbrellas, umbrella_envelope


def generate_tarps_from_polygons(
        tarps,               # GeoDataFrame : géom. = Polygon 2D (x,y)
        ground_data,         # GeoDataFrame : polygones du sol
        start_tid=0,
        height=5.0,          # distance sol → face inférieure (m)
        thickness=0.05       # 0 → membrane sans épaisseur
    ):
    """
    Soulève chaque bâche (Polygon) au‑dessus du sol et crée ses surfaces 3D.

    Retourne :
      - tarps_df  : GeoDataFrame d’origine enrichi de 'tid'
      - surfaces  : GeoDataFrame des faces (class_id : 20=bottom, 21=top, 22=side)
    """
    tarps = tarps.copy()
    tarps['tid'] = range(start_tid, start_tid + len(tarps))

    records = []

    for idx, row in tarps.iterrows():
        poly_2d = row['geometry']
        tid     = row['tid']

        # 1) altitude du sol au centre
        center   = Point(poly_2d.centroid.x, poly_2d.centroid.y)
        center_3 = update_z_coordinate(center, ground_data['geometry'])
        base_z   = center_3.z + height

        # 2) face inférieure (bottom)
        bottom_coords = [(x, y, base_z) for (x, y) in poly_2d.exterior.coords[:-1]]
        bottom_poly   = Polygon(bottom_coords)
        records.append({'tid': tid, 'geometry': bottom_poly, 'class_id': 20})

        # 3) face supérieure (top) si épaisseur > 0
        if thickness > 0:
            top_poly = translate(bottom_poly, zoff=thickness)
            records.append({'tid': tid, 'geometry': top_poly, 'class_id': 21})

            # 4) côtés (side‑walls) pour fermer le volume
            coords_bot = list(bottom_poly.exterior.coords)
            coords_top = list(top_poly.exterior.coords)
            for i in range(len(coords_bot) - 1):
                p0b, p1b = coords_bot[i], coords_bot[i+1]
                p1t, p0t = coords_top[i+1], coords_top[i]
                side = Polygon([p0b, p1b, p1t, p0t])
                records.append({'tid': tid, 'geometry': side, 'class_id': 22})

    surfaces = gpd.GeoDataFrame(records, columns=['tid', 'geometry', 'class_id'])
    return tarps, surfaces

# helper pour remonter chaque vertex en z=sol+height
def translate_line_z(line, ground_polys, height):
    new_pts = []
    for x,y in line.coords:
        p = Point(x, y)
        p3 = update_z_coordinate(p, ground_polys)
        new_pts.append((p3.x, p3.y, p3.z + height))
    return type(line)(new_pts)


def generate_parapluies_from_polygons(
        polygons,           # GeoDataFrame : géom. = Polygon 2D (x,y)
        ground_data,        # GeoDataFrame : polygones du sol
        start_tid=0,
        height=5.0,         # distance sol → centre du parapluie
        radius=1.0,         # rayon de chaque parapluie (m)
        spacing=0.8         # espacement entre parapluies (m)
    ):
    """
    Génère des disques ("parapluies") régulièrement espacés à l’intérieur des polygones.

    Retourne :
      - parapluies_df  : GeoDataFrame des parapluies (avec leur centre)
      - surfaces       : GeoDataFrame des disques 3D (class_id = 23)
    """


    records = []
    centers = []

    tid = start_tid

    for idx, row in polygons.iterrows():
        poly = row['geometry']
        circle_centers = polygon_to_circles(poly, radius=radius, spacing=spacing)

        for center_2d in circle_centers:
            center_3d = update_z_coordinate(center_2d, ground_data['geometry'])
            base_z = center_3d.z + height
            center_3d_elevated = Point(center_3d.x, center_3d.y, base_z)

            # Génère un disque 2D à z constant
            num_points = 16
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_coords = [(center_3d.x + radius * np.cos(a),
                              center_3d.y + radius * np.sin(a),
                              base_z) for a in angles]
            circle = Polygon(circle_coords)

            records.append({
                'tid': tid,
                'geometry': circle,
                'class_id': 23
            })

            centers.append({
                'tid': tid,
                'geometry': Point(center_3d.x, center_3d.y)
            })

            tid += 1

    surfaces = gpd.GeoDataFrame(records, columns=['tid', 'geometry', 'class_id'])
    parapluies_df = gpd.GeoDataFrame(centers, columns=['tid', 'geometry'])

    return parapluies_df, surfaces


def polygon_to_circles(poly, radius=1.0, spacing=0.8):
    """
    Remplit un polygone 2D avec des centres de cercles régulièrement espacés.

    Retourne une liste de shapely Point.
    """
    circles = []
    xmin, ymin, xmax, ymax = poly.bounds

    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            pt = Point(x, y)
            if poly.contains(pt):
                circles.append(pt)
            y += spacing
        x += spacing
    return circles