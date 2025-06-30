
# Urban Heat Island and Pedestrian Comfort in Valais (CH)
An open-source framework for quantifying Urban Heat Islands (UHIs) and Pedestrian Comfort in Monthey and Conthey, Valais, Switzerland.

## Usage of codes

### Open-source files and import to QGIS
Choose the geographical area (Monthey or Conthey) for the study.  
Step 1: Download required files  
Step 2: Import them to QGIS.

#### SwissBuildings3D
- Geodatabase format from https://www.swisstopo.admin.ch/en/geodata/landscape/buildings3d3.html#download  
- Import *Floor*, *Roof*, *Wall* layers

#### MO Cadaster
- GeoPackage format from https://geodienste.ch/services/av  
- Import *Couverture_du_sol* layer  
- Filter by `"Genre" IN (0, 1, 2, 3, 4, 5)`  
- Join with RegBL dataset from https://www.housing-stat.ch/fr/madd/public.html to add the `RegBL_EGID` identifiers using the QGIS spatial join tool

#### Ground Surface
- XYZ file format from https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html  
- Choose resolution of 2 meter or 0.5 meter and load it with `pandas.read_table`

#### Ground Types
- GeoPackage format from https://www.swisstopo.admin.ch/de/landschaftsmodell-swisstlm3d  
- Import `tlm_bb_bodenbedeckung` layer, filter by `"objektart" IN ('Wald', 'Gehoelzflaeche', 'Gebueschwald', 'Wald offen')`  
- Import `tlm_strassen_strasse` layer, filter by `"objektart" NOT IN ('Verbindung','Platz')`

### QGIS
Create a new GeoPackage file. Export layers with the "export features" option of QGIS (without the "fid" field), using the following layer names:

- `zone_cad`: MO features of buildings connected to district heating network (DHN)  
- `zone_tout`: MO features of all buildings in the study area  
- `centrale`: Point feature for thermal heating station coordinates  
- `floor`: SwissBuildings3D features of floors  
- `wall`: SwissBuildings3D features of walls  
- `roof`: SwissBuildings3D features of roofs  
- `streets`: SwissTLM3D features of streets  
- `green`: Features of existing green areas  
- `trees`: Existing tree positions (points)  
- `pedestrian`: Existing pedestrian positions (points)

### Layers for creating mitigation scenarios
- `sidewalk`: Polygons representing sidewalks (specific to Monthey)  
- `tarps`: Polygons representing shading structures (tarps and umbrellas)  
- `soil_green`: Polygons representing modifiable green spaces  
- `new_trees`: Points representing locations for new trees

## Code custom modifications
In addition to the newly created GeoPackage, a climate file (.cli) and a horizon file (.hor) must be provided and placed in the directory where the code repository was cloned.

Before running `main_code.py`, modify the following parameters directly in the script. Each modification is marked with `#TODO`:

- `gpkg_filepath = r"---.gpkg"`: Path to the GeoPackage containing necessary layers  
- `create_geometry_3D = True/False` (default: `False`): Enables detailed 3D geometry simulation (longer runtime)  
- `calculate_volume_3D = True/False` (default: `False`): Enables volume calculations from SwissBuildings3D geometries  
- `citysim_filepath = r"---/CitySim.exe"`: Path to CitySim solver executable  
- `directory_path = r"---"`: Output directory to store simulation results  
- `climate_file = r"---.cli"`: Path to the climate file  
- `horizon_file = r"---.hor"`: Path to the horizon file

## Python libraries
Required Python libraries are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Results
The generated `.xml` file is located in the directory specified by `directory_path`.  
CitySim simulations produce several result files, including three key indicators saved as layers in the original `.gpkg`:

- `All_AST`  
- `Tsat_@_Tmax`  
- `Pedestrian_@_Ta_max`

## Background
- Olivier Chavanne developed a framework to process and analyze building data, simulating surface temperatures using CitySim.  
- Zetong Liu added ground surfaces and new pavements.  
- This current project extends the framework by:
  - Implementing baseline scenarios for UHI quantification  
  - Adding terrain around buildings to create continuous urban surfaces  
  - Integrating trees and pedestrians into simulations  
  - Introducing new indicators for UHI quantification and pedestrian comfort

## Trees and pedestrian
The inclusion of trees and pedestrians follows the methodology from Coccolo (2017):  
https://infoscience.epfl.ch/entities/publication/7a844472-d333-4867-8364-48772278a7a5  
Points in the GeoPackage are converted to tree and pedestrian geometries in the simulation.  
Pedestrian geometries are created specifically to compute Mean Radiant Temperature (MRT).

## UHI indicators
Two new indicators were introduced to quantify UHIs, calculated for the warmest average annual day at maximum temperature:

- **Surface Temperature (Ts)**: Represents surface heating  
- **Sol-air Temperature (Tsol-air)**: Captures convective heat exchange and the effects of air temperature and solar radiation  
  https://en.wikipedia.org/wiki/Sol-air_temperature

## Pedestrian comfort indicators
The **Universal Thermal Climate Index (UTCI)** is used for its relevance across diverse climates.  
The number of **thermally comfortable hours per year** is computed using this index.

## References
- https://github.com/ochavanne/CAD-O  
- https://github.com/ZetongLiu/UHI_CH_sp  
- https://github.com/kaemco/CitySim-Solver/wiki/The-district#groundsurface  
- Silvia Coccolo, *Bioclimatic Design of Sustainable Campuses using Advanced Optimisation Methods*, 2017  
  DOI: https://doi.org/10.5075/epfl-thesis-7756  
  EPFL Infoscience: http://infoscience.epfl.ch/record/231147
