"""Download real public NASA Earth Observatory demo scenes.
Run from the project root: py download_sample_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

OUT=Path(__file__).parent/'sample_data'
OUT.mkdir(exist_ok=True)
FILES={
 'flood_before.jpg':'https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/147000/147204/lousiana_oli_2020226_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=1200&w=1800',
 'flood_after.jpg':'https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/147000/147204/lousiana_oli_2020242_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=1200&w=1800',
 'deforestation_1986.jpg':'https://eoimages.gsfc.nasa.gov/images/imagerecords/150000/150257/boliviaradial_tm_1986183_lrg.jpg',
 'deforestation_2022.jpg':'https://eoimages.gsfc.nasa.gov/images/imagerecords/150000/150257/boliviaradial_oli_2022234_lrg.jpg',
 'urban_jakarta_2004.jpg':'https://eoimages.gsfc.nasa.gov/images/imagerecords/5000/5693/jakarta_ast_2004_lrg.jpg',
 'agriculture_cerrado.jpg':'https://eoimages.gsfc.nasa.gov/images/imagerecords/85000/85364/cerradosoy_tmo_2001_2013_lrg.jpg',
}
for name,url in FILES.items():
    target=OUT/name
    if target.exists():
        print('exists:',name); continue
    print('downloading:',name)
    urlretrieve(url,target)
print('Done. Real NASA satellite scenes are in sample_data/.')
