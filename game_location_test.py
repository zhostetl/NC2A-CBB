from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd 
import os 
import glob

raw_data_file = r'C:\Users\zhostetl\Documents\11_CBB\team_stats.xlsx'

df = pd.read_excel(raw_data_file)

sdf = df[df['Team']=='Baylor Bears']


def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="city_distance_calculator")
    location = geolocator.geocode(city_name, timeout=10)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

def calculate_distance(city1, city2):
    coordinates1 = get_coordinates(city1)
    coordinates2 = get_coordinates(city2)

    if coordinates1 and coordinates2:
        distance = geodesic(coordinates1, coordinates2).miles
        return distance
    else:
        return None


# baylor_loc = get_coordinates('Waco, Texas')
baylor_loc = 'Waco, Texas'

for idx, row in sdf.iterrows():
    game_state = row['State']
    game_coords = get_coordinates(game_state)
    

    # distance = geodesic(baylor_loc, game_coords).kilometers
    # print(f"distance: {distance}\n")
    
    distance_travelled = calculate_distance(baylor_loc, game_state)
    print(f"game location: {game_state}\n distance traveled: {distance_travelled} (miles)\n")