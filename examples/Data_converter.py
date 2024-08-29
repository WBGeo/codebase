# Importing necessary libraries
import numpy as np
import pandas as pd
import os


#%%

# Some data imports specifically for this example

# # Define the path to data
# data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
# path_to_data = data_path + "/data/input_data/jan_models/"
#
# # Load the data as pandas df
# orientations_df = pd.read_csv(path_to_data + "model2_orientations.csv")
# surface_points_df = pd.read_csv(path_to_data + "model2_surface_points.csv")

cwd = os.getcwd()

# orientations_df = pd.read_csv(path_to_data + "model2_orientations.csv")
# surface_points_df = pd.read_csv(path_to_data + "model2_surface_points.csv")

orientations_df = pd.read_csv(cwd + "/examples/data/tutorial_model_orientations.csv")
surface_points_df = pd.read_csv(cwd + "/examples/data/tutorial_model_surface_points.csv")

# transform orientations to pole vector format (seems more intuitive)
azimuth_rad = np.radians(orientations_df.azimuth.to_numpy())
dip_rad = np.radians(orientations_df.dip.to_numpy())
polarity = orientations_df.polarity.to_numpy()

orientations_df['azimuth'] = np.sin(dip_rad) * np.sin(azimuth_rad) * polarity
orientations_df['dip'] = np.sin(dip_rad) * np.cos(azimuth_rad) * polarity
orientations_df['polarity'] = np.cos(dip_rad) * polarity

orientations_df.rename(columns={'X': 'X', 'Y': 'Y', 'Z': 'Z',
                                'azimuth': 'G_x', 'dip': 'G_y', 'polarity': 'G_z', 'formation': 'formation'},
                       inplace=True)

orientations_df.to_csv(cwd + "/examples/data/model12_orientations_df.csv", index=False)

surface_points_df.to_csv(cwd + "/examples/data/model12_surface_points_df.csv", index=False)
