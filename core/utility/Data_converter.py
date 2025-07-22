# Importing necessary libraries
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
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

    orientations_df = pd.read_csv(cwd + "/examples/data/modelWeisweilerMini_orientations.csv")
    surface_points_df = pd.read_csv(cwd + "/examples/data/modelWeisweilerMini_surface_points.csv")

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

    orientations_df.to_csv(cwd + "/examples/data/modelWeisweilerMini_orientations_df.csv", index=False)

    surface_points_df.to_csv(cwd + "/examples/data/modelWeisweilerMini_surface_points_df.csv", index=False)

    #%%

    # surface_points_df = pd.read_csv(cwd + "/examples/data/modelWeisweilerMini_surface_points_df.csv")
    orientations_df = pd.read_csv(cwd + "/examples/data/modelWeisweilerMini_orientations_df.csv")
    orientations_df

    #%%

    # surface_points_df[['X', 'Y']] = surface_points_df[['Y', 'X']]
    orientations_df[['X', 'Y']] = orientations_df[['Y', 'X']]
    orientations_df[['G_y', 'G_x']] = orientations_df[['G_x', 'G_y']]

    #%%

    # surface_points_df.to_csv(cwd + "/examples/data/modelWeisweilerMini_surface_points_df2.csv", index=False)
    orientations_df.to_csv(cwd + "/examples/data/modelWeisweilerMini_orientations_df.csv", index=False)

    #%%

    # Load the DataFrame from a CSV file
    orientations_df = pd.read_csv(cwd + "/examples/data/modelWeisweilerMini_orientations_df.csv")

    # Define the new order of columns
    new_column_order = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'formation']  # Replace with your column names

    # Reorder the DataFrame columns
    orientations_df = orientations_df[new_column_order]
    orientations_df

    #%%

    # Save the reordered DataFrame to a new CSV file
    orientations_df.to_csv(cwd + "/examples/data/modelWeisweilerMini_orientations_df.csv", index=False)


