from core.object_components import InputData
import numpy as np
import pandas as pd


def geo_input_data_fix(name: str, extent_str: str, resolution_str: str,
                       surface_points_file: str,
                       orientations_file: str,
                       mapping_file: str,
                       with_faults: bool
                       ) -> InputData:
    # TODO: Provide a proper input type which does not require strings
    import os
    import pathlib

    datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
    print(datadir)
    print(os.path.join(datadir, 'examples/data/', orientations_file))
    print(os.path.join(datadir, 'examples/data/'))

    extent = np.array([int(i.strip()) for i in extent_str.split(",")])
    resolution = np.array([int(i.strip()) for i in resolution_str.split(",")])
    surface_points = pd.read_csv(os.path.join(datadir, 'examples/data/', surface_points_file))
    orientations = None
    if orientations_file != 'None':
        orientations = pd.read_csv(os.path.join(datadir, 'examples/data/', orientations_file))

    mapping_object = {}
    if mapping_file != 'None':
        with open (os.path.join(datadir, 'examples/data/', mapping_file), 'r') as f:
            import json
            mapping_object = json.load(f)

    return InputData(
        name=name,
        extent=extent,
        resolution=resolution,
        surface_points=surface_points,
        orientations=orientations,
        mapping_object=mapping_object['mapping'] if 'mapping' in mapping_object else {},
        faults=mapping_object['faults'] if with_faults and 'faults' in mapping_object else None,
        # fault_relations=np.array(
        #     [[0, 1, 1],
        #      [0, 0, 0],
        #      [0, 0, 0]]
        # ),
        fault_relations=None, # TODO: fault relations
    )


