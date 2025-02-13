
from core.object_components import GeomodelResults
import os

cwd = os.getcwd()


def modify_input_file(file_name, parameters, additional_tag="", extension=".i"):
    """
    Simple function to modify an input template file for MOOSE.

    Args:
        file_name (str): Path of template file.
        parameters (list): List of parameters written to the file.
        additional_tag (str): Additional tag for the output file name.
        extension (str): File type extension.
    """

    # Open the file with read only permit
    inputfile = open(file_name + ".template", "r")
    lines = inputfile.read()

    # Replace the target strings
    for i in range(0, len(parameters)):
        place_holder = "$" + str(i) + "$"
        lines = str.replace(lines, place_holder, parameters[i])

    # Close the file
    auto_inputfile = open(file_name + additional_tag + extension, "w")
    auto_inputfile.write(lines)


def export_data_to_moose(geomodel_results: GeomodelResults):
    """
     Get parameters from geological model and write them to a MOOSE input file.

     Args:
         geomodel_results (GeomodelResults): The results of the geological model.
     """

    file_name = cwd + '/core/meshing_components/moose_mesh_input'
    ext = '_'
    scale = 1000

    ids = geomodel_results.lith_block
    ids = ids.astype(int)
    nx, ny, nz = geomodel_results.resolution

    # model extent
    xmin, xmax, ymin, ymax, zmin, zmax = geomodel_results.extent

    units = ids.reshape((nx, ny, nz))
    # flatten MOOSE conform
    units = units.flatten('F')

    # create unit ID string for the fstring
    unit_string = '\n  '.join(map(str, units))

    # surfs = [item for sublist in geomodel_results.mapping_object.values() for item in sublist]

    surfs = [item if isinstance(item, tuple) else (item,) for item in geomodel_results.mapping_object.values()]
    surfs = [val for sublist in surfs for val in sublist]

    uids = list(range(1, len(surfs) + 1))

    # Very simple check for basement
    # TODO: Improve consistency
    if "basement" not in surfs:
        surfs.append("basement")
        uids.append(len(surfs))

    surfs_string = ' '.join(surfs)
    ids_string = ' '.join(map(str, uids))
    surfs_string = "'" + surfs_string + "'"
    ids_string = "'" + ids_string + "'"

    modify_input_file(file_name
                      , [str(nx), str(nz), str(ny),
                         str(xmin / scale), str(xmax / scale),
                         str(zmin / scale), str(zmax / scale),
                         str(ymin / scale), str(ymax / scale),
                         ids_string, surfs_string, unit_string],
                      ext + str(geomodel_results.name))
