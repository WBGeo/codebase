import liquid_earth_sdk as le

def push_geosolution_to_le(geosolution, space_name, model_name, api_token):
    """
    Push a geosolution to Liquid Earth.

    Parameters:
    - geosolution: The geosolution object containing the results.
    - space_name: The name of the Liquid Earth space to upload to.
    - model_name: The name of the model in Liquid Earth.
    - api_token: The API token for authentication.

    Returns:
    - link: The link to the uploaded mesh in Liquid Earth.
    """
    meshes = geosolution.meshes_to_subsurface() # Convert geosolution to subsurface mesh format.requires a gempy solution
    link = le.upload_mesh_to_new_space(space_name, meshes, model_name, api_token)
    return link