[Mesh]
  [input_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
    use_for_exodus_restart = true
  []
  [frac_inj]
    type = ExtraNodesetGenerator
    input = input_mesh
    coord = '-96.997662 -19.995474 -4140.000000'
    new_boundary = 'frac_inj'
  []
[]

[GlobalParams]
  # non linear variable declaration
  pore_pressure = pore_pressure
  temperature = temp
  displacements = 'disp_x disp_y disp_z'
  gravity_acceleration = 9.8065
  # common user objects
  supg_uo = supg
  scaling_uo = scaling
  porosity_uo = constant_phi
  permeability_uo = constant_k
  fluid_density_uo = constant_fluid_rho
  fluid_viscosity_uo = constant_eta
  fluid_thermal_conductivity_uo = lambda_f
  fluid_heat_capacity_uo = c_f
  fluid_thermal_expansion_uo = alpha_f
  solid_density_uo = constant_solid_rho
  # common fluid properties
  fluid_modulus = 2.5e09
  fluid_viscosity_initial = 3e-04
  fluid_density_initial = 1110
  # common solid properties
  young_modulus = 55.0e+09
  solid_thermal_expansion = 1.0e-06
[]

[Variables]
  [disp_x]
    initial_from_file_var = disp_x
    initial_from_file_timestep = 1
  []
  [disp_y]
    initial_from_file_var = disp_y
    initial_from_file_timestep = 1
  []
  [disp_z]
    initial_from_file_var = disp_z
    initial_from_file_timestep = 1
  []
  [pore_pressure]
    initial_from_file_var = pore_pressure
    initial_from_file_timestep = 2
  []
  [temp]
    initial_from_file_var = temp
    initial_from_file_timestep = 2
  []
[]

[Kernels]
  [momentum_x]
    type = GolemKernelM
    variable = disp_x
    component = 0
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
  []
  [momentum_y]
    type = GolemKernelM
    variable = disp_y
    component = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
  []
  [momentum_z]
    type = GolemKernelM
    variable = disp_z
    component = 2
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
  []
  [darcy]
    type = GolemKernelH
    variable = pore_pressure
  []
  [storageH]
    type = GolemKernelTimeH
    variable = pore_pressure
  []
  [fourier]
    type = GolemKernelT
    variable = temp
  []
  [storageT]
    type = GolemKernelTimeTSUPG
    variable = temp
  []
  [advective]
    type = GolemKernelTHSUPG
    variable = temp
  []
[]

[AuxVariables]
  [stress_xx]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = stress_xx
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [stress_yy]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = stress_yy
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [stress_zz]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = stress_zz
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [stress_xy]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = stress_xy
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [stress_xz]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = stress_xz
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [stress_yz]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = stress_yz
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [sigma_n]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = sigma_n
    initial_from_file_timestep = 1
    block = 'seismic_events'
  []
  [tau]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = tau
    initial_from_file_timestep = 1
    block = 'seismic_events'
  []
  [biot]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = biot
    initial_from_file_timestep = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [ST]
    order = CONSTANT
    family = MONOMIAL
    initial_from_file_var = ST
    initial_from_file_timestep = 1
    block = 'seismic_events'
  []
  [thermal_stress_xx]
    order = CONSTANT
    family = MONOMIAL
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
  []
  [thermal_stress_yy]
    order = CONSTANT
    family = MONOMIAL
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
  []
  [thermal_stress_zz]
    order = CONSTANT
    family = MONOMIAL
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
  []
  # CFF AuxVariables
  [tau_fault]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [delta_tau]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [cff]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [fault_length]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
    [InitialCondition]
      type = RandomIC
      max = 100
      min = 10
      seed = 100
    []
  []
  [friction_static]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
    [InitialCondition]
      type = ConstantIC
      value = 0.85
    []
  []
  [friction_dynamic]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
    [InitialCondition]
      type = ConstantIC
      value = 0.65
    []
  []
  [r_factor]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
    [InitialCondition]
      type = RandomIC
      min = 0.1
      max = 0.5
      seed = 100
    []
  []
  [cumulative_seismic_moment]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [seismic_moment]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [cumulative_seismic_event]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [seismic_event]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [cumulative_fault_slip]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [fault_slip]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [cumulative_interevent_time]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [interevent_time]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
  [source_radius]
    order = CONSTANT
    family = MONOMIAL
    block = 'seismic_events'
  []
[]

[AuxKernels]
  [stress_xx_aux]
    type = GolemHD2LDRankTwoTensorAux
    variable = stress_xx
    material_user_object = stress_xx_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_yy_aux]
    type = GolemHD2LDRankTwoTensorAux
    variable = stress_yy
    material_user_object = stress_yy_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_zz_aux]
    type = GolemHD2LDRankTwoTensorAux
    variable = stress_zz
    material_user_object = stress_yy_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_xy_aux]
    type = GolemHD2LDRankTwoTensorAux
    variable = stress_xy
    material_user_object = stress_xy_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_xz_aux]
    type = GolemHD2LDRankTwoTensorAux
    variable = stress_xz
    material_user_object = stress_xz_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_yz_aux]
    type = GolemHD2LDRankTwoTensorAux
    variable = stress_yz
    material_user_object = stress_yz_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [sigma_n_aux]
    type = GolemMappedStressAux
    variable = sigma_n
    mapped_stress = 'stress_xx stress_yy stress_zz stress_xy stress_xz stress_yz'
    stress_type = normal
    block = 'seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [tau_aux]
    type = GolemMappedStressAux
    variable = tau
    mapped_stress = 'stress_xx stress_yy stress_zz stress_xy stress_xz stress_yz'
    stress_type = shear
    block = 'seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [biot_aux]
    type = GolemHD2LDRealAux
    variable = biot
    material_user_object = biot_uo
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [ST_aux]
    type = GolemSTAux
    variable = ST
    normal_stress = sigma_n
    shear_stress = tau
    biot_var = biot
    block = 'seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [thermal_stress_xx]
    type = GolemThermalStress
    variable = thermal_stress_xx
    index = 0
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
    execute_on = 'TIMESTEP_END'
  []
  [thermal_stress_yy]
    type = GolemThermalStress
    variable = thermal_stress_yy
    index = 1
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
    execute_on = 'TIMESTEP_END'
  []
  [thermal_stress_zz]
    type = GolemThermalStress
    variable = thermal_stress_zz
    index = 2
    block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2'
    execute_on = 'TIMESTEP_END'
  []
  # CFF AuxKernels
  [tau_fault_aux]
    type = MaterialRealAux
    variable = tau_fault
    property = tau_fault
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [delta_tau_aux]
    type = MaterialRealAux
    variable = delta_tau
    property = delta_tau
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cff_aux]
    type = MaterialRealAux
    variable = cff
    property = cff
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cumulative_seismic_moment_aux]
    type = MaterialRealAux
    variable = cumulative_seismic_moment
    property = cumulative_seismic_moment
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [seismic_moment_aux]
    type = GolemIncrementalVariableAux
    variable = seismic_moment
    var_incr = cumulative_seismic_moment
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cumulative_seismic_event_aux]
    type = MaterialRealAux
    variable = cumulative_seismic_event
    property = cumulative_seismic_event
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [seismic_event_aux]
    type = GolemIncrementalVariableAux
    variable = seismic_event
    var_incr = cumulative_seismic_event
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cumulative_fault_slip_aux]
    type = MaterialRealAux
    variable = cumulative_fault_slip
    property = cumulative_fault_slip
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [fault_slip_aux]
    type = GolemIncrementalVariableAux
    variable = fault_slip
    var_incr = cumulative_fault_slip
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cumulative_interevent_time_aux]
    type = MaterialRealAux
    variable = cumulative_interevent_time
    property = 'recurrence_time'
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [interevent_time_aux]
    type = GolemIncrementalVariableAux
    variable = interevent_time
    var_incr = cumulative_interevent_time
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [source_radius_aux]
    type = MaterialRealAux
    variable = source_radius
    property = 'source_size'
    block = 'seismic_events'
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Functions]
  [pressure_gradient]
    type = ParsedFunction
    expression = 'rho_f*g*(z-c)+p0'
    symbol_names = 'rho_f g c p0'
    symbol_values = '1.11e-3 -9.8065 -135 0.1'
  []
  [temp_gradient]
    type = ParsedFunction
    expression = 'grad*(z-c)+T0'
    symbol_names = 'grad c T0'
    symbol_values = '-0.028 -3816 137.5'
  []
  [lithostatic_load]
    type = ParsedFunction
    expression = 'a*b*(z-c)'
    symbol_names = 'a b c'
    symbol_values = '2.5e-03 9.8065 65'
  []
  [top_shear_x]
    type = ParsedFunction
    expression = '5.28'
  []
  [top_shear_y]
    type = ParsedFunction
    expression = '-9.17'
  []
  [injection_flowrate]
    type = GolemFunctionReadFile
    file = 'files/injection_flowrate.csv'
  []
  [frac_aperture]
    type = GolemFunctionReadFile
    file = 'files/frac_aperture.csv'
  []
[]

[DiracKernels]
  [HInjection]
    type = GolemDiracKernelTH
    variable = pore_pressure
    source_point = '-96.997662 -19.995474 -4140.000000'
    source_type = injection
    function_rate = injection_flowrate
    start_time = 0.0
    end_time = 279000
  []
[]

[BCs]
  [temp_top_bc]
    type = FunctionDirichletBC
    function = temp_gradient
    boundary = 'top'
    variable = temp
  []
  [temp_bottom_bc]
    type = GolemHeatFlowBC
    value = 0.072
    boundary = 'bottom'
    variable = temp
  []
  [temp_injection]
    type = DirichletBC
    variable = temp
    boundary = frac_inj
    value = 30
  []
  [pressure_bc]
    type = FunctionDirichletBC
    function = pressure_gradient
    boundary = 'west east'
    variable = pore_pressure
  []
  [load_x_top]
    type = FunctionNeumannBC
    function = top_shear_x
    boundary = 'top'
    variable = disp_x
  []
  [load_y_top]
    type = FunctionNeumannBC
    function = top_shear_y
    boundary = 'top'
    variable = disp_y
  []
  [load_z_top]
    type = FunctionNeumannBC
    function = lithostatic_load
    boundary = 'top'
    variable = disp_z
  []
  [disp_z_bottom]
    type = DirichletBC
    variable = disp_z
    boundary = 'bottom'
    value = 0.0
  []
  [disp_x_east]
    type = DirichletBC
    variable = disp_x
    boundary = 'east'
    value = 0.10
  []
  [disp_x_west]
    type = DirichletBC
    variable = disp_x
    boundary = 'west'
    value = -0.10
  []
  [disp_y_north]
    type = DirichletBC
    variable = disp_y
    boundary = 'north'
    value = -0.64
  []
  [disp_y_south]
    type = DirichletBC
    variable = disp_y
    boundary = 'south'
    value = 0.64
  []
[]

[Materials]
  [unit_1]
    type = GolemMaterialTHMElasticSUPG
    block = 'hannover1 hannover2'
    # fluid materials
    porosity_initial = 0.01
    permeability_type = 'orthotropic'
    permeability_initial = '4.9346e-17 4.9346e-17 1.2336e-17'
    # thermal materials
    solid_heat_capacity_uo = cp_1
    solid_thermal_conductivity_uo = lambda_1
    # mechanical materials
    poisson_ratio = 0.18
  []
  [unit_2]
    type = GolemMaterialTHMElasticSUPG
    block = 'elbe_alternative'
    # fluid materials
    porosity_initial = 0.03
    permeability_type = 'orthotropic'
    permeability_initial = '3.2e-16 3.2e-16 8.0e-17'
    # thermal materials
    solid_heat_capacity_uo = cp_2
    solid_thermal_conductivity_uo = lambda_2
    # mechanical materials
    poisson_ratio = 0.18
  []
  [unit_3]
    type = GolemMaterialTHMElasticSUPG
    block = 'elbe_ssII'
    # fluid materials
    porosity_initial = 0.08
    permeability_type = 'orthotropic'
    permeability_initial = '6.4e-16 6.4e-16 1.6e-16'
    # thermal materials
    solid_heat_capacity_uo = cp_3
    solid_thermal_conductivity_uo = lambda_3
    # mechanical materials
    poisson_ratio = 0.18
  []
  [unit_4]
    type = GolemMaterialTHMElasticSUPG
    block = 'elbe_ssI'
    # fluid materials
    porosity_initial = 0.15
    permeability_type = 'orthotropic'
    permeability_initial = '1.28e-15 1.28e-15 3.2e-16'
    # thermal materials
    solid_heat_capacity_uo = cp_4
    solid_thermal_conductivity_uo = lambda_4
    # mechanical materials
    poisson_ratio = 0.18
  []
  [unit_5]
    type = GolemMaterialTHMElasticSUPG
    block = 'havel'
    # fluid materials
    porosity_initial = 0.001
    permeability_type = 'orthotropic'
    permeability_initial = '9.87e-17 9.87e-17 2.4675e-17'
    # thermal materials
    solid_heat_capacity_uo = cp_5
    solid_thermal_conductivity_uo = lambda_5
    # mechanical materials
    poisson_ratio = 0.18
  []
  [unit_6]
    type = GolemMaterialTHMElasticSUPG
    block = 'volc1 volc2'
    # fluid materials
    porosity_initial = 0.005
    permeability_type = 'orthotropic'
    permeability_initial = '9.87e-17 9.87e-17 2.4675e-17'
    # thermal materials
    solid_heat_capacity_uo = cp_6
    solid_thermal_conductivity_uo = lambda_6
    # mechanical materials
    poisson_ratio = 0.20
  []
  [fracs_multi]
    type = GolemMaterialTHMFracElasticSUPG
    block = 'frac_multi'
    material_type = frac
    fracture_aperture_uo = constant_aperture
    porosity_initial = 1.0
    permeability_type = 'isotropic'
    permeability_initial = 1e-10
    poisson_ratio = 0.28
  []
  [fracs_water]
    type = GolemMaterialTHMFracElasticSUPG
    block = 'frac_water'
    material_type = frac
    fracture_aperture_uo = function_aperture
    porosity_initial = 1.0
    permeability_type = 'isotropic'
    permeability_uo = permeability_cubic_law
    permeability_initial = 1.0
    poisson_ratio = 0.28
  []
  [seismic_plane]
    type = GolemMaterialTHMFracElasticCFFSUPG
    block = 'seismic_events'
    material_type = 'frac'
    fracture_aperture_uo = constant_aperture
    porosity_initial = 1.0
    permeability_type = 'isotropic'
    permeability_initial = 1e-13
    poisson_ratio = 0.28
    # CFF properties
    normal_stress = sigma_n
    shear_stress = tau
    friction_static = friction_static
    friction_dynamic = friction_dynamic
    r_factor = r_factor
    #f_length = fault_length
    cff_limit = 0.0
    patch_type = circular
    source_size_type = variable
    source_radius = fault_length
    # slip computation
    #fault_stiffness = 300e9
  []
  [well]
    type = GolemMaterialTHMElasticSUPG
    block = 'well_GtGrSk405'
    material_type = 'well'
    scaling_factor_initial = 0.05
    porosity_initial = 1
    permeability_type = 'isotropic'
    permeability_initial = 3.125e-04
    poisson_ratio = 0.28
  []
[]

[UserObjects]
  [supg]
    type = GolemSUPG
    execute_on = 'INITIAL'
  []
  [scaling]
    type = GolemScaling
    characteristic_stress = 1e6
    execute_on = 'INITIAL'
  []
  [constant_phi]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  []
  [constant_k]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  []
  [constant_eta]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  []
  [constant_fluid_rho]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  []
  [permeability_cubic_law]
    type = GolemPermeabilityCubicLaw
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [lambda_f]
    type = GolemFluidThermalConductivityConstant
    value = 0.65
    execute_on = 'INITIAL'
  []
  [c_f]
    type = GolemFluidHeatCapacityConstant
    value = 4200
    execute_on = 'INITIAL'
  []
  [alpha_f]
    type = GolemFluidThermalExpansionConstant
    value = 9.1e-05
    execute_on = 'INITIAL'
  []
  [lambda_1]
    type = GolemSolidThermalConductivityConstant
    value = 1.94
    execute_on = 'INITIAL'
  []
  [cp_1]
    type = GolemSolidHeatCapacityConstant
    value = 920
    execute_on = 'INITIAL'
  []
  [lambda_2]
    type = GolemSolidThermalConductivityConstant
    value = 1.91
    execute_on = 'INITIAL'
  []
  [cp_2]
    type = GolemSolidHeatCapacityConstant
    value = 920
    execute_on = 'INITIAL'
  []
  [lambda_3]
    type = GolemSolidThermalConductivityConstant
    value = 3.1
    execute_on = 'INITIAL'
  []
  [cp_3]
    type = GolemSolidHeatCapacityConstant
    value = 920
    execute_on = 'INITIAL'
  []
  [lambda_4]
    type = GolemSolidThermalConductivityConstant
    value = 3.18
    execute_on = 'INITIAL'
  []
  [cp_4]
    type = GolemSolidHeatCapacityConstant
    value = 920
    execute_on = 'INITIAL'
  []
  [lambda_5]
    type = GolemSolidThermalConductivityConstant
    value = 3.0
    execute_on = 'INITIAL'
  []
  [cp_5]
    type = GolemSolidHeatCapacityConstant
    value = 1000
    execute_on = 'INITIAL'
  []
  [lambda_6]
    type = GolemSolidThermalConductivityConstant
    value = 2.31
    execute_on = 'INITIAL'
  []
  [cp_6]
    type = GolemSolidHeatCapacityConstant
    value = 1380
    execute_on = 'INITIAL'
  []
  [constant_solid_rho]
    type = GolemSolidDensityConstant
    value = 2650.0
    execute_on = 'INITIAL'
  []
  [constant_aperture]
    type = GolemFractureApertureConstant
    value = 1e-02
    execute_on = 'INITIAL'
  []
  [function_aperture]
    type = GolemFractureApertureFunction
    function = frac_aperture
    execute_on = 'INITIAL TIMESTEP_END'
  []
  # mapping stress UserObjects
  [stress_xx_uo]
    type = GolemHD2LDRankTwoTensor
    mat_prop = 'stress'
    index_i = 0
    index_j = 0
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_yy_uo]
    type = GolemHD2LDRankTwoTensor
    mat_prop = 'stress'
    index_i = 1
    index_j = 1
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_zz_uo]
    type = GolemHD2LDRankTwoTensor
    mat_prop = 'stress'
    index_i = 2
    index_j = 2
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_xy_uo]
    type = GolemHD2LDRankTwoTensor
    mat_prop = 'stress'
    index_i = 0
    index_j = 1
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_xz_uo]
    type = GolemHD2LDRankTwoTensor
    mat_prop = 'stress'
    index_i = 0
    index_j = 2
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  [stress_yz_uo]
    type = GolemHD2LDRankTwoTensor
    mat_prop = 'stress'
    index_i = 1
    index_j = 2
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
  # ST UserObjects
  [biot_uo]
    type = GolemHD2LDReal
    mat_prop = 'biot_coefficient'
    file_name = 'map_uo.csv'
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
    execute_on = 'TIMESTEP_END'
  []
[]

[VectorPostprocessors]
  [sampling]
    type = GolemElementValueSampler
    variable = 'sigma_n tau ST cff delta_tau tau_fault fault_length cumulative_seismic_event seismic_event r_factor cumulative_fault_slip fault_slip'
    block = 'seismic_events'
    outputs = 'csv_vp'
    sort_by = id
    out_type = elem_centroid
    centroid_type = vertex_average
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Postprocessors]
  [cum_mo]
    type = ElementAverageValue
    variable = cumulative_seismic_moment
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ave_cum_mo]
    type = CumulativeValuePostprocessor
    postprocessor = cum_mo
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_cum_mo]
    type = ChangeOverTimePostprocessor
    postprocessor = ave_cum_mo
    compute_relative_change = true
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [mo]
    type = ElementAverageValue
    variable = seismic_moment
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ave_mo]
    type = CumulativeValuePostprocessor
    postprocessor = mo
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_mo]
    type = ChangeOverTimePostprocessor
    postprocessor = ave_mo
    compute_relative_change = true
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cum_ev]
    type = ElementAverageValue
    variable = cumulative_seismic_event
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ave_cum_ev]
    type = CumulativeValuePostprocessor
    postprocessor = cum_ev
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_cum_ev]
    type = ChangeOverTimePostprocessor
    postprocessor = ave_cum_ev
    compute_relative_change = true
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [sum_ev]
    type = GolemElementVariableSum
    variable = 'seismic_event'
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_sum_ev]
    type = ChangeOverTimePostprocessor
    postprocessor = 'sum_ev'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [sum_cum_ev]
    type = GolemElementVariableSum
    variable = 'cumulative_seismic_event'
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_sum_cum_ev]
    type = ChangeOverTimePostprocessor
    postprocessor = 'sum_cum_ev'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ev]
    type = ElementAverageValue
    variable = seismic_event
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ave_ev]
    type = CumulativeValuePostprocessor
    postprocessor = ev
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_ev]
    type = ChangeOverTimePostprocessor
    postprocessor = ave_ev
    compute_relative_change = true
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [cum_slip]
    type = ElementAverageValue
    variable = cumulative_fault_slip
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ave_cum_slip]
    type = CumulativeValuePostprocessor
    postprocessor = cum_slip
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_cum_slip]
    type = ChangeOverTimePostprocessor
    postprocessor = ave_cum_slip
    compute_relative_change = true
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [slip]
    type = ElementAverageValue
    variable = fault_slip
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [ave_slip]
    type = CumulativeValuePostprocessor
    postprocessor = slip
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [change_slip]
    type = ChangeOverTimePostprocessor
    postprocessor = ave_slip
    compute_relative_change = true
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [flow_rate]
    type = FunctionValuePostprocessor
    function = injection_flowrate
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [source_radius]
    type = ElementAverageValue
    variable = source_radius
    block = 'seismic_events'
    outputs = 'csv_av'
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Preconditioning]
  active = 'fieldsplit'
  [fieldsplit]
    type = FSP
    topsplit = 'HTM'
    [HTM]
      splitting = 'H T M'
      splitting_type = multiplicative
      petsc_options = '-snes_ksp_ew'
      petsc_options_iname = '-ksp_type
                             -ksp_rtol -ksp_max_it
                             -snes_type -snes_linesearch_type
                             -snes_atol -snes_stol -snes_max_it'
      petsc_options_value = 'fgmres
                             1.0e-12 50
                             newtonls basic
                             1.0e-4 0 100'
    []
    [H]
      vars = 'pore_pressure'
      petsc_options_iname = '-ksp_type
                             -pc_type -pc_hypre_type'
      petsc_options_value = 'preonly
                             hypre boomeramg'
    []
    [T]
      vars = 'temp'
      petsc_options_iname = '-ksp_type
                             -pc_type
                             -sub_pc_type -sub_pc_factor_levels
                             -ksp_rtol -ksp_max_it'
      petsc_options_value = 'fgmres
                             asm
                             ilu 1
                             1e-04 500'
    []
    [M]
      vars = 'disp_x disp_y disp_z'
      petsc_options_iname = '-ksp_type
                             -pc_type
                             -sub_pc_type -sub_pc_factor_levels
                             -ksp_rtol -ksp_max_it'
      petsc_options_value = 'fgmres
                             asm
                             ilu 1
                             1e-04 500'
    []
  []
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  start_time = 0.0
  end_time = 279000
  dt = 1800
  compute_scaling_once = false
  automatic_scaling = true
[]

[Outputs]
  print_linear_residuals = false
  perf_graph = true
  [out]
    type = Exodus
    file_base = 'transient_result'

    sync_times = '0 1800 3600 5400 7200 9000 10800 12600 14400
    16200 18000 19800 21600 23400 25200 27000 28800 30600 32400
    34200 36000 37800 39600 41400 43200 45000 46800 48600 50400
    52200 54000 55800 57600 59400 61200 63000 64800 66600 68400
    70200 72000 73800 75600 77400 79200 81000 82800 84600 86400
    88200 90000 91800 93600 95400 97200 99000 100800 102600 104400
    106200 108000 109800 111600 113400 115200 117000 118800 120600
    122400 124200 126000 127800 129600 131400 133200 135000 136800
    138600 140400 142200 144000 145800 147600 149400 151200 153000
    154800 156600 158400 160200 162000 163800 165600 167400 169200
    171000 172800 174600 176400 178200 180000 181800 183600 185400
    187200 189000 190800 192600 194400 196200 198000 199800 201600
    203400 205200 207000 208800 210600 212400 214200 216000 217800
    219600 221400 223200 225000 226800 228600 230400 232200 234000
    235800 237600 239400 241200 243000 244800 246600 248400 250200
    252000 253800 255600 257400 259200 261000 262800 264600 266400
    268200 270000 271800 273600 275400 277200 279000'
    sync_only = true
  []
  [csv_vp]
    type = CSV
    execute_on = 'INITIAL TIMESTEP_END'
    sync_times = '0 1800 3600 43200 86400 129600 172800 216000 259200 279000'
    sync_only = true
  []
  [csv_av]
    type = CSV
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]
