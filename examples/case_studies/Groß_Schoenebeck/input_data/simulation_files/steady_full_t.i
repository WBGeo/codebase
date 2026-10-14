[Mesh]
  [original_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
  [rename_domain]
    type = RenameBlockGenerator
    input = original_mesh
    old_block = '8 7 6 5 4 3 2 1
     9 10 11 12'
    new_block = 'hannover1 hannover2 elbe_alternative elbe_ssII elbe_ssI havel volc1 volc2
    frac_water frac_multi  seismic_events
    well_GtGrSk405'
  []
  [rename_boundary]
    type = RenameBoundaryGenerator
    input = rename_domain
    old_boundary = 'top bottom right back front left'
    new_boundary = 'top bottom east south north west'
  []
[]

[GlobalParams]
  # non linear variable declaration
  pore_pressure = 'pore_pressure'
  temperature = 'temp'
  displacements = 'disp_x disp_y disp_z'
  gravity_acceleration = 9.8065
  # common user objects
  scaling_uo = scaling
  porosity_uo = constant_phi
  permeability_uo = constant_k
  fluid_density_uo = constant_fluid_rho
  fluid_viscosity_uo = constant_eta
  fluid_thermal_conductivity_uo = lambda_f
  fluid_heat_capacity_uo = c_f
  solid_density_uo = constant_solid_rho
  # common fluid properties
  fluid_modulus = 2.5e09
  fluid_viscosity_initial = 3e-04
  fluid_density_initial = 1110
  # common solid properties
  young_modulus = 55.0e+09
[]

[Functions]
  [pressure_gradient]
    type = ParsedFunction
    expression = '-rho_f*g*(z-c)+p0'
    symbol_names = 'rho_f g c p0'
    symbol_values = '1.11e-3 9.865 -135 0.1'
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
[]

[Variables]
  [pore_pressure]
    [InitialCondition]
      type = FunctionIC
      function = pressure_gradient
    []
  []
  [temp]
    [InitialCondition]
      type = FunctionIC
      function = temp_gradient
    []
  []
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
[]

[Problem]
  type = FEProblem
  kernel_coverage_check = false
[]

[Kernels]
  [darcy]
    type = GolemKernelH
    variable = pore_pressure
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [fourier]
    type = GolemKernelT
    variable = temp
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
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
  [storageH]
    type = GolemKernelTimeH
    variable = pore_pressure
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [storageT]
    type = GolemKernelTimeT
    variable = temp
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
  [advective]
    type = GolemKernelTH
    variable = temp
    block = 'hannover1 hannover2 elbe_alternative
    elbe_ssII elbe_ssI havel volc1 volc2
    frac_multi frac_water seismic_events'
  []
[]

[GolemMapStressAction]
  write_map = false
  file_name = 'map.csv'
  action_blocks = 'hannover1 hannover2 elbe_alternative
  elbe_ssII elbe_ssI havel volc1 volc2
  frac_multi frac_water seismic_events'
  frac_blocks = 'frac_multi frac_water seismic_events'
  index_i = '0 1 2 0 0 1'
  index_j = '0 1 2 1 2 2'
  epsilon = 1e-4
  patch_size = 10
  compute_ST = true
[]

[Materials]
  [unit_1]
    type = GolemMaterialTHMElastic
    block = 'hannover1 hannover2'
    # fluid properties
    porosity_initial = 0.01
    permeability_type = 'orthotropic'
    permeability_initial = '4.9346e-17 4.9346e-17 1.2336e-17'
    # thermal properties
    solid_heat_capacity_uo = cp_1
    solid_thermal_conductivity_uo = lambda_1
    # mechanical properties
    poisson_ratio = 0.18
  []
  [unit_2]
    type = GolemMaterialTHMElastic
    block = 'elbe_alternative'
    # fluid properties
    porosity_initial = 0.03
    permeability_type = 'orthotropic'
    permeability_initial = '3.2e-16 3.2e-16 8.0e-17'
    # thermal properties
    solid_heat_capacity_uo = cp_2
    solid_thermal_conductivity_uo = lambda_2
    # mechanical properties
    poisson_ratio = 0.18
  []
  [unit_3]
    type = GolemMaterialTHMElastic
    block = 'elbe_ssII'
    # fluid properties
    porosity_initial = 0.08
    permeability_type = 'orthotropic'
    permeability_initial = '6.4e-16 6.4e-16 1.6e-16'
    # thermal properties
    solid_heat_capacity_uo = cp_3
    solid_thermal_conductivity_uo = lambda_3
    # mechanical properties
    poisson_ratio = 0.18
  []
  [unit_4]
    type = GolemMaterialTHMElastic
    block = 'elbe_ssI'
    # fluid properties
    porosity_initial = 0.15
    permeability_type = 'orthotropic'
    permeability_initial = '1.28e-15 1.28e-15 3.2e-16'
    # thermal properties
    solid_heat_capacity_uo = cp_4
    solid_thermal_conductivity_uo = lambda_4
    # mechanical properties
    poisson_ratio = 0.18
  []
  [unit_5]
    type = GolemMaterialTHMElastic
    block = 'havel'
    # fluid properties
    porosity_initial = 0.001
    permeability_type = 'orthotropic'
    permeability_initial = '9.87e-17 9.87e-17 2.4675e-17'
    # thermal properties
    solid_heat_capacity_uo = cp_5
    solid_thermal_conductivity_uo = lambda_5
    # mechanical properties
    poisson_ratio = 0.18
  []
  [unit_6]
    type = GolemMaterialTHMElastic
    block = 'volc1 volc2'
    # fluid properties
    porosity_initial = 0.005
    permeability_type = 'orthotropic'
    permeability_initial = '9.87e-17 9.87e-17 2.4675e-17'
    # thermal properties
    solid_heat_capacity_uo = cp_6
    solid_thermal_conductivity_uo = lambda_6
    # mechanical properties
    poisson_ratio = 0.20
  []
  [fracs]
    type = GolemMaterialTHMFracElastic
    block = 'frac_multi frac_water'
    material_type = frac
    fracture_aperture_uo = constant_aperture
    porosity_initial = 1
    permeability_type = 'isotropic'
    permeability_initial = 1e-10
    poisson_ratio = 0.28
  []
  [seismic_plane]
    type = GolemMaterialTHMFracElastic
    block = 'seismic_events'
    material_type = 'frac'
    fracture_aperture_uo = constant_aperture
    porosity_initial = 1
    permeability_type = 'isotropic'
    permeability_initial = 1e-13
    poisson_ratio = 0.28
  []
  [well]
    type = GolemMaterialTHMElastic
    block = 'well_GtGrSk405'
    material_type = 'well'
    scaling_factor_initial = 0.05
    porosity_initial = 1
    permeability_type = 'isotropic'
    permeability_initial = 3.125e-04
    poisson_ratio = 0.28
  []
[]

[BCs]
  [pressure_bc]
    type = FunctionDirichletBC
    function = pressure_gradient
    boundary = 'west east'
    variable = pore_pressure
  []
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
  [load_z_top]
    type = FunctionNeumannBC
    function = lithostatic_load
    boundary = 'top'
    variable = disp_z
  []
  # Diagnostic horizontal tractions representing the surface-relative
  # tangential component of the original vertical load on the inclined top
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

[UserObjects]
  [scaling]
    type = GolemScaling
    characteristic_stress = 1e6
    characteristic_length = 1
    characteristic_time = 1
    characteristic_temperature = 1
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
[]



[Postprocessors]
  [sandstone_xx]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = stress_xx
  []
  [sandstone_yy]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = stress_yy
  []
  [sandstone_zz]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = stress_zz
  []
  [sandstone_xy]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = stress_xy
  []
  [sandstone_yz]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = stress_yz
  []
  [sandstone_xz]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = stress_xz
  []
  [sandstone_pp]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4035'
    variable = pore_pressure
  []
  [volcanics_xx]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = stress_xx
  []
  [volcanics_yy]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = stress_yy
  []
  [volcanics_zz]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = stress_zz
  []
  [volcanics_xy]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = stress_xy
  []
  [volcanics_yz]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = stress_yz
  []
  [volcanics_xz]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = stress_xz
  []
  [volcanics_pp]
    type = PointValue
    outputs = 'csv'
    point = '-184.387799 -16.646398 -4135'
    variable = pore_pressure
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
                             1 0 100'
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
  end_time = 1.0
  [TimeStepper]
    type = TimeSequenceStepper
    time_sequence = '0 1'
  []
[]

[Outputs]
  print_linear_residuals = false
  perf_graph = true
  checkpoint = true
  [out]
    type = Exodus
  []
  [csv]
    type = CSV
  []
[]