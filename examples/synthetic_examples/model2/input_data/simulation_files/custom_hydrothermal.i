# Solving hydrothermal problem

[Mesh]
  [input_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
  [create_sideset]
    type = SideSetsFromNodeSetsGenerator
    input = input_mesh
  []
[]

[GlobalParams]
  pore_pressure = pore_pressure
  temperature = temperature
  scaling_uo = scaling
  supg_uo = supg
[]

[Variables]
  [./pore_pressure]
    order = FIRST
    family = LAGRANGE
    initial_condition = 0.0
  [../]
  [./temperature]
    order = FIRST
    family = LAGRANGE
    [InitialCondition]
      type = FunctionIC
      function = initial_temperature
    []
  [../]
[]

[Kernels]
  [./HKernel]
    type = GolemKernelH
    variable = pore_pressure
  [../]
  [./temp_time]
    type = GolemKernelTimeTSUPG
    variable = temperature
  [../]
  [./TKernel]
    type = GolemKernelT
    variable = temperature
  [../]
  [./THKernel]
    type = GolemKernelTHSUPG
    variable = temperature
  [../]
[]

[AuxVariables]
  [./vx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vz]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./darcyx]
    type = GolemDarcyVelocity
    variable = vx
    component = 0
  [../]
  [./darcyy]
    type = GolemDarcyVelocity
    variable = vy
    component = 1
  [../]
  [./darcyz]
    type = GolemDarcyVelocity
    variable = vz
    component = 2
  [../]
[]


[Functions]
  [initial_temperature]
    type = ParsedFunction
    expression = 'gradient*(z-z_bottom)+temperature_bottom'
    symbol_names = 'gradient z_bottom temperature_bottom'
    symbol_values = '-0.05 0.0 60.0'
  []
[]


[BCs]
  [pressure_bottom]
    type = DirichletBC
    variable = pore_pressure
    boundary = bottom
    value = 1000000.0
  []
  [pressure_top]
    type = DirichletBC
    variable = pore_pressure
    boundary = top
    value = 0.0
  []
  [temperature_bottom]
    type = DirichletBC
    variable = temperature
    boundary = bottom
    value = 60.0
  []
  [temperature_top]
    type = DirichletBC
    variable = temperature
    boundary = top
    value = 10.0
  []
[]

[Materials]
  [./block_1]
    type = GolemMaterialTHSUPG
    block = 1
    porosity_initial = 0.05
    porosity_uo = porosity_1
    permeability_initial = 1e-15
    permeability_uo = permeability_1
    fluid_viscosity_initial = 0.001
    fluid_viscosity_uo = fluid_viscosity_1
    fluid_density_initial = 1000.0
    fluid_density_uo = fluid_density_1
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_1
    fluid_heat_capacity_uo = fluid_heat_capacity_1
    solid_thermal_conductivity_uo = solid_thermal_conductivity_1
    solid_heat_capacity_uo = solid_heat_capacity_1
    solid_density_uo = solid_density_1
  [../]
  [./block_2]
    type = GolemMaterialTHSUPG
    block = 2
    porosity_initial = 0.15
    porosity_uo = porosity_2
    permeability_initial = 1e-14
    permeability_uo = permeability_2
    fluid_viscosity_initial = 0.001
    fluid_viscosity_uo = fluid_viscosity_2
    fluid_density_initial = 1000.0
    fluid_density_uo = fluid_density_2
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_2
    fluid_heat_capacity_uo = fluid_heat_capacity_2
    solid_thermal_conductivity_uo = solid_thermal_conductivity_2
    solid_heat_capacity_uo = solid_heat_capacity_2
    solid_density_uo = solid_density_2
  [../]
  [./block_3]
    type = GolemMaterialTHSUPG
    block = 3
    porosity_initial = 0.15
    porosity_uo = porosity_3
    permeability_initial = 1e-14
    permeability_uo = permeability_3
    fluid_viscosity_initial = 0.001
    fluid_viscosity_uo = fluid_viscosity_3
    fluid_density_initial = 1000.0
    fluid_density_uo = fluid_density_3
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_3
    fluid_heat_capacity_uo = fluid_heat_capacity_3
    solid_thermal_conductivity_uo = solid_thermal_conductivity_3
    solid_heat_capacity_uo = solid_heat_capacity_3
    solid_density_uo = solid_density_3
  [../]
  [./block_4]
    type = GolemMaterialTHSUPG
    block = 4
    porosity_initial = 0.15
    porosity_uo = porosity_4
    permeability_initial = 1e-14
    permeability_uo = permeability_4
    fluid_viscosity_initial = 0.001
    fluid_viscosity_uo = fluid_viscosity_4
    fluid_density_initial = 1000.0
    fluid_density_uo = fluid_density_4
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_4
    fluid_heat_capacity_uo = fluid_heat_capacity_4
    solid_thermal_conductivity_uo = solid_thermal_conductivity_4
    solid_heat_capacity_uo = solid_heat_capacity_4
    solid_density_uo = solid_density_4
  [../]
  [./block_5]
    type = GolemMaterialTHSUPG
    block = 5
    porosity_initial = 0.15
    porosity_uo = porosity_5
    permeability_initial = 1e-14
    permeability_uo = permeability_5
    fluid_viscosity_initial = 0.001
    fluid_viscosity_uo = fluid_viscosity_5
    fluid_density_initial = 1000.0
    fluid_density_uo = fluid_density_5
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_5
    fluid_heat_capacity_uo = fluid_heat_capacity_5
    solid_thermal_conductivity_uo = solid_thermal_conductivity_5
    solid_heat_capacity_uo = solid_heat_capacity_5
    solid_density_uo = solid_density_5
  [../]
  [./block_6]
    type = GolemMaterialTHSUPG
    block = 6
    material_type = FRAC
    fracture_aperture_uo = fracture_aperture_6
    porosity_initial = 0.02
    porosity_uo = porosity_6
    permeability_initial = 1e-19
    permeability_uo = permeability_6
    fluid_viscosity_initial = 0.001
    fluid_viscosity_uo = fluid_viscosity_6
    fluid_density_initial = 1000.0
    fluid_density_uo = fluid_density_6
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_6
    fluid_heat_capacity_uo = fluid_heat_capacity_6
    solid_thermal_conductivity_uo = solid_thermal_conductivity_6
    solid_heat_capacity_uo = solid_heat_capacity_6
    solid_density_uo = solid_density_6
  [../]
[]

[UserObjects]
  [./scaling]
    type = GolemScaling
    execute_on = 'INITIAL'
  [../]
  [./supg]
    type = GolemSUPG
    execute_on = 'INITIAL'
  [../]
  [./porosity_1]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_1]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_1]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_1]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_1]
    type = GolemFluidThermalConductivityConstant
    value = 0.6
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_1]
    type = GolemFluidHeatCapacityConstant
    value = 4186.0
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_1]
    type = GolemSolidThermalConductivityConstant
    value = 3.0
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_1]
    type = GolemSolidHeatCapacityConstant
    value = 846.1538461538462
    execute_on = 'INITIAL'
  [../]
  [./solid_density_1]
    type = GolemSolidDensityConstant
    value = 2600.0
    execute_on = 'INITIAL'
  [../]
  [./porosity_2]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_2]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_2]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_2]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_2]
    type = GolemFluidThermalConductivityConstant
    value = 0.6
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_2]
    type = GolemFluidHeatCapacityConstant
    value = 4186.0
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_2]
    type = GolemSolidThermalConductivityConstant
    value = 2.5
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_2]
    type = GolemSolidHeatCapacityConstant
    value = 846.1538461538462
    execute_on = 'INITIAL'
  [../]
  [./solid_density_2]
    type = GolemSolidDensityConstant
    value = 2600.0
    execute_on = 'INITIAL'
  [../]
  [./porosity_3]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_3]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_3]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_3]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_3]
    type = GolemFluidThermalConductivityConstant
    value = 0.6
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_3]
    type = GolemFluidHeatCapacityConstant
    value = 4186.0
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_3]
    type = GolemSolidThermalConductivityConstant
    value = 2.5
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_3]
    type = GolemSolidHeatCapacityConstant
    value = 846.1538461538462
    execute_on = 'INITIAL'
  [../]
  [./solid_density_3]
    type = GolemSolidDensityConstant
    value = 2600.0
    execute_on = 'INITIAL'
  [../]
  [./porosity_4]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_4]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_4]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_4]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_4]
    type = GolemFluidThermalConductivityConstant
    value = 0.6
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_4]
    type = GolemFluidHeatCapacityConstant
    value = 4186.0
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_4]
    type = GolemSolidThermalConductivityConstant
    value = 2.0
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_4]
    type = GolemSolidHeatCapacityConstant
    value = 807.6923076923077
    execute_on = 'INITIAL'
  [../]
  [./solid_density_4]
    type = GolemSolidDensityConstant
    value = 2600.0
    execute_on = 'INITIAL'
  [../]
  [./porosity_5]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_5]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_5]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_5]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_5]
    type = GolemFluidThermalConductivityConstant
    value = 0.6
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_5]
    type = GolemFluidHeatCapacityConstant
    value = 4186.0
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_5]
    type = GolemSolidThermalConductivityConstant
    value = 2.0
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_5]
    type = GolemSolidHeatCapacityConstant
    value = 807.6923076923077
    execute_on = 'INITIAL'
  [../]
  [./solid_density_5]
    type = GolemSolidDensityConstant
    value = 2600.0
    execute_on = 'INITIAL'
  [../]
  [./porosity_6]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_6]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_6]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_6]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_6]
    type = GolemFluidThermalConductivityConstant
    value = 0.6
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_6]
    type = GolemFluidHeatCapacityConstant
    value = 4186.0
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_6]
    type = GolemSolidThermalConductivityConstant
    value = 0.5
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_6]
    type = GolemSolidHeatCapacityConstant
    value = 884.6153846153846
    execute_on = 'INITIAL'
  [../]
  [./solid_density_6]
    type = GolemSolidDensityConstant
    value = 2600.0
    execute_on = 'INITIAL'
  [../]
  [./fracture_aperture_6]
    type = GolemFractureApertureConstant
    value = 40.0
    execute_on = 'INITIAL'
  [../]
[]

[Preconditioning]
  active = 'fieldsplit'
  [./fieldsplit]
    type = FSP
    topsplit = pT
    [./pT]
      splitting = 'p T'
      splitting_type = multiplicative
      petsc_options = '-snes_ksp_ew'
      petsc_options_iname = '-ksp_type -ksp_rtol -ksp_max_it -snes_type -snes_linesearch_type -snes_atol -snes_stol -snes_max_it'
      petsc_options_value = 'fgmres 1.0e-12 50 newtonls basic 1.0e-04 0 100'
    [../]
    [./p]
      vars = 'pore_pressure'
      petsc_options_iname = '-ksp_type -pc_type -pc_hypre_type'
      petsc_options_value = 'preonly hypre boomeramg'
    [../]
    [./T]
      vars = 'temperature'
      petsc_options_iname = '-ksp_type -pc_type -sub_pc_type -sub_pc_factor_levels -ksp_rtol -ksp_max_it'
      petsc_options_value = 'fgmres asm ilu 1 1.0e-04 500'
    [../]
  [../]
[]

[Executioner]
  type = Transient
  solve_type = Newton
  automatic_scaling = true
  start_time = 0.0
  end_time = 500000000000.0
  dt = 166666666666.66666
  dtmin = 166666666.66666666
  num_steps = 3
  nl_abs_tol = 1e-08
[]

[Outputs]
  time_step_interval = 1
  print_linear_residuals = true
  perf_graph = true
  [out]
    type = Exodus
    file_base = 'output/wbgeo'
  []
[]
