# WBGeo Model 1 simple-reservoir Exodus block mapping:
#   1 2 3 = lithological 3D volume blocks (uniform_rock)
#   4 5   = 2D fracture blocks
#   6     = 1D well block

[Mesh]
  [input_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
  [boundary_sidesets]
    type = SideSetsFromNodeSetsGenerator
    input = input_mesh
  []
  [injection_node]
    type = ExtraNodesetGenerator
    input = boundary_sidesets
    coord = '400 500 500'
    use_closest_node = true
    new_boundary = injection_node
  []
[]

[Variables]
  [pore_pressure]
    order = FIRST
    family = LAGRANGE
    [InitialCondition]
      type = ConstantIC
      value = 1e7
    []
  []
  [temperature]
    order = FIRST
    family = LAGRANGE
    [InitialCondition]
      type = ConstantIC
      value = 150
    []
  []
[]

[GlobalParams]
  pore_pressure = pore_pressure
  temperature = temperature
  fluid_density_initial = 1e3
  fluid_density_uo = fluid_density_constant
  fluid_viscosity_initial = 1e-3
  fluid_viscosity_uo = fluid_viscosity_constant
  fluid_heat_capacity_uo = fluid_heat_capacity_constant
  fluid_thermal_conductivity_uo = fluid_thermal_conductivity_constant
  scaling_uo = scaling
  supg_uo = supg
[]

[Kernels]
  [Darcy]
    type = GolemKernelH
    variable = pore_pressure
  []
  [H_storage]
    type = GolemKernelTimeH
    variable = pore_pressure
  []
  [T_storage]
    type = GolemKernelTimeTSUPG
    variable = temperature
  []
  [Fourier_diffusion]
    type = GolemKernelT
    variable = temperature
  []
  [T_advection]
    type = GolemKernelTHSUPG
    variable = temperature
  []
[]

[BCs]
  [pressure_bc]
    type = DirichletBC
    variable = pore_pressure
    boundary = 'top bottom front back left right'
    value = 1e7
  []
  [temperature_bc]
    type = DirichletBC
    variable = temperature
    boundary = 'top bottom front back left right'
    value = 150
  []
  [injection_temperature]
    type = DirichletBC
    variable = temperature
    boundary = injection_node
    value = 70
  []
[]

[Materials]
  [uniform_rock]
    type = GolemMaterialTHSUPG
    block = '1 2 3'
    porosity_initial = 0.1
    porosity_uo = porosity_constant
    fluid_modulus = 2.5e9
    permeability_type = isotropic
    permeability_initial = 1e-15
    permeability_uo = permeability_constant
    solid_density_uo = solid_density_domain
    solid_thermal_conductivity_uo = lambda_domain
    solid_heat_capacity_uo = cp_domain
  []
  [fractures]
    type = GolemMaterialTHSUPG
    block = '4 5'
    material_type = FRAC
    fracture_aperture_uo = aperture_constant
    porosity_initial = 1
    porosity_uo = porosity_constant
    fluid_modulus = 2.5e9
    permeability_initial = 0
    permeability_uo = permeability_cubic_law
  []
  [well]
    type = GolemMaterialTHSUPG
    block = '6'
    material_type = well
    scaling_factor_initial = 0.05
    porosity_initial = 1
    porosity_uo = porosity_constant
    permeability_initial = 3.12e-4
    permeability_uo = permeability_constant
    fluid_modulus = 2.5e9
  []
[]

[AuxVariables]
  [frac_permeability]
    order = CONSTANT
    family = MONOMIAL
    block = '4 5'
  []
  [frac_aperture]
    order = CONSTANT
    family = MONOMIAL
    block = '4 5'
  []
[]

[DiracKernels]
  [injection]
    type = GolemDiracKernelTH
    variable = pore_pressure
    source_point = '400 500 500'
    source_type = injection
    constant_rate = 3
    start_time = 0
    end_time = 3153600014
  []
  [extraction]
    type = GolemDiracKernelTH
    variable = pore_pressure
    source_point = '600 500 500'
    source_type = extraction
    constant_rate = 3
    start_time = 0
    end_time = 3153600014
  []
[]

[AuxKernels]
  [frac_permeability_aux]
    type = MaterialStdVectorAux
    variable = frac_permeability
    property = permeability
    index = 0
    execute_on = TIMESTEP_END
  []
  [frac_aperture_aux]
    type = MaterialRealAux
    variable = frac_aperture
    property = scaling_factor
    block = '4 5'
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[UserObjects]
  [scaling]
    type = GolemScaling
    characteristic_stress = 1
    characteristic_length = 1
    characteristic_time = 1
    characteristic_temperature = 1
    execute_on = INITIAL
  []
  [supg]
    type = GolemSUPG
    execute_on = INITIAL
  []
  [fluid_density_constant]
    type = GolemFluidDensityConstant
    execute_on = INITIAL
  []
  [fluid_viscosity_constant]
    type = GolemFluidViscosityConstant
    execute_on = INITIAL
  []
  [porosity_constant]
    type = GolemPorosityConstant
    execute_on = INITIAL
  []
  [permeability_constant]
    type = GolemPermeabilityConstant
    execute_on = INITIAL
  []
  [permeability_cubic_law]
    type = GolemPermeabilityCubicLaw
    execute_on = INITIAL
  []
  [solid_density_domain]
    type = GolemSolidDensityConstant
    value = 2.6e3
    execute_on = INITIAL
  []
  [lambda_domain]
    type = GolemSolidThermalConductivityConstant
    value = 3
    execute_on = INITIAL
  []
  [cp_domain]
    type = GolemSolidHeatCapacityConstant
    value = 950
    execute_on = INITIAL
  []
  [aperture_constant]
    type = GolemFractureApertureConstant
    value = 1e-4
    execute_on = INITIAL
  []
  [fluid_heat_capacity_constant]
    type = GolemFluidHeatCapacityConstant
    value = 4.2e3
    execute_on = INITIAL
  []
  [fluid_thermal_conductivity_constant]
    type = GolemFluidThermalConductivityConstant
    value = 0.65
    execute_on = INITIAL
  []
[]

[Postprocessors]
  [pressure_injection]
    type = PointValue
    point = '400 500 500'
    variable = pore_pressure
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
  [temperature_injection]
    type = PointValue
    point = '400 500 500'
    variable = temperature
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
  [pressure_extraction]
    type = PointValue
    point = '600 500 500'
    variable = pore_pressure
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
  [temperature_extraction]
    type = PointValue
    point = '600 500 500'
    variable = temperature
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
[]

[Preconditioning]
  [fieldsplit]
    type = FSP
    full = true
    topsplit = HT
    [HT]
      splitting = 'H T'
      splitting_type = additive
      petsc_options_iname = '-ksp_type -ksp_rtol -ksp_max_it -snes_type -snes_linesearch_type -snes_atol -snes_rtol -snes_stol -snes_max_it'
      petsc_options_value = 'fgmres 1e-12 100 newtonls basic 1e-10 1e-4 0 1000'
    []
    [H]
      vars = pore_pressure
      petsc_options_iname = '-ksp_type -pc_type -pc_hypre_type'
      petsc_options_value = 'preonly hypre boomeramg'
    []
    [T]
      vars = temperature
      petsc_options_iname = '-ksp_type -pc_type -sub_pc_type -sub_pc_factor_levels -ksp_rtol -ksp_max_it'
      petsc_options_value = 'fgmres asm ilu 1 1e-4 500'
    []
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  start_time = 0
  end_time = 3153600014
  automatic_scaling = true
  compute_scaling_once = false
  [TimeStepper]
    type = TimeSequenceStepper
    time_sequence = '1 86400 604800 2592000 7776000 15552000 31536000 63072000 126144000 315360000 630720000 946080000 1261440000 1576800000 1892160000 2207520000 2522880000 2838240000 3153600014'
  []
[]

[Outputs]
  print_linear_residuals = true
  perf_graph = true
  print_nonlinear_converged_reason = true
  [exo]
    type = Exodus
    file_base = 'output/wbgeo'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [csv]
    type = CSV
  []
[]
