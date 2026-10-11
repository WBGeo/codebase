# ParaView reservoir example states

The state file is optional for visualisation examples for the GOLEM simple-reservoir workflows, but not for standalone datasets. Run the corresponding model first to generate `wbgeo.e`.

- `model1_reservoir_visualisation.pvsm` refers to `golem_simple_reservoir_output/wbgeo.e`.
- `model2_reservoir_visualisation.pvsm` refers to `golem_custom_reservoir_output/wbgeo.e`.

These paths are relative rather than author-specific absolute paths. ParaView may resolve them relative to its working directory, not the state file's directory. When loading the state, use ParaView's **Search files under directory** or file relocation prompt to select the actual generated Exodus output.

