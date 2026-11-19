# ParaView reservoir example states

The state files are optional visualisation setups for the GOLEM simple-reservoir examples; they are not standalone datasets. Run the corresponding example section first to generate `wbgeo.e`:

- `model1_reservoir_visualisation.pvsm` refers to `golem_simple_reservoir_output/wbgeo.e`, written by section 3.4 of `WBGeo1.0_model1.py`.
- `model2_reservoir_visualisation.pvsm` refers to `golem_custom_reservoir_output/wbgeo.e`, written by section 3.5 of `WBGeo1.0_model2.py`.

Both folders are created inside the GOLEM workspace of the examples (`./projects` by default, or `GOLEM_PROJECT_DIR` if set).

These paths are relative rather than author-specific absolute paths. ParaView may resolve them relative to its working directory, not the state file's directory. When loading the state, use ParaView's **Search files under directory** or file relocation prompt to select the actual generated Exodus output.
