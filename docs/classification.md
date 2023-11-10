# Using the application to classify audio files

### A note on folder structure

It is recommended to keep the following folder and filename structure: `{input_directory}\<brood_name>\any_name_you_want\<date>_<time>.{extension}`. This will allow graphs and summary files to be automatically created. The parts in curly braces (`{...}`) are parameters that you will need to provide to the program. 

Example of a correct folder structure: `C:\Desktop\classification_input\Furka10_22a\22_06_07\20220603_135337.WAV`. 

In this case, the input parameters are as follows (where you should put them will be explained later):
* `input_directory`: `C:\Desktop\classification_input`
* `extension`: `WAV`


### Using the deature extraction model
1. The files that will be undergoing prediction should all be copied to a desired directory.
2. Fill the configuration file `classify_fe.yml` with the following:
    * `model_path` - path to the file extraction model (with `.skops` extension).
    * `input_directory` - directory with optional subdirectories containing the files that will undergo prediction.
    * `output_directory` - directory that the predicted labels and summaries will be saved in.
    * `merge_window` - events at most this number of seconds apart will be joined together (default: 3.0).
    * `cut_length` - events at most this long will be deleted(default: 2.2).
    * `extension` - file extension without the leading dot (default: flac).
    * `processing_batch_size` - the number of files that will be processed simultaneously. The bigger this number, the higher the memory requirements (default: 100).
    * `create_plots` - whether or not to create summary csv and plots. Will emit a warning if true and a proper file directory is not provided, though the prediction will procees without error. (default: true).
3. Run one of the following command in the project directory depenging on which model you chose `SnowfinchWire.BeggingCallsAnalyzer` (NOT `SnowfinchWire.BeggingCallsAnalyzer/beggingcallsanalyzer`). This directory should contain the `classify.py` script.
```
py -3.9 classify.py fe --config classify_fe.yml
```

### Using the deep learning model (OpenSoundscape)
1. The files that will be undergoing prediction should all be copied to a desired folder.
2. Fill the `classify_oss.yml` configuration file with the following:
    * `model_path` - path to the file extraction model (with `.model` extension).
    * `input_directory` - directory with optional subdirectories containing the files that will undergo prediction.
    * `output_directory` - directory that the prediction will be saved in.
    * `merge_window` - Feeding events at most this number of seconds apart will be joined together (default: 3.0).
    * `cut_length` - Feeding events at most this long will be deleted (default: 2.2).
    * `contact_merge_window` - Contact events at most this number of seconds apart will be joined together (default: 10).
    * `contact_cut_length` - Contact events at most this long will be deleted (default: 2).
    * `extension` - file extension without the leading dot (default: flac).
    * `threshold` - values above the threshold are classified as feeding. If the model classified feeding too often, consider increasing this value, if it classifies feeding too rarely, consider decreasing it. (default: 0.85).
    * `contact_threshold` - values above the threshold are classified as feeding. If the model classified contact too often, consider increasing this value, if it classifies contact too rarely, consider decreasing it. (default: 0.887).
    * `processing_batch_size` - the number of files that will be processed simultaneously. The bigger this number, the higher the memory requirements (default: 100).
    * `inference_batch_size` - batch size used by the neural network.  The bigger this number, the higher the memory and CPU requirements(default: 100).
    * `create_plots` - whether or not to create summary csv and plots. Will emit a warning if true and a proper file directory is not provided, though the prediction will procees without error. (default: true).
3. Run one of the following command in the project directory depenging on which model you chose `SnowfinchWire.BeggingCallsAnalyzer` (NOT `SnowfinchWire.BeggingCallsAnalyzer/beggingcallsanalyzer`). This directory should contain the `classify.py`.
```
py -3.9 classify.py oss --config classify_oss.yml
```