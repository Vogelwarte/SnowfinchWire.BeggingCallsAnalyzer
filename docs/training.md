## Using the application to train new models

### Using feature extraction model
1. The files that will be used to train the model should all be copied to a desired folder. They should all be accompanied by a .txt file containing audacity labels with an appropriate name.
    * The name of the file with the labels should contain the full name of the audio file (without the extension), that can be followed by any suffix. A correct pair of audio and label files may look as follows:
        ```
        AT_1_8_1_22-BA09_BS04-247AA5015C0309FF_20220608_120000.flac
        AT_1_8_1_22-BA09_BS04-247AA5015C0309FF_20220608_120000_ACG.txt
        ```
2. Fill the `train_fe.yml` configuration file with the following:
    * `training_data_path` - path to the files prepared in step 1
    * `win_length` - window length used for FFT
    * `hop_length` - hop length used for FFT
    * `window_type` - window type used for FFT
    * `overlap_percentage` - value from 0 to 1. How much of the window needs to overlap with an event to consider that the window contains the event.

4. Run one of the following command in the project directory depenging on which model you chose `SnowfinchWire.BeggingCallsAnalyzer` (NOT `SnowfinchWire.BeggingCallsAnalyzer/beggingcallsanalyzer`). This directory should contain the `train.py` script.
```
py -3.9 train.py fe train_fe.yml
```

### Using deep learning model (OpenSoundscape)

**Warning**: training the deep learning model without a GPU might take a long time. It is also _recommended_ by the developers of OpenSoundscape to perform it on Linux, however it should still work on Windows.

1. The files that will be used to train the model should all be copied to a desired folder. They should all be accompanied by a .txt file containing audacity labels with an appropriate name.
    * The name of the file with the labels should contain the full name of the audio file (without the extension), that can be followed by any suffix. A correct pair of audio and label files may look as follows:
        ```
        AT_1_8_1_22-BA09_BS04-247AA5015C0309FF_20220608_120000.flac
        AT_1_8_1_22-BA09_BS04-247AA5015C0309FF_20220608_120000_ACG.txt
        ```
2. Cut the training files into smaller parts by using the `training_chunks_from_Audacity.py` script. `SoX` is a required system dependency.
    * Specify the input directory with the `-i` option and the output directory with the `-o` option. Example invocation:
        ```
        python training_chunks_from_Audacity.py -i ./training_recordings -o ./training_recordings_chunks
        ```

3. Fill the `train_oss.yml` configuration file with the following:
    * `training_data_path` - path to the audio chunks prepared in step 2
    * `win_length` - window length used for FFT
    * `batch_size` - batch side used in mini-batch neural network learning
    * `num_workers` - how many subprocesses to use for data loading 
    * `epochs` - maximum number of epochs used for learning
4. Run one of the following command in the project directory depenging on which model you chose `SnowfinchWire.BeggingCallsAnalyzer` (NOT `SnowfinchWire.BeggingCallsAnalyzer/beggingcallsanalyzer`). This directory should contain the `train.py` script.
```
py -3.9 train.py oss train_oss.yml
```