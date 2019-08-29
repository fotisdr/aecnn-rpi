# Real-time Python framework for speech enhancement using AECNNs

Fotios Drakopoulos, Deepak Baby and Sarah Verhulst. Real-time audio processing on a Raspberry Pi using deep neural
networks, 23rd International Congress on Acoustics, 9 to 13 September 2019 in Aachen, Germany.

This work received funding from the European Research Council (ERC) under the Horizon 2020 Research and Innovation Programme (grant agreement No 678120 RobSpear)

----

The Keras framework for the implementation of the AECNN models is adapted from [here](https://github.com/deepakbaby/se_relativisticgan). The necessary scripts can be found in the AECNN folder.

## Preparation of data
1. The [dataset](https://datashare.is.ed.ac.uk/handle/10283/1942) of Valentini et al. was used for the training and testing of the models. The `download_dataset.sh` script can be used to download the dataset (it requires [sox](http://sox.sourceforge.net/) to downsample the data to 16 kHz).
    ```bash
    ./download_dataset.sh
    ```

2. The data need to be segmented in the training and testing sets depending on the desired window size (the input/output size of the AECNN model). For this, the variable `opts ['window_size']` needs to be defined in the `prepare_data.py` script.
    ```python
    python prepare_data.py
    ```

## Training the model
```python
python run_aecnn.py
```
The `opts` variable needs to be edited to modify the architecture configurations of the AECNN model.

## Running the trained model in real-time
```python
python audio_processing.py -m model_directory -n model_input_size -f keras
```
The directory of the trained model needs to be defined with the `-m` argument as well as the input/output size of the model with the `-n` argument. Keras or Tensorflow can be used as the frontend (`-f`) and 0% or 50% overlap (`-o`) or frame buffering (`-b`) can be applied.

----

### Benchmarking a model
A trained model can be benchmarked within the current framework in terms of execution time, in order to get the time constrains for different settings. A .wav file needs to be provided (`-i`) and the (parent) directory containing the model folder(s) needs to be defined (`-d`). This way, multiple models can be benchmarked with this script.
```python
python model_benchmark.py -i wav_file -d parent_directory -f keras -it 1
```

### Converting a Keras model to protobuf format (.pb)
A trained model can be converted to protobuf format for inference in Tensorflow:
```python
python tensorflow_converter.py -m model_directory
```

### Measure the complexity of a Keras model
The number of parameters and floating-point operations of a trained model can be computed:
```python
python measure_complexity.py -m model_directory
```
