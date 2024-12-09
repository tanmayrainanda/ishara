# ishara

## Overview

This project implements a deep learning model for American Sign Language (ASL) fingerspelling recognition as part of the Google ASLFR Competition. The model uses a hybrid architecture combining Squeezeformer and Conformer blocks with Conv1D layers to achieve accurate fingerspelling detection from hand landmark data.

## Dataset

The dataset is provided by Google's ASLFR (American Sign Language Fingerspelling Recognition) Competition and consists of:

- Hand landmark positions (x, y, z coordinates)
- Face landmark positions
- Pose landmarks
- Corresponding text labels (fingerspelled words)

### Data Format

- Input: Sequences of landmark coordinates
- Output: Character sequences representing fingerspelled words
- Features include:
  - 21 right hand landmarks
  - 21 left hand landmarks
  - 40 face landmarks
  - 10 pose landmarks

## Model Architecture

The model implements a hybrid architecture with the following key components:

1. **Input Processing**

   - Landmark normalization
   - Temporal padding and resizing
   - Masking for missing frames

2. **Core Architecture**

   - Configurable number of Conv1D blocks
   - Squeezeformer blocks for sequence processing
   - Conformer blocks for enhanced feature extraction
   - Efficient Channel Attention (ECA)
   - Gated Linear Units (GLU)
   - Swish activation functions

3. **Output Layer**
   - Dense layers with CTC loss
   - Character-level prediction

## Requirements

```
tensorflow>=2.6.0
tensorflow-addons
numpy
pandas
tqdm
matplotlib
python-Levenshtein
```

### Training

```python
# Configure model parameters
model = get_model(
    dim=256,
    num_conv_squeeze_blocks=2,
    num_conv_conform_blocks=2,
    kernel_sizes=[11, 5, 3],
    num_conv_per_block=3,
    dropout_rate=0.2
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=N_EPOCHS,
    callbacks=[validation_callback, lr_callback, WeightDecayCallback()]
)
```

### Inference

```python
# Load model and make predictions
interpreter = tf.lite.Interpreter("model.tflite")
prediction_fn = interpreter.get_signature_runner(REQUIRED_SIGNATURE)
output = prediction_fn(inputs=frame)
```

## Model Evaluation

The model is evaluated using:

- Levenshtein distance for character-level accuracy
- Normalized character error rate
- Real-time inference speed

## TFLite Conversion

The model can be converted to TFLite format for deployment:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)
tflite_model = converter.convert()
```

## Results

- Validation accuracy: [To be filled]
- Average inference time: [To be filled]
- Model size: [To be filled]

## License

 Copyright 2024 Niharika Gupta, Tanay Srinivasa, Tanmay Nanda, Zoya Ghoshal

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Acknowledgments

- Google ASLFR Competition for providing the dataset
- TensorFlow team for the framework
- Original Squeezeformer and Conformer paper authors

## References

1. Squeezeformer: https://arxiv.org/abs/2206.00888
2. Conformer:  https://arxiv.org/abs/2005.08100
3. Google ASLFR Competition: https://www.kaggle.com/competitions/asl-fingerspelling

## Contact

Tanmay Nanda - tanmaynanda360@gmail.com