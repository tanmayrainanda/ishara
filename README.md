---
license: apache-2.0
---

## Ishara: ASL Fingerspelling Recognition

Ishara is a deep learning model designed for accurate recognition of American Sign Language (ASL) fingerspelling. It is based on a hybrid architecture that combines **Squeezeformer** and **Conformer** blocks with **Conv1D layers** for efficient feature extraction from hand, face, and pose landmark data.

This model is a submission to the Google ASLFR Competition and achieves robust performance on character-level prediction tasks.

---

## Model Description

Ishara processes sequences of normalized hand, face, and pose landmarks to predict fingerspelled words at the character level. The architecture is designed to handle temporal variability and missing data using a combination of:

- **Squeezeformer blocks**: For efficient sequence modeling.
- **Conformer blocks**: For enhanced feature extraction.
- **Conv1D layers**: For initial temporal feature extraction.

The output predictions are character-level sequences optimized using **Connectionist Temporal Classification (CTC)** loss.

---

## Dataset

The model was trained and evaluated on the dataset provided by the [Google ASLFR Competition](https://www.kaggle.com/competitions/asl-fingerspelling), which consists of:

- **Hand landmarks**: 21 points each for left and right hands.
- **Face landmarks**: 40 key points.
- **Pose landmarks**: 10 key points.
- **Labels**: Text sequences representing fingerspelled words.

---

## Usage

### Inference with TFLite

The model is available in TensorFlow Lite format for real-time inference. To use the model:

```python
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()

# Define input-output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input a sequence of landmarks
input_data = ... # Preprocessed input sequence
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicted Sequence:", output_data)
```

---

### Training Workflow

You can replicate the training process using TensorFlow. The training loop is as follows:

```python
from model import get_model

# Define the model
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

---

## Model Evaluation

The model's performance is evaluated using:

- **Levenshtein Distance**: Measures character-level accuracy.
- **Normalized Character Error Rate (CER)**: Quantifies the model's robustness.
- **Real-Time Inference Speed**: Assessed on 1080p video inputs.

---

## Results

- **Normalised Levenshtein Distance**: [0.728]
- **Inference Speed**: [200ms]
- **Model Size**: [17.9 Mb]

---

## Deployment

The model is optimized for deployment in real-time systems using TensorFlow Lite. This makes it suitable for integration into mobile and embedded systems for ASL recognition tasks.

---

## License

This model is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

---

## Acknowledgments

- **Google ASLFR Competition**: For providing the dataset.
- **TensorFlow Team**: For the deep learning framework.
- **Paper Authors**: For inspiring the architecture.
  - [Squeezeformer](https://arxiv.org/abs/2206.00888)
  - [Conformer](https://arxiv.org/abs/2005.08100)

---

## Citation

If you use this model, please consider citing:

```
@misc{ishara_asl,
  title={Ishara: ASL Fingerspelling Recognition},
  author={Niharika Gupta, Tanay Srinivasa, Tanmay Nanda, Zoya Ghoshal},
  year={2025},
  howpublished={\url{https://huggingface.co/ishara-asl}}
}
```

---

## Contact

For questions or collaboration, feel free to reach out:

- **Tanmay Nanda**: tanmaynanda360@gmail.com
