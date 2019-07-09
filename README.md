# gcp-ai-tf-custom-op-predictor
A Custom Prediction Routine for Google AI Platform that can be used to deploy TensorFlow models with custom operators.

Certain TensorFlow models make use of custom operators, e.g. Google's [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1). These models can not be deployed on Google Cloud AI Platform using the standard TensorFlow runtime.

This repository contains a [Custom Prediction Routine](https://cloud.google.com/ml-engine/docs/tensorflow/custom-prediction-routines) which
mimics the API of the standard runtime and can be used to install and load libraries with custom operator code.

## Usage

1. Change `setup.py` and `predictor.py` to install and load the necessary library.
2. Build a package.

    ```
    python setup.py sdist
    ```

3. Upload the package to Google Cloud storage.

    ```
    gsutil cp dist/custom-op-tf-predictor-0.1.tar.gz $GS_BUCKET/
    ```

4. Create a model version using the custom predictor.

    ```
    gcloud beta ai-platform versions create $VERSION_NAME\
        --model=$MODEL_NAME\
        --origin=$MODEL_DIR\
        --python-version=3.5\
        --runtime-version=1.13\
        --package-uris=$GS_BUCKET/custom-op-tf-predictor-0.1.tar.gz\
        --prediction-class=predictor.CustomOpTfPredictor
    ```
