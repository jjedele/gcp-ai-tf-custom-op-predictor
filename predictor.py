from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.contrib.predictor import predictor

# registers operator library with TF on import
import tf_sentencepiece

RecordData = List[Dict[str, Any]]
TensorData = Dict[str, List[Any]]


class CustomOpTfPredictor:
    """A custom prediction routine for Google AI Platform that can be used to deploy
    TensorFlow models depending on custom operators."""

    @classmethod
    def from_path(cls, model_dir):
        """Create an instance of CustomOpTfPredictor from the given path.

        Args:
            model_dir (str): The path to the stored model.

        Returns:
            CustomOpTfPredictor: The created predictor instance.
        """
        predictor = tf.contrib.predictor.from_saved_model(model_dir)
        return cls(predictor)

    def __init__(self, predictor: predictor.Predictor):
        """Constructor.

        Args:
            predictor (predictor.Predictor): Predictor for loaded model.
        """
        self._predictor = predictor

    def predict(self, instances, **kwargs):
        """Make predictions for given data.

        Args:
            instances (RecordData): A list of prediction input instances.
            **kwargs (Any): A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            RecordData: A list of outputs containing the prediction results.
        """
        model_input = self._to_tensor_format(instances)
        model_output = self._predictor(model_input)
        reply = self._to_record_format(model_output)
        return reply

    def _to_tensor_format(self, instances):
        """Convert data from record format to tensor format.

        Args:
            instances (RecordData): Data instances as received in API request.

        Returns:
            TensorData: Mapping of fields to data tensors as expected by tf.contrib.predictor.

        Example:
            >>> p._to_tensor_format([{"k1": "v11", "k2": "v12"}, {"k1": "v21", "k2": "v22"}])
            {"k1": ["v11", "v21"], "k2": ["v12", "v22"]}

        Todos:
            This doctest is unstable because of dictionary key ordering. 
        """
        # collect keys
        keys = set()
        for instance in instances:
            keys.update(instance.keys())

        # restructure data
        restructured = {}
        for key in keys:
            # we use .get() over [] because it gives us np.ndarray elements
            # as plain Python data which is JSON serializable
            restructured[key] = [instance.get(key) for instance in instances]

        return restructured

    def _to_record_format(self, data):
        """Convert data from tensor format to record format.

        Args:
            data (RecordData): Data in tensor format as returned by tf.contrib.predictor.

        Returns:
            TensorData: List of JSON serializable data records.
        
        Example:
            >>> p._to_record_format({"k1": ["v11", "v21"], "k2": ["v12", "v22"]})
            [{"k1": "v11", "k2": "v12"}, {"k1": "v21", "k2": "v22"}]
        """
        keys = list(data.keys())
        n_preds = len(data[keys[0]])

        restructured = []
        for i in range(n_preds):
            restructured.append({key: data[key].item(i) for key in keys})

        return restructured

