from typing import List, Optional

import lightgbm as lgb
import pandas as pd

from mlserver import types
from mlserver.model import MLModel
from mlserver.utils import get_model_uri
from mlserver.codecs import NumpyCodec, NumpyRequestCodec


WELLKNOWN_MODEL_FILENAMES = ["model.bst"]


class LightGBMModel(MLModel):
    """
    Implementationof the MLModel interface to load and serve `lightgbm` models.
    """

    async def load(self) -> bool:
        model_uri = await get_model_uri(
            self._settings, wellknown_filenames=WELLKNOWN_MODEL_FILENAMES
        )

        self._model = lgb.Booster(model_file=model_uri)

        self._categorical_columns = self._load_categorical(self._model, model_uri)

        self.ready = True
        return self.ready

    def _load_categorical(
        self, model: lgb.Booster, model_file: str
    ) -> Optional[List[str]]:
        # LightGBM does not currently force trained categorical columns to
        # categorical during predict, so pull the `categorical_feature` from the
        # saved model
        # https://github.com/microsoft/LightGBM/issues/5244
        categorical_feature = None
        with open(model_file) as f:
            for line in f:
                if line.startswith("[categorical_feature: "):
                    content = (
                        line.replace("[categorical_feature: ", "")
                        .replace("]", "")
                        .strip()
                        .split(",")
                    )
                    categorical_feature = [
                        int(value) for value in content if value != ""
                    ]

        if categorical_feature is None:
            return None

        feature_name = model.feature_name()
        return [feature_name[i] for i in categorical_feature]

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        decoded = self.decode_request(payload, default_codec=NumpyRequestCodec)

        if isinstance(decoded, pd.DataFrame) and self._categorical_columns:
            overlap = set(self._categorical_columns) & set(decoded.columns)
            decoded = decoded.astype({column: "category" for column in overlap})

        prediction = self._model.predict(decoded)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[NumpyCodec.encode_output(name="predict", payload=prediction)],
        )
