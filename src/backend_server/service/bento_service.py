import bentoml
import numpy as np
from bentoml.io import NumpyNdarray


BENTO_MODEL_TAG = "timeseries_model:trurguaha67zouhl"
forcast_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()
forcast_service = bentoml.Service("forcasting", runners=[forcast_runner])


@forcast_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def forecast(input: np.ndarray) -> np.ndarray:
    return forcast_runner.predict.run(input)
