import bentoml
import numpy as np
from bentoml.io import NumpyNdarray


BENTO_MODEL_TAG = "timeseries_model:trurguaha67zouhl"
forcast_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()
forecast_service = bentoml.Service("forecasting", runners=[forcast_runner])


@forecast_service.api(input=NumpyNdarray(), output=NumpyNdarray())
async def forecast(input: np.ndarray) -> np.ndarray:
    return await forcast_runner.predict.async_run(input)
