from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from keras.losses import mean_squared_error # type: ignore
import bentoml


def load_model_and_save_to_bento(model_path: Path) -> None:
    model = load_model(model_path, custom_objects={"mse": mean_squared_error})
    with tf.device("/device:GPU:0"):
        bento_model = bentoml.keras.save_model("timeseries_model", model)
    print(f"bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    path = Path("src/service/Model/best_tot_Exp_model.h5")
    load_model_and_save_to_bento(path)
