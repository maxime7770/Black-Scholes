

from keras.models import load_model

model_full = load_model("model_full.h5")
model_extremes = load_model("model_extremes.h5")
model_sparse = load_model("model_sparse.h5")


print('predicted price for model_full:', model_full.predict(
    [[0.0, 50.0, 0.01, 0.01, 0.001, 0.25]])[0, 0])
print('predicted price for model_extremes:', model_extremes.predict(
    [[0.0, 50.0, 0.01, 0.01, 0.001, 0.25]])[0, 0])
print('predicted price for model_sparse:', model_sparse.predict(
    [[0.0, 50.0, 0.01, 0.01, 0.001, 0.25]])[0, 0])
