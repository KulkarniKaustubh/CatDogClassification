import tensorflow as tf
from tensorflow import keras as K

# loading model
jsonFile = open('cats_vs_dogs.json', 'r')
loadedJSONModel = jsonFile.read()
jsonFile.close()
loadedModel = K.models.model_from_json(loadedJSONModel)

# loading weights using this model
loadedModel.load_weights('cats_vs_dogs_weights.h5')

# compiling the loaded model
loadedModel.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# evaluating the loaded model
loadedModel.evaluate(validationData, batch_size = BATCH_SIZE)
