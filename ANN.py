import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import pickle
import neptune.new as neptune
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Flatten

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0


run = neptune.init_run(
    project="yhya201389/Selected-images-ann",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjM2FjZDdmOC00MWI4LTRkZGMtYjk3Yi1hYjZmOGVkYzMwNGMifQ==",
) 

neptune_callback = NeptuneCallback(
    run=run,
    base_namespace="visualizations",  # optionally set a custom namespace name
    log_model_diagram=True,
    log_on_batch=True,
)

neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

model = keras.Sequential()

model.add(Flatten(input_shape=(10000, 1)))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(20))
model.add(Activation('softmax'))

# opt = keras.optimizers.Adam(learning_rate=.1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
hist = model.fit(X, y,batch_size = 20, epochs=10, verbose=1, validation_split = .2, callbacks=[neptune_cbk])

# evaluate the model
model.save("wh.model")

