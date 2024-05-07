import fasttext


PRETRAINED_MODEL_PATH = 'project/src/main/scala/cs6320/models/lid.176.ftz'

model = fasttext.load_model(PRETRAINED_MODEL_PATH)

text = "This is a sample text in English."
predictions = model.predict(text)

predictions
