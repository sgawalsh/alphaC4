import tensorflow as tf
import engine, pdb, minimax, pickle, numpy, NNfunctions, config
from keras.utils import np_utils

simpleOrConvolutional = "convolutional"
policyOrValue = "policy"
title = "the_simple_champ"

FUNCTION_OUTPUT_MAP = {
	"policy": [minimax.miniTree.getMove, 7, "categorical_crossentropy", "boardsToMoves"],
	"value3Cat": [minimax.miniTree.genGameResults, 3, "categorical_crossentropy", "boardsToResults"],
	"value1Cat": [minimax.miniTree.genGameResults, 1, "sparse_categorical_crossentropy", "boardsToResults"]
}

MODEL_INIT_MAP = {
	"convolutional": NNfunctions.makeConvModel,
	"simple": NNfunctions.makeSimpleModel
}

functionOutputSet = FUNCTION_OUTPUT_MAP[policyOrValue]
modelInitFn = MODEL_INIT_MAP[simpleOrConvolutional]

boards, retData = NNfunctions.getPickle(functionOutputSet[3], functionOutputSet[0], config.miniMaxDefaultDepth, 200, functionOutputSet[1])

model = tf.keras.models.Sequential()
modelInitFn(model, boards.shape[1:], functionOutputSet[1])
model.compile(optimizer = 'adam', loss = functionOutputSet[2], metrics = ['accuracy'])
model.fit(boards, retData, epochs = 3, batch_size = 500)
model.summary()

boards_test, retData_test = NNfunctions.getPickle(functionOutputSet[3] + "Test", functionOutputSet[0], config.miniMaxDefaultDepth, 100, functionOutputSet[1])

val_loss, val_acc = model.evaluate(boards_test, retData_test)
print(val_loss, val_acc)

print(model.predict(boards_test[:10]))
print(retData_test[:10])

model.save("oldModels/" + policyOrValue + "/" + simpleOrConvolutional + "/" + title)
#new_model = tf.keras.models.load_model("the_simple_champ")

#predictions = new_model.predict([x_test])
#print(predictions)