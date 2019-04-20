import os, tensorflow as tf, config, head2Head, pdb

def modelFaceoff():
	folderList = os.listdir(os.path.dirname(os.path.realpath(__file__)) + "\\models")
	if folderList:
		print("Models:")
		
		_ = 0
		modelArray = []
		while _ < 2:# load 2 models
			while True:
				for i, folder in enumerate(folderList):
					print(str(i + 1) + ". " + folder)
				try:
					modelInd = int(input(("Which model do you want to load for player {}?\n> ".format(_))))
					if modelInd > len(folderList) or modelInd < 1:
						print("Value out of range")
					else:
						break
				except ValueError:
					print("Which model do you want to load?\n> ")
			
			if modelInd:
				modelName = os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + folderList[modelInd - 1]
				modelArray.append([tf.keras.models.load_model(modelName + "\\the_value_champ"), tf.keras.models.load_model(modelName + "\\the_policy_champ"), config.getIntResponse("What should the recursion count be?")], folderList[modelInd - 1])
				_ += 1
		head2Head.head2Head(False, modelArray[0][3], modelArray[1][3], modelArray[0][0], modelArray[0][1], modelArray[1][0], modelArray[1][1], config.getIntResponse("How many games should be played?"), modelArray[0][2], modelArray[1][2])
	else:
		print("No models exist")
				
modelFaceoff()