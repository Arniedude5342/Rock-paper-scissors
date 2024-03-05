
import cv2
import numpy as np
import tensorflow as tf

camera = cv2.VideoCapture(0)
mymodel = tf.keras.models.load_model('keras_model.h5')

while True:
	
	status , frame = camera.read()

	if status:

		frame = cv2.flip(frame , 1)
		resize_frame = cv2.resize(frame,(224,224))
		resize_frame = np.expand_dims(resize_frame,axis=0)
		resize_frame = resize_frame/255
		predictions = mymodel.predict(resize_frame)
		
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %,")
		
		
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
