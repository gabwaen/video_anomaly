import pandas as pd
import cv2

######
## Display
def display(filename):
	vid = cv2.VideoCapture(filename)

	while True:
		ret, frame = vid.read()
		if ret != True:
			break

		cv2.imshow('Tracer Analyzer', frame)

		key = cv2.waitKey(50)

		if key & 0xFF == ord('q'):
			break

		if key & 0xFF == ord('e'):
			exit()
	vid.release()
######

df = pd.read_csv('new_results.csv', sep=',')


FP = (df['Conflicts'] > 0) & (df['Class'] == 0)
FN = (df['Conflicts'] == 0) & (df['Class'] == 1)

# print (df['File'][FN])


for file in df['File'][FP]:
# for file in df['File'][FN]:
	path = f'./RWF-2000/val/NonFight/{file}' 
	# path = f'./RWF-2000/val/Fight/{file}' 

	display(path)
