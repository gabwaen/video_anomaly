import pandas as pd
import numpy as np
import cv2

np.random.seed(42)

file = '0Ow4cotKOuw_0'
# file = '1Kbw1bUw_1'
# file = '39BFeYnbu-I_0'
# file = 'EFv961C5RgY_0'



filename = f"./RWF-2000/val/Fight/{file}.avi"


table = pd.read_csv(f'{file}.tsv', sep='\t')
print(table.head())

info = dict(zip(list(table['ID'].iloc), [int(x) for x in table['Radius'].iloc]))
color  = 1	

expand_ratio = 2



######
## Intersections
vid = cv2.VideoCapture(filename)
w = int(vid.get(3))
h = int(vid.get(4))

mser = cv2.MSER_create()


for ts in range(1, table.shape[1]-1):
	col = f'Instant {ts}'

	ret, img = vid.read()
	if ret != True:
		break
	
	frame = np.zeros((h, w), dtype=np.uint8)

	for i in range(table.shape[0]):
		center = table[col].iloc[i]
		if str(center) == 'nan':
			continue
		center = tuple(int(x) for x in eval(center))

		_id = table['ID'].iloc[i]

		radius = int(info[_id]*expand_ratio)

		tmp = np.zeros((h, w), dtype=np.uint8)
		cv2.circle(  tmp, center, radius, color, -1)
		frame = cv2.add(frame, tmp)

	frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)[1]

	if (frame > 0).any():

		regions = mser.detectRegions(frame)[:-1]
		greater = (0, 0)

		for c in regions:
			for region in c:
				included = []

				reg = np.zeros((h, w), dtype=np.uint8)
				hull = cv2.convexHull(np.asarray(region).reshape(-1, 1, 2))
				cv2.fillPoly(reg, [hull], 255)

				area = reg.sum()/255

				for i in range(table.shape[0]):
					center = table[col].iloc[i]
					if str(center) == 'nan':
						continue
					center = tuple(int(x) for x in eval(center))

					_id = table['ID'].iloc[i]

					radius = int(info[_id]*expand_ratio)
					tmp = np.zeros((h, w), dtype=np.uint8)
					cv2.circle(  tmp, center, radius, 255, -1)
					tmp = cv2.bitwise_and(tmp,reg)

					if (tmp > 0).any():
						included.append(list(center)+[radius])

				print (len(included))
				xs, ys, rds = zip(*included)
				xmin = min(xs)
				xmax = max(xs)
				ymin = min(ys)
				ymax = max(ys)

				xc, yc = (xmax+xmin)//2, (ymax+ymin)//2
				radius = ((xmax-xmin)+(ymax-ymin))//2

				# print (area, radius**2)
				# if area < (radius**2)/len(included) :
				# 	continue

				radius = int(sum(rds)/(len(rds)-1))
				cv2.circle(img, (xc, yc), radius, (0,255,0), 5)

				radius = int(sum(rds)/(len(rds)-1))

				if radius > greater[0]:
					greater = (radius, (xc,yc))

		if greater != (0,0):
			radius, (xc,yc) = greater
			cv2.circle(img, (xc, yc), radius, (0,127,255), 10)

	tmp = cv2.bitwise_or(img, img, mask=frame)
	img = cv2.addWeighted(img, 0.5, tmp, 0.5, 0, 0)

	cv2.imshow('Tracer Analyzer', img)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break
vid.release()
######
