import pandas as pd
import numpy as np
import math
import cv2

np.random.seed(42)




if __name__ == "__main__":
	# file = '0Ow4cotKOuw_0'
	file = 'EFv961C5RgY_0'
	# file = '10uSOcwS_0'



	filename = f"./RWF-2000/val/Fight/{file}.avi"
	# filename = f"./RWF-2000/val/NonFight/{file}.avi"


	table = pd.read_csv(f'{file}.tsv', sep='\t')
	print(table.head())

color  = 1	

_id = lambda x : table['ID'].iloc[x]


within_degree = lambda dgr, target : dgr <= target or dgr >= (360-target)
COLORS = np.random.randint(0, 255, size=(table.shape[0], 3), dtype="uint8")

def getVector(row, ts):
	c1, _ = get(row, ts)
	c2, _ = get(row, ts-1)

	if c1 is None or c2 is None:
		return (0,0)

	vec = np.asarray([c1[0]-c2[0], c1[1]-c2[1]])
	return vec / np.linalg.norm(vec)


expand_ratio = 1



def intersection_type(c1, r1, c2, r2):
	x1, y1 = c1
	x2, y2 = c2

	d = ((x2-x1)**2 + (y2-y1)**2)

	if d > (r1+r2)**2:
		return 0
	if d < (r1-r2)**2:
		return -1
	return 1



def get(row, ts):
	t = f'Instant {ts}'

	c = table[t].iloc[row]
	if str(c) == 'nan':
		return None, None
	c = tuple(int(x) for x in eval(c))

	r = int(info[_id(row)]*expand_ratio)

	return c, r



def minimal_enclosing_circle(c1, r1, c2, r2):
	eps = 1/2

	if r1 > r2:
		c1, c2 = c2, c1
		r1, r2 = r2, r1
	elif r1 == r2:
		eps = 0

	delta = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** (1/2)
	theta = eps + (r2 - r1)/(2*delta)

	r = int((r1+r2+delta)/2)
	c = [int((1-theta) * c1[i] + theta * c2[i]) for i in range(2)]

	return tuple(c), r











info = dict(zip(list(table['ID'].iloc), [int(x) for x in table['Radius'].iloc]))
old = {}


######
## Intersections
vid = cv2.VideoCapture(filename)
w = int(vid.get(3))
h = int(vid.get(4))

ret, img = vid.read()
cv2.imshow('Tracer Analyzer', img)
if cv2.waitKey(1) & 0xFF == ord('q'):
	exit(0)

for ts in range(2, table.shape[1]-1):
	ret, img = vid.read()
	if ret != True:
		break
	
	AoC = {} ## Areas of Conflict

	for i in range(table.shape[0]-1):
		c1, r1 = get(i, ts)
		v1 = getVector(i, ts)

		if c1 is None:
			continue

		color  = tuple([int(x) for x in COLORS[i]])
		cv2.circle(  img, c1, r1, color, 1)

		for j in range(i+1, table.shape[0]):
			c2, r2 = get(j, ts)
			v2 = getVector(j, ts)

			if c2 is None:
				continue

			if max([r1,r2])* 2/3 > min([r1,r2]):
				continue


			intersects = intersection_type(c1, r1, c2, r2)
			key = tuple([i,j])

			if not intersects:
				continue

			if intersects == -1 and key in old:
				# bigger = np.argmax([r1,r2])
				# AoC[tuple([i,j])] = {"center":[c1,c2][bigger], "radius":int([r1,r2][bigger]/expand_ratio)}
				AoC[key] = -1
				continue
			
			vd = np.asarray([c1[0]-c2[0], c1[1]-c2[1]])
			vd = vd / np.linalg.norm(vd)

			d1 = np.dot(v1, vd)
			d2 = np.dot(v2, vd)

			a1 = math.degrees(np.arccos(d1))
			a2 = math.degrees(np.arccos(d2))

			if within_degree(a1, 22) or within_degree(a2, 22) or key in old:
				AoC[key] = 1
				# cx = int((c1[0]+c2[0])/2)
				# cy = int((c1[1]+c2[1])/2)
				# AoC[tuple([i,j])] = {"center":(cx,cy), "radius":int((r1+r2)/expand_ratio)}

	circles = []
	for key, iType in AoC.items():
		i, j = key

		c1, r1 = get(i, ts)
		c2, r2 = get(j, ts)

		old[key] = 1
		
		if iType == -1:
			bigger = np.argmax([r1,r2])
			circles.append({"center":[c1,c2][bigger], "radius":int([r1,r2][bigger]/expand_ratio)})
			continue

		cx = int((c1[0]+c2[0])/2)
		cy = int((c1[1]+c2[1])/2)
		circles.append({"center":(cx,cy), "radius":int((r1+r2)/expand_ratio)})

	repeat = True
	while 1:
		print (repeat)
		if not repeat:
			break
		repeat = False

		i = 0
		while 1:
			if i >= len(circles):
				break

			kill = []
			new  = []
			obj1 = circles[i]

			c1 = obj1["center"]
			r1 = obj1["radius"]

			for j in range(i+1, len(circles)):
				obj2 = circles[j]

				c2 = obj2["center"]
				r2 = obj2["radius"]

				intersects = intersection_type(c1, r1, c2, r2)

				if not intersects:
					continue

				kill.append(j)
				new.append((c2, r2))


			if len(kill):
				for j in sorted(kill, reverse=True):
					del circles[j]

				new.append((c1, r1))
				while 1:
					c1, r1 = new.pop()
					c2, r2 = new.pop()

					c,r = minimal_enclosing_circle(c1, r1, c2, r2)

					new.append((c,r))

					if len(new)==1:
						break

				circles[i] = {"center":c, "radius":r}
				repeat = True


			i+=1

	for obj in circles:
		center = obj["center"]
		radius = obj["radius"]

		cv2.circle(  img, center, radius, 1, 1)


	cv2.imshow('Tracer Analyzer', img) ### ts:: 104 buga
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break


vid.release()
######
