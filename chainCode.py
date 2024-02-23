codeList = [5, 6, 7, 4, -1, 0, 3, 2, 1]

def getChainCode(x1, y1, x2, y2):
	dx = x2 - x1
	dy = y2 - y1
	hashKey = 3 * dy + dx + 4
	return codeList[hashKey]

'''This function generates the list of
chaincodes for given list of points'''
def generateChainCode(ListOfPoints):
	chainCode = []
	for i in range(len(ListOfPoints) - 1):
		a = ListOfPoints[i]
		b = ListOfPoints[i + 1]
		chainCode.append(getChainCode(a[0], a[1], b[0], b[1]))
	return chainCode


'''This function generates the list of points for
a straight line using Bresenham's Algorithm'''
def Bresenham2D(x1, y1, x2, y2):
	ListOfPoints = []
	ListOfPoints.append([x1, y1])
	xdif = x2 - x1
	ydif = y2 - y1
	dx = abs(xdif)
	dy = abs(ydif)
	if(xdif > 0):
		xs = 1
	else:
		xs = -1
	if (ydif > 0):
		ys = 1
	else:
		ys = -1
	if (dx > dy):

		# Driving axis is the X-axis
		p = 2 * dy - dx
		while (x1 != x2):
			x1 += xs
			if (p >= 0):
				y1 += ys
				p -= 2 * dx
			p += 2 * dy
			ListOfPoints.append([x1, y1])
	else:

		# Driving axis is the Y-axis
		p = 2 * dx-dy
		while(y1 != y2):
			y1 += ys
			if (p >= 0):
				x1 += xs
				p -= 2 * dy
			p += 2 * dx
			ListOfPoints.append([x1, y1])
	return ListOfPoints

def DriverFunction(x_values,y_values):
    result=''
    for i in range(0,len(x_values)-1):
        (x1, y1) = (x_values[i], y_values[i])
        (x2, y2) = (x_values[i+1], y_values[i+1])
        ListOfPoints = Bresenham2D(x1, y1, x2, y2)
        chainCode    = generateChainCode(ListOfPoints)
        chainCodeString = "".join(str(e) for e in chainCode)
        # print ('Chain code for the straight line from', (x1, y1),'to', (x2, y2), 'is', chainCodeString)
        result=result+chainCodeString
    return result


