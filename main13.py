'''
	go against wind
'''
# import library
import sys
import matplotlib.animation as animation
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import tensorflow as tf
from tensorflow_probability import edward2 as ed

# options
np.set_printoptions(threshold=sys.maxsize)

def unit_vector(v1):
    """ Returns the unit vector of the vector.  """
    return v1 / (np.linalg.norm(v1)+d)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'::

	>>> angle_between((1, 0, 0), (0, 1, 0))
	1.5707963267948966
	>>> angle_between((1, 0, 0), (1, 0, 0))
	0.0
	>>> angle_between((1, 0, 0), (-1, 0, 0))
	3.141592653589793
	"""
	if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
		return 0
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def WindMatrix(xLoc,preVecLoc):
	windRet = np.zeros((20,20))
	for boat in range(amountBoats):
		if preVecLoc[boat,0] < 0:
			windRet[xLoc[boat,0]%20,xLoc[boat,1]-1] = 1
			windRet[(xLoc[boat,0]+1)%20,xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]+1)%20,xLoc[boat,1]] = 0.6
			windRet[(xLoc[boat,0]+2)%20,xLoc[boat,1]-1] = 0.4
			windRet[(xLoc[boat,0]+2)%20,xLoc[boat,1]] = 0.25
		elif preVecLoc[boat,0] > 0:
			windRet[xLoc[boat,0]%20,xLoc[boat,1]-1] = 1
			windRet[xLoc[boat,0]-1,xLoc[boat,1]-1] = 0.6
			windRet[xLoc[boat,0]-1,xLoc[boat,1]] = 0.6
			windRet[xLoc[boat,0]-2,xLoc[boat,1]-1] = 0.4
			windRet[xLoc[boat,0]-2,xLoc[boat,1]] = 0.25
		else:
			windRet[xLoc[boat,0],xLoc[boat,1]-1] = 1

	return windRet

def WindMatrixself(xLoc,preVecLoc,boat):
	windRet = np.zeros((20,20))
	if preVecLoc[boat,0] < 0:
			windRet[xLoc[boat,0]%20,xLoc[boat,1]-1] = 1
			windRet[(xLoc[boat,0]+1)%20,xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]+1)%20,xLoc[boat,1]] = 0.6
			windRet[(xLoc[boat,0]+2)%20,xLoc[boat,1]-1] = 0.4
			windRet[(xLoc[boat,0]+2)%20,xLoc[boat,1]] = 0.25
	elif preVecLoc[boat,0] > 0:
		windRet[xLoc[boat,0]%20,xLoc[boat,1]-1] = 1
		windRet[xLoc[boat,0]-1,xLoc[boat,1]-1] = 0.6
		windRet[xLoc[boat,0]-1,xLoc[boat,1]] = 0.6
		windRet[xLoc[boat,0]-2,xLoc[boat,1]-1] = 0.4
		windRet[xLoc[boat,0]-2,xLoc[boat,1]] = 0.25
	else:
		windRet[xLoc[boat,0],xLoc[boat,1]-1] = 1

	return windRet

def WindType(xLoc,wind): # fix
	return wind[xLoc[1]-1,xLoc[0]-1]

def TVectorStart(): # Base travel possibility start
	VecRet = np.matrix([[0, -1, 0],
						[1, 1, 1],
						[1, 0, 2],
						[1, -1, 3],
						[0, -1, 4],
						[-1,-1, 5],
						[-1, 0, 6],
						[-1, 1, 7],
						[0, 0, 1],
						[0, 0, 2],
						[0, 0, 3],
						[0, 0, 5],
						[0, 0, 6],
						[0, 0, 7]])

	return VecRet, 14

def prioretyFunction(xNewLoc, xLoc, currentBoat,oppositeBoat,iObs, preVecObsLoc):
	#stillVar = (xLoc[oppositeBoat,0,iObs]-xLoc[oppositeBoat,0,iObs-1])*(xLoc[currentBoat,0,iObs]-xLoc[currentBoat,0,iObs-1]) != 0
	if (np.linalg.norm(xNewLoc[oppositeBoat,:]-xLoc[oppositeBoat,:,iObs]) == 0 and
		np.linalg.norm(xLoc[oppositeBoat,:,iObs]-xLoc[oppositeBoat,:,iObs-1]) == 0):# standing still
		return 0
	elif np.linalg.norm(xNewLoc[currentBoat,:]-xLoc[currentBoat,:,iObs]) == 0 and np.linalg.norm(xLoc[currentBoat,:,iObs]-xLoc[currentBoat,:,iObs-1]) == 0:# standing still
		return 1

	stillvar1 = preVecObsLoc[currentBoat,0,iObs-1]*preVecObsLoc[oppositeBoat,0,iObs-1] <= 0 and preVecObsLoc[currentBoat,1,iObs-1] >= 0 and preVecObsLoc[oppositeBoat,1,iObs-1] >= 0
	stillvar2 = preVecObsLoc[currentBoat,0,iObs-1]*preVecObsLoc[oppositeBoat,0,iObs-1] <= 0 and preVecObsLoc[currentBoat,1,iObs-1] <= 0 and preVecObsLoc[oppositeBoat,1,iObs-1] <= 0

	if (preVecObsLoc[currentBoat,1,iObs] >= 0 and preVecObsLoc[oppositeBoat,1,iObs] >= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] <= 0 and
		(stillvar1)): # Rule 10
		return  np.sign(preVecObsLoc[oppositeBoat,0,iObs])
	elif (preVecObsLoc[currentBoat,1,iObs] <= 0 and preVecObsLoc[oppositeBoat,1,iObs] <= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] <= 0 and
		(stillvar1)): # Rule 10
	 	return  np.sign(preVecObsLoc[oppositeBoat,0,iObs]-preVecObsLoc[currentBoat,0,iObs])



	stillvar1 = preVecObsLoc[currentBoat,0,iObs-1]*preVecObsLoc[oppositeBoat,0,iObs-1] <= 0 and preVecObsLoc[currentBoat,1,iObs-1] <= 0 and preVecObsLoc[oppositeBoat,1,iObs-1] >= 0
	stillvar2 = preVecObsLoc[currentBoat,0,iObs-1]*preVecObsLoc[oppositeBoat,0,iObs-1] <= 0 and preVecObsLoc[currentBoat,1,iObs-1] >= 0 and preVecObsLoc[oppositeBoat,1,iObs-1] <= 0
	if (preVecObsLoc[currentBoat,1,iObs] <= 0 and preVecObsLoc[oppositeBoat,1,iObs] >= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] <= 0 and
		(stillvar1)): # Rule 10
		return  1
	elif (preVecObsLoc[currentBoat,1,iObs] >= 0 and preVecObsLoc[oppositeBoat,1,iObs] <= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] <= 0 and
		(stillvar2)): # Rule 10
	 	return  -1

	if preVecObsLoc[currentBoat,1,iObs] <= 0 and preVecObsLoc[oppositeBoat,1,iObs] >= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] >= 0:# Rule 11
		return -1
	elif preVecObsLoc[currentBoat,1,iObs] >= 0 and preVecObsLoc[oppositeBoat,1,iObs] <= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] >= 0:# Rule 11
		return 1
	###
	close = ((np.abs(xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == 0)
			or (np.abs(xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == 1 and  np.abs(xLoc[oppositeBoat,0,iObs]-xLoc[currentBoat,0,iObs]) == 0)
			or (((xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == 1) and (preVecObsLoc[oppositeBoat,0,iObs] >= 0)*(xLoc[currentBoat,0,iObs]- xLoc[oppositeBoat,0,iObs]) == 1)
			or (((xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == -1) and (preVecObsLoc[oppositeBoat,0,iObs] >= 0)*(xLoc[currentBoat,0,iObs]- xLoc[oppositeBoat,0,iObs]) == -1))

	priorety = (xLoc[oppositeBoat,1,iObs] > xLoc[currentBoat,1,iObs]) or (xLoc[oppositeBoat,1,iObs] == xLoc[currentBoat,1,iObs] and (preVecObsLoc[oppositeBoat,0,iObs] >= 0)*(xLoc[oppositeBoat,0,iObs] < xLoc[currentBoat,0,iObs])) # fix
	priorety2 = xLoc[oppositeBoat,1,iObs] > xLoc[currentBoat,1,iObs]

	if (preVecObsLoc[currentBoat,1,iObs] >= 0 and preVecObsLoc[oppositeBoat,1,iObs] >= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] >= 0 and
		(close)): # Rule 11 upwind
		return  priorety
	elif (preVecObsLoc[currentBoat,1,iObs] >= 0 and preVecObsLoc[oppositeBoat,1,iObs] >= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] >= 0 and
		(~close)): # Rule 12
		return  ~priorety2

	close = ((np.abs(xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == 0)
			or (np.abs(xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == 1 and  np.abs(xLoc[oppositeBoat,0,iObs]-xLoc[currentBoat,0,iObs]) == 0)
			or ((np.abs(xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == 1) and  ~(preVecObsLoc[oppositeBoat,0,iObs]>= 0)*(xLoc[currentBoat,0,iObs]- xLoc[oppositeBoat,0,iObs]) == 1)
			or ((np.abs(xLoc[oppositeBoat,1,iObs]-xLoc[currentBoat,1,iObs]) == -1) and ~np.sign(preVecObsLoc[oppositeBoat,0,iObs])*(xLoc[currentBoat,0,iObs]- xLoc[oppositeBoat,0,iObs]) == -1))

	priorety = ((xLoc[oppositeBoat,1,iObs] > xLoc[currentBoat,1,iObs]) or (xLoc[oppositeBoat,1,iObs] == xLoc[currentBoat,1,iObs] and (preVecObsLoc[oppositeBoat,0,iObs]>=0)*(xLoc[oppositeBoat,0,iObs] < xLoc[currentBoat,0,iObs])))
	priorety2 = xLoc[oppositeBoat,1,iObs] > xLoc[currentBoat,1,iObs]
	if (preVecObsLoc[currentBoat,1,iObs] <= 0 and preVecObsLoc[oppositeBoat,1,iObs] <= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] >= 0 and (close)): # Rule 11 downwind
		return  priorety
	elif (preVecObsLoc[currentBoat,1,iObs] >= 0 and preVecObsLoc[oppositeBoat,1,iObs] >= 0 and preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] >= 0 and
		(~close)): # Rule 12
		return  priorety2

	else:
		return 0


def collidStart(xNewLoc, boat, iObs, xLoc, xs,preVecObsLoc):
	for i in range(amountBoats):
		if i != boat:
			if prioretyFunction(xNewLoc, xLoc, boat, i, iObs,preVecObsLoc) <= 0:
				if (xNewLoc[i,0]-xs[0])**2+(xNewLoc[i,1]-xs[1])**2 < 0.01:
					return 0
	return 1

def nextStep(boat, iObs, xObsLoc, preVecObsLoc, xNewLoc, preVecNewLoc, minVecLoc, createIndicator, infiniteloop, tmpMinSortLoc, sIndexLoc, sObsLoc, wind):
	minvalue = 10000000
	windSelf = WindMatrixself(xObsLoc[:,:,iObs+1],preVecObsLoc[:,:,iObs+1],boat)
	for k in range(sizeStart):
		xsx, xsy = vecStart[k,0], vecStart[k,1] # + int(round(preVecObsLoc[j,0,iObs]*sLoc[j])), vecStart[k,1] + int(round(preVecObsLoc[j,1,iObs]*sLoc[j]))
		distanceLoc = np.linalg.norm([xsx,xsy])
		vector1 = unit_vector(vecStart[vecStart[k,2],:2])
		degree = angle_between(vector1, preVecObsLoc[boat,:,iObs])


		Indextmp = (sObsLoc[boat, iObs] < 0.7 and (degree < 1 or degree == 0)) or (sObsLoc[boat, iObs] >= 0.7 and degree < 2.2 and distanceLoc != 0)
		if Indextmp:
			xs = [xObsLoc[boat,0,iObs]+xsx,xObsLoc[boat,1,iObs]+xsy]
			sTmp = sObsLoc[boat, iObs].copy()*max((1-degree/3.6-0.2*(distanceLoc == 0)),0) + (1.2-sObsLoc[boat, iObs])*0.36 ### supdate -0.1*distanceLoc == 0

			if xs[0]<2 or xs[0] > 17 or xs[1]< 2 or xs[1] > 17: # go outside
				tmpMin = 1000000
			elif (xObsLoc[boat,1,iObs] == 14 and (((xObsLoc[boat,0,iObs] <= 3 and xsy > 0) or (xObsLoc[boat,0,iObs] == 3 and xsy > 0 and xsy <0)) or
				((xObsLoc[boat,0,iObs] >=16 and xObsLoc[boat,0,iObs] <= 19) and xsy > 0) or (xObsLoc[boat,0,iObs] ==15 and xsy > 0 and xsy > 0))): ### FIX
				tmpMin = 1000000

			else:
				tmpMin = valuePosition (xs, degree, t, xsx, xsy, k, iObs, boat, sTmp, wind, windSelf)

			if infiniteloop > 5:
				tmpMin = tmpMin * np.random.normal(1, c)
			if createIndicator == 1:
				tmpMinSortLoc[boat,k] = tmpMin

			#collidStart(xNewLoc, boat, iObs, xLoc, xs)
			if tmpMin < minvalue and collidStart(xObsLoc[:,:,iObs+1],boat, iObs, xObsLoc[:,:,:], xs, preVecObsLoc):
				sObsLoc[boat, iObs+1] = sTmp
				xNewLoc[boat,:] = xs.copy()
				minVecLoc[boat] = tmpMin
				if distanceLoc == 0:
					preVecNewLoc[boat,:] =  unit_vector([vecStart[vecStart[k,2],0], vecStart[vecStart[k,2],1]])
				else:
					preVecNewLoc[boat,:] = vector1
				minvalue = tmpMin
				if createIndicator == 1:
					minSortK[boat,sIndexLoc] = k
		else:
			if createIndicator == 1:
				tmpMinSortLoc[boat,k] = 1000000

def valuePosition (xPostion, degree, t, xsx, xsy, k, iObs, boat, s, wind, windSelf):
	cs = 0
	wValue = 0.05*(wind[xs[0],xs[1]]-windSelf[xs[0],xs[1]])
	if (T-(t+iObs)) <= 2:
		cs = 1.5
	elif (T-(t+iObs)) <= 3:
		cs = 0.75
	elif (T-(t+iObs)) <= 10:
		cs = 0.2
	return Z[xPostion[0],xPostion[1] ,T-(t+iObs)-1, vecStart[k,2],strat[boat]]*(2-(s-1)*cs+s*wValue)

def potentialFuncStart0(fblr, stratIndex): #potential function

	zRet = np.zeros((20,20))
	init = 0
	jumpy = 0.5
	jumpx= 0.3
	jumpx2= 1
	if stratIndex == 0:
		for j in range(16):
			for i in range(15-j+1):
				zRet[min(4+i+j,19),15-j] = init+jumpx*i+j*jumpy

		for j in range(16):
			for i in range(5+j):
				zRet[4+j-i,14-j] = j*jumpy + 5/(j+1) + i*jumpx

		for j in range(15):
			for i in range(5-j):
				zRet[15+j+i,14-j] += 2/(j+1)
	elif stratIndex == 1:
		for j in range(16):
			for i in range(min(15+j,20)):

				zRet[5-min(j,5)+i,14-j] = 0.1+jumpx*i+j*jumpy

		for j in range(5):
			for i in range(5-j):
				zRet[4-i-j,14-j] = 2.0+jumpx*i+j*jumpy

		for j in range(15):
			for i in range(5+j):
				zRet[15-j+i,14-j] += 2/(j+1)

	elif stratIndex == 2:
		for j in range(15):
			for i in range(10):

				zRet[9-i,14-j] = 0.1+jumpx*i+j*jumpy
				zRet[10+i,14-j] = 0.1+jumpx*i+j*jumpy

		for j in range(15):
			for i in range(5+j):
				zRet[4+j-i,14-j] += 2/(j+1)

		for j in range(15):
			for i in range(5-j):
				zRet[15+j+i,14-j] += 2/(j+1)



	elif stratIndex == 3:
		for j in range(15):
			for i in range(10):

				zRet[9-i,14-j] = 0.1+jumpx*i+j*jumpy
				zRet[10+i,14-j] = 0.1+jumpx*i+j*jumpy

		for j in range(5):
			for i in range(5-j):
				zRet[4-i-j,14-j] = 2.0+jumpx*i+j*jumpy

		for j in range(15):
			for i in range(5+j):
				zRet[15-j+i,14-j] += 2/(j+1)

	elif stratIndex == 4:
		for j in range(15):
			for i in range(15):

				zRet[14-i,14-j] = 0.1+jumpx*i+j*jumpy

		for j in range(15):
			for i in range(5):
				zRet[15+i,14-j] = 0.1+jumpx*i+j*jumpy

		for j in range(16):
			for i in range(5+j):
				zRet[4+j-i,14-j] = j*jumpy + 5/(j+1) + i*jumpx

		for j in range(15):
			for i in range(5-j):
				zRet[15+j+i,14-j] += 2/(j+1)


	for j in range(5):
		for i in range(20):
			zRet[i,15+j] = 7+j*2

	if stratIndex == 0 or stratIndex == 2 or stratIndex == 4:
		if fblr < 6:
			zRet[:,14] = zRet[:,14] + 1
			zRet[:,13] = zRet[:,13] + 0.5
			zRet[:,12] + zRet[:,12] + 0.3
	elif stratIndex == 1 or stratIndex == 3:
		if (fblr >= 3 or fblr == 0):
			zRet[:,14] = zRet[:,14] + 1
			zRet[:,13] = zRet[:,13] + 0.5
			zRet[:,12] + zRet[:,12] + 0.3

	return zRet

def potentialFuncStartPost(tPre,fblr,stratIndex):
	Z[:, :, -tPre-1, fblr, stratIndex] = Z[:, :, -tPre, fblr, stratIndex].copy()
	if stratIndex == 0 or stratIndex == 2 or stratIndex == 4:
		Z[(5-tPre-1):(14-tPre-1),14+tPre+1,-tPre-1,fblr,stratIndex] = Z[(5-tPre+1):(14-tPre+1),14+tPre-1,-tPre,fblr,stratIndex].copy() - 0.3
	if stratIndex == 1 or stratIndex == 3:
		Z[(5+tPre+1):(14+tPre+1),14+tPre+1,-tPre-1,fblr,stratIndex] = Z[(5+tPre-1):(14+tPre-1),14+tPre-1,-tPre,fblr,stratIndex].copy() - 0.3


def potentialFuncStartpre(Zloc, fblr, t,stratIndex): #potential function
	if fblr == 0:
		fMat = np.matrix([[0.2, 0, 0.2],
						[0.15, 0.1, 0.15],
						[0.1, 0, 0.1]])

	elif fblr == 4:
		fMat = np.matrix([[0.07 , 0, 0.07],
						[0.12, 0.1, 0.12],
						[0.2, 0.2, 0.2]])




	elif fblr == 1:
		fMat = np.matrix([[0.25, 0, 0.3],
						[0.15, 0.1, 0.25],
						[0, 0.05, 0.15]])




	elif fblr == 7:
		fMat = np.matrix([[0.3, 0, 0.25],
						[0.2, 0.1, 0.15],
						[0.15, 0.05, 0]])


	elif fblr == 3:
		fMat = np.matrix([[0, 0, 0.15],
						[0.15, 0.1, 0.2],
						[0.2, 0.25, 0.25]])


	elif fblr == 5:
		fMat = np.matrix([[0.15, 0, 0],
						[0.2, 0.1, 0.15],
						[0.25, 0.25, 0.2]])



	elif fblr == 6:
		fMat = np.matrix([[0.1, 0, 1.5],
						[0, 0.1, 0.2],
						[0.1, 0.15, 0.15]])


	elif fblr == 2:
		fMat = np.matrix([[0.15, 0, 0.1],
						[0.2, 0.1, 0],
						[0.15, 0.15, 0.1]])



	procentMat = np.matrix([[0.8, 0, 0.8],
					[1, 1, 1],
					[1, 1, 1]])


	procentMat = procentMat / np.sum(procentMat) * 8

	if stratIndex == 0:
		sprintMat =np.matrix([[1, 0, 0],
		 					[0., 0, 0],
							[0, 0, 0]])
	elif stratIndex == 4 or stratIndex == 2:
		sprintMat =np.matrix([[1, 0, 0],
		 					[0, 0, 0],
							[0, 0, 0]])
	elif stratIndex == 3 or stratIndex == 1:
		sprintMat =np.matrix([[0, 0, 1],
		 					[0, 0, 0],
							[0, 0, 0]])
	else:
		sprintMat =np.matrix([[0, 0, 0],
		 					[0, 0, 0],
							[0, 0, 0]])

	fMat = fMat/np.sum(fMat)

	fAngel = np.matrix([[7, 0, 1],
		[6, 8, 2],
		[5, 4, 3]])

	zRet = np.zeros((20,20))
	for j in range(1,19):
		for i in range(1,19):
			tmp = 0
			for k in range(3):
				for l in range(3):
					if t < 5:
						if j >= 0 and j <= 19 and i == 14 and t < 5:
							if fAngel[l,k] == 8:
								for centerIndex in range(8):
									tmp = tmp + Zloc[j,i,centerIndex]*(fMat[l,k]+sprintMat[l,k])/8*procentMat[l,k]
							else:
								tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(fMat[l,k]+sprintMat[l,k])*procentMat[l,k]
						else:
							if fAngel[l,k] == 8:
								for centerIndex in range(8):
									tmp = tmp + Zloc[j,i,centerIndex]*(fMat[l,k]+sprintMat[l,k])/8
							else:
								tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(fMat[l,k]+sprintMat[l,k])
					else:
						if j >= 0 and j <= 19 and i == 14 and t < 5:
							if fAngel[l,k] == 8:
								for centerIndex in range(8):
									tmp = tmp + Zloc[j,i,centerIndex]*fMat[l,k]/8*procentMat[l,k]
							else:
								tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*fMat[l,k]*procentMat[l,k]
						else:
							if fAngel[l,k] == 8:
								for centerIndex in range(8):
									tmp = tmp + Zloc[j,i,centerIndex]*fMat[l,k]/8
							else:
								tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*fMat[l,k]
			zRet[j,i] += tmp
	zRet[:,0]=zRet[:,1]
	zRet[:,19]=zRet[:,18]
	zRet[0,:]=zRet[1,:]
	zRet[19,:]=zRet[18,:]
	return zRet.copy()


def nextStepSecond(boat, PositionNext, iObs, xObsLoc,preVecObsLoc,xNewLoc,preVecNewLoc, tmpMinSortLoc,sIndexLoc): # FIX!!!!
	sortMinVec = np.argsort(tmpMinSortLoc)
	index = 0
	while sortMinVec[index] != minSortK[boat, sIndexLoc]:
		index += 1

	if PositionNext > 0:
		for i in range(PositionNext):
			if index == 14:
				print("break")
				break
			index += 1
			xsx, xsy = vecStart[sortMinVec[index],0], vecStart[sortMinVec[index],1]# + int(round(preVecObsLoc[j,0,iObs]*sLoc[j])), vecStart[sortMinVec[index],1] + int(round(preVecObsLoc[j,1,iObs]*sLoc[j])) # snew?
			xs = [xObsLoc[boat,0,iObs]+xsx,xObsLoc[boat,1,iObs]+xsy]
			#collidStart(xNewLoc, boat, iObs, xLoc, xs)
			while collidStart(xObsLoc[:,:,iObs+1], boat, iObs, xObsLoc[:,:,:], xs, preVecObsLoc) <= 0 and index < 13: #fix
				index += 1
				xsx, xsy = vecStart[sortMinVec[index],0], vecStart[sortMinVec[index],1]# + int(round(preVecObsLoc[j,0,iObs]*sLoc[j])), vecStart[sortMinVec[index],1] + int(round(preVecObsLoc[j,1,iObs]*sLoc[j])) # snew?
				xs = [xObsLoc[boat,0,iObs]+xsx,xObsLoc[boat,1,iObs]+xsy]
	else:
		xsx, xsy = vecStart[sortMinVec[index],0], vecStart[sortMinVec[index],1] # + int(round(preVecObsLoc[j,0,iObs]*sLoc[j])), vecStart[sortMinVec[index],1] + int(round(preVecObsLoc[j,1,iObs]*sLoc[j])) # snew?
		xs = [xObsLoc[boat,0,iObs]+xsx,xObsLoc[boat,1,iObs]+xsy]

	xNewLoc[boat,:] = xs.copy()
	if sortMinVec[index] == 9:
		preVecNewLoc[boat,:] = [0,1]
	else:
		preVecNewLoc[boat,:] =  unit_vector([vecStart[vecStart[sortMinVec[index],2],0], vecStart[vecStart[sortMinVec[index],2],1]])


def recurrsion(crash, xObsLoc, preVecObsLoc, sIndex, xNewLoc, preVecNewLoc, boat, iObs, collidVecLoc, minVecLoc, tmpMinSortLoc, sObsLoc):
	if crash == 0 and iObs < Kobs: # make sure a crash hasn't happend
		for i in range(Kn): # observation before finding next position
			wind = WindMatrix(xObs[:,:,iObs+1,sIndex],preVecObs[:,:,iObs+1,sIndex])
			for j in range(amountBoats): # finding best next step
				if collidVecLoc[j] == 1:
					if j == boat:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 1, 0, tmpMinSortLoc[:,:,sIndex], sIndex, sObsLoc[:,:,sIndex], wind)
					else:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 0, 0, tmpMinSortLoc[:,:,sIndex], sIndex, sObsLoc[:,:,sIndex], wind)
				else:
					xNewLoc[j,:] = xObsLoc[j,:,iObs+1,0].copy()
					preVecNewLoc[j,:] = preVecObsLoc[j,:,iObs+1,0].copy()
			xObsLoc[:,:,iObs+1,sIndex] = xNewLoc.copy()
			preVecObsLoc[:,:,iObs+1,sIndex] = preVecNewLoc.copy()

		for tmpIndex in range(1, Kobsvec[iObs]):
			collidVecNew = collidVecLoc.copy()
			nextStepSecond(boat, tmpIndex, iObs, xObsLoc[:,:,:,sIndex], preVecObsLoc[:,:,:,sIndex], xNewLoc, preVecNewLoc, tmpMinSortLoc[boat,:,sIndex], sIndex)
			sIndextmp = sWidthNew[iObs,sIndex] + tmpIndex
			xObsLoc[:,:,:iObs+1,sIndextmp] = xObsLoc[:,:,:iObs+1,sIndex].copy()
			preVecObsLoc[:,:,:iObs+1,sIndextmp] = preVecObsLoc[:,:,:iObs+1,sIndex].copy()
			xObsLoc[:,:,iObs+1,sIndextmp] = xNewLoc.copy()
			preVecObsLoc[:,:,iObs+1,sIndextmp] = preVecNewLoc.copy()
			sObsLoc[:,:,sIndextmp] = sObsLoc[:,:,sIndex].copy()

			collidVar = 0
			infiniteloop = 0
			while collidVar == 0:
				wind = WindMatrix(xObs[:,:,iObs+1,sIndextmp],preVecObs[:,:,iObs+1,sIndextmp])
				infiniteloop += 1
				collidVar = 1
				for j in range(amountBoats):
					#collidStart(xNewLoc, boat, iObs, xLoc, xs)
					if collidStart(xObsLoc[:,:,iObs+1,sIndextmp],j, iObs, xObsLoc[:,:,:,sIndextmp], xObsLoc[j,:,iObs+1,sIndextmp], preVecObsLoc[:,:,:,sIndextmp]) <= 0: # if the boat collid
						if j == boat:
							nextStep(j, iObs,xObsLoc[:,:,:,sIndextmp],preVecObsLoc[:,:,:,sIndextmp],xNewLoc,preVecNewLoc, minVecLoc[:,sIndextmp], 1, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind)
						else:
							nextStep(j, iObs,xObsLoc[:,:,:,sIndextmp],preVecObsLoc[:,:,:,sIndextmp],xNewLoc,preVecNewLoc, minVecLoc[:,sIndextmp], 0, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind)
							collidVecNew[j] = 1
				xObsLoc[:,:,iObs+1,sIndextmp] = xNewLoc.copy()
				preVecObsLoc[:,:,iObs+1,sIndextmp] = preVecNewLoc.copy()
				for j in range(amountBoats):
					collidVar = collidStart(xObsLoc[:,:,iObs+1,sIndextmp],j, iObs, xObsLoc[:,:,:,sIndextmp], xObsLoc[j,:,iObs+1,sIndextmp], preVecObsLoc[:,:,:,sIndextmp])*collidVar # keeps being 1 if they don't collid

				if infiniteloop == 40:
					minVecLoc[boat,sIndextmp] = 10000
					collidVar == 0
					crash = 1

			wind = WindMatrix(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp])
			for i in range(amountBoats):
				if collidVecLoc[i] == 1:
					windSelf = WindMatrixself(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp],i)
					sObsLoc[i,iObs+1,sIndextmp] = sObs[i,iObs+1,sIndextmp]*(1-(wind[xObsLoc[i,0,iObs+1,sIndextmp],xObs[i,1,iObs+1,sIndextmp]]-windSelf[xObs[i,0,iObs+1,sIndextmp],xObs[i,1,iObs+1,sIndextmp]])/4)


			recurrsion(crash, xObsLoc, preVecObsLoc, sIndextmp, xNewLoc, preVecNewLoc, boat, iObs + 1, collidVecNew, minVecLoc, tmpMinSortLoc, sObsLoc)

		###### make sure they dont collid
		xNewLoc = xObsLoc[:,:,iObs+1,sIndex].copy()
		preVecNewLoc = preVecObsLoc[:,:,iObs+1,sIndex].copy()
		collidVecNew = collidVecLoc
		sIndextmp = sIndex
		collidVar = 0
		infiniteloop = 0
		while collidVar == 0:
			wind = WindMatrix(xObs[:,:,iObs+1,sIndextmp],preVecObs[:,:,iObs+1,sIndextmp])
			infiniteloop += 1
			collidVar = 1
			for j in range(amountBoats):
				#collidStart(xNewLoc, boat, iObs, xLoc, xs)
				if collidStart(xObsLoc[:,:,iObs+1,sIndex],j,iObs, xObsLoc[:,:,:,sIndex], xObsLoc[j,:,iObs+1,sIndex], preVecObsLoc[:,:,:,sIndextmp]) <= 0: # if the boat collid
					if j == boat:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 1, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp], wind)
					else:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 0, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind)
						collidVecNew[j] = 1
			xObsLoc[:,:,iObs+1,sIndex] = xNewLoc.copy()
			preVecObsLoc[:,:,iObs+1,sIndex] = preVecNewLoc.copy()
			for j in range(amountBoats):
				collidVar = collidStart(xObsLoc[:,:,iObs+1,sIndex],j,iObs, xObsLoc[:,:,:,sIndex], xObsLoc[j,:,iObs+1,sIndex], preVecObsLoc[:,:,:,sIndextmp])*collidVar # keeps being 1 if they don't collid

			if infiniteloop == 40:
				minVecLoc[boat,sIndex] = 10000
				collidVar == 0
				crash = 1


		wind = WindMatrix(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp])
		for i in range(amountBoats):
			if collidVecLoc[i] == 1:
				windSelf = WindMatrixself(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp],i)
				sObsLoc[i,iObs+1,sIndextmp] = sObs[i,iObs+1,sIndextmp]*(1-(wind[xObsLoc[i,0,iObs+1,sIndextmp]%20,xObs[i,1,iObs+1,sIndextmp]%20]-windSelf[xObs[i,0,iObs+1,sIndextmp]%20,xObs[i,1,iObs+1,sIndextmp]]%20)/4)
		# recurrsion(crash, xObsLoc, preVecObsLoc, sIndex, xNewLoc, preVecNewLoc, boat, iObs, collidVecLoc, minVecLoc)
		recurrsion(crash, xObsLoc, preVecObsLoc, sIndextmp, xNewLoc, preVecNewLoc, boat, iObs + 1, collidVecNew, minVecLoc, tmpMinSortLoc, sObsLoc)


start = time.time() # time to finish
startBoxesnew = np.matrix([[5,9,2,5],
				[8,12,2,5],
				[11, 15,2,5],
				[5,9,2,5],
				[8,12,2,5]]) # temporary start boxes

amountBoats = 10 # amount of boats
xMax, xMin, yMax, yMin=10 ,-10, 15, -5 #frame
xh, yh = 0.01, 0.01 #stepsize
xMesh, yMesh= int((xMax-xMin)/xh+1), int((yMax-yMin)/yh+1) # amount of nodes
K, Kn, KobsMax, Kobs2 = 1, 3, 5, 2 #observation rounds, Rounds before changing strate
d, T = 10 ** -4, 25 #avoid dividing by zero, Time
sWidthMax = 6

######
strat= np.zeros(amountBoats, dtype=  np.int) # Starting Stratergies
strat[0] = 0
strat[1] = 1
strat[2] = 2
strat[3] = 3
strat[4] = 4
strat[5] = 0
strat[6] = 1
strat[7] = 2
strat[8] = 3
strat[9] = 4

print(strat)
#fbout = np.zeros((K,amountBoats)) #finish positions

xs, x0, xNew = np.zeros(2, dtype=np.int), np.zeros((amountBoats,2), dtype=np.int), np.zeros((amountBoats,2), dtype=np.int) # update step, value of update, start position
xObs = np.zeros((amountBoats,2,KobsMax+1, sWidthMax), dtype=np.int) # observation x

xNextFinal = np.ones((amountBoats,2,KobsMax+1), dtype=np.int)
preVecFinal = np.ones((amountBoats,2,KobsMax+1))
sVecFinal = np.ones((amountBoats,KobsMax))

###### START
yMeshstart, xMeshstart = int(yMesh/40) + 1 ,int(xMesh/10) + 1
xhStart, yhStart = xh*10, yh*10

vecStart , sizeStart = TVectorStart()
preVec, preVecNew, preVecObs = np.zeros((amountBoats,2)), np.zeros((amountBoats,2)), np.zeros((amountBoats,2,KobsMax+1,sWidthMax))
xPath = np.zeros((amountBoats,2,T+1))
xPathVec = np.zeros((amountBoats,2,T+1))
tmpMinSort = np.ones((amountBoats, sizeStart, sWidthMax))
minSortK = np.ones((amountBoats,sWidthMax), dtype=np.int)
stratAmount = 10
Z = np.zeros((20,20,T+3,8,stratAmount))

colorVec = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(amountBoats)]


#################### #################### #################### #################### ####################
#################### #################### #################### #################### ####################
#################### #################### #################### #################### ####################
s0 = np.ones(amountBoats)
sObs = np.ones((amountBoats,KobsMax, sWidthMax))
save = 1

if save == 0:
	for stratIndex in range(0,stratAmount):
		for i in range(8):
			Z[:,:,0,i,stratIndex] = potentialFuncStart0(i,stratIndex)
			# if i == 0:
			# 	plt.imshow(np.transpose(Z[:,:,0,i,stratIndex]))
			# 	plt.colorbar()
			# 	plt.savefig('imagesPotential/Potential' + str(stratIndex) + '.png')
			# 	plt.show()


	for tPre in range(0,4):
		for stratIndex in range(0,stratAmount):
			for i in range(8):
				potentialFuncStartPost(tPre, i, stratIndex)
				# if stratIndex == 0 and i == 1:
				# 	plt.imshow(np.transpose(Z[:,:,-tPre-1,i,stratIndex]))
				# 	plt.colorbar()
				# 	plt.pause(1)
				# 	plt.show()



	for stratIndex in range(0,stratAmount):
		print(stratIndex)
		for t in range(T-1):
			for tvec in range(8):
				Z[:,:,t+1,tvec,stratIndex] = potentialFuncStartpre(Z[:,:,t,:,stratIndex],tvec,t,stratIndex)
				# if tvec == 0 and stratIndex == 4 and t < 10: #and t%4==0:
				# 	plt.imshow(np.transpose(Z[:,:,t+1,tvec,stratIndex]))
				# 	plt.colorbar()
				# 	plt.pause(1)
				# 	plt.show()

	np.save('potential.npy', Z)

else:
	Z = np.load('potential.npy')



# xUni = ed.Uniform(low=0*np.ones(amountBoats), high=math.pi/2*np.ones(amountBoats))
# with tf.Session() as sess:
#   initialAngel = xUni.eval()
initialAngel = np.random.uniform(low = 0.0, high = math.pi/2, size = amountBoats)


for i in range(amountBoats):
	preVec[i,0] = math.cos(initialAngel[i])
	preVec[i,1] = math.sin(initialAngel[i])
	error = 1
	infiniteloop = 0
	while(error):
		infiniteloop += 1
		error = 0
		startpositionx = np.random.randint(startBoxesnew[strat[i],0],startBoxesnew[strat[i],1])
		startpositiony = np.random.randint(startBoxesnew[strat[i],2],startBoxesnew[strat[i],3])
		x0[i,:] =  [startpositionx,startpositiony]
		for j in range(i):
			if np.linalg.norm(x0[i,:]-x0[j,:])<0.1:
				error = 1
		if infiniteloop == 40:
			print("Crash happend start")
			sys.exit()

xPath[:,:,0] = x0.copy()
xPathVec[:,:,0] = preVec.copy()

c=0.08

for t in range(T):
	if t == 0:
		Kobs = 2
		Kobsvec = [3,1]
		sWidthNew = np.matrix([0,0,0])
		sWidth = 3
		minVec = np.zeros((amountBoats,sWidth))
	elif T-t == 7:
		c= 0.02
		Kobs = 4
		Kobsvec = [3,2,1,1]
		sWidthNew = np.matrix([[0,0,0,0,0,0],[2,3,4,5,6,7]])
		sWidth = 6
		minVec = np.zeros((amountBoats,sWidth))
	elif T-t == 4:
		c= 0.01
		Kobs = 4
		Kobsvec = [3,1,1,1]
		sWidthNew = np.matrix([[0,0,0,0,0,0],[2,3,4,5,6,7]])
		sWidth = 3
		minVec = np.zeros((amountBoats,sWidth))
	elif T-t == 2:
		c = 0
		Kobs = 3
		Kobsvec = [3,1,1]
		sWidthNew = np.matrix([0,0,0])
		sWidth = 3
		minVec = np.zeros((amountBoats,sWidth))
	elif T-t == 1:
		c = 0
		Kobs = 1
		Kobsvec = [1,1,1]
		sWidthNew = np.matrix([0,0,0])
		sWidth = 1
		minVec = np.zeros((amountBoats,sWidth))

	for tmpIndex in range(sWidth):
		sObs[:,0,tmpIndex] = s0.copy()
		xObs[:,:,0,tmpIndex] = x0.copy()
		xObs[:,:,-1,tmpIndex] = xPath[:,:,t-1].copy()
		preVecObs[:,:,0,tmpIndex] = preVec.copy()
		preVecObs[:,:,-1,tmpIndex] = xPathVec[:,:,t-1].copy()
		#preVecObs[:,:,-2,tmpIndex] = xPathVec[:,:,t-2].copy()
	print("t", t)
	for iObs in range(Kobs): # observation rounds
		wind = WindMatrix(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0])
		for i in range(Kn): # observation before finding next position
			for j in range(amountBoats): # finding best next step
				nextStep(j, iObs,xObs[:,:,:,0],preVecObs[:,:,:,0],xNew,preVecNew ,minVec[:,0], iObs+1, 0, tmpMinSort[:,:,0], 0, sObs[:,:,0],wind)
			xObs[:,:,iObs+1,0] = xNew.copy()
			preVecObs[:,:,iObs+1,0] = preVecNew.copy()


	####### make sure they dont collid
		collidVar = 0
		infiniteloop = 0
		while collidVar == 0:
			wind = WindMatrix(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0])

			infiniteloop += 1
			collidVar = 1
			for j in range(amountBoats):
				#collidStart(xNewLoc, boat, iObs, xLoc, xs)
				if collidStart(xObs[:,:,iObs+1,0], j, iObs, xObs[:,:,:,0], xObs[j,:,iObs+1,0],preVecObs[:,:,:,0]) <= 0: # if the boat collid
					nextStep(j,iObs,xObs[:,:,:,0],preVecObs[:,:,:,0],xNew,preVecNew, minVec[:,0], iObs+1, infiniteloop, tmpMinSort[:,:,0], 0, sObs[:,:,0], wind)

			xObs[:,:,iObs+1,0] = xNew.copy()
			preVecObs[:,:,iObs+1,0] = preVecNew.copy()
			for j in range(amountBoats):
				collidVar = collidStart(xObs[:,:,iObs+1,0], j, iObs, xObs[:,:,:,0], xObs[j,:,iObs+1,0],preVecObs[:,:,:,0])*collidVar # keeps being 1 if they don't collid
			if infiniteloop == 40:
				print("Crash happend!!!")
				sys.exit()


		wind = WindMatrix(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0])
		for i in range(amountBoats):
			windSelf = WindMatrixself(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0],i)
			sObs[i,iObs+1,0] = sObs[i,iObs+1,0]*(1-(wind[xObs[i,0,iObs+1,0],xObs[i,1,iObs+1,0]]-windSelf[xObs[i,0,iObs+1,0],xObs[i,1,iObs+1,0]])/4)
########################################## ########################################## ##########################################
########################################## ########################################## ##########################################
########################################## Step 2
	for iObs2 in range(Kobs2):
		for boat in range(amountBoats):
			collidVec = np.zeros(amountBoats)
			collidVec[boat] = 1
			xNew = xObs[:,:,0,0].copy()
			preVecNew = preVecObs[:,:,0,0].copy()

			# recurrsion(crash, xObsLoc, preVecObsLoc, sIndex, xNewLoc, preVecNewLoc, boat, iObs, collidVecLoc, minVecLoc)
			recurrsion(0, xObs, preVecObs, 0, xNew, preVecNew, boat, 0, collidVec, minVec, tmpMinSort, sObs)

			if iObs2 + 1 == Kobs2:
				minimumArgument = np.argsort(minVec[boat,:]*np.random.normal(1, c, sWidth))
			else:
				minimumArgument = np.argsort(minVec[boat,:])
			xNextFinal[boat,:,:] = xObs[boat,:,:,minimumArgument[0]].copy()
			preVecFinal[boat,:,:] = preVecObs[boat,:,:,minimumArgument[0]].copy()
			sVecFinal[boat,:] = sObs[boat,:,minimumArgument[0]].copy()
		xObs[:,:,:,0] = xNextFinal[:,:,:].copy()
		preVecObs[:,:,:,0] = preVecFinal[:,:,:].copy()
		sObs[:,:,0] = sVecFinal[:,:].copy()

########################################## ########################################## ##########################################
########################################## ########################################## ##########################################
################ Step 3
	collidVar = 0
	infiniteloop = 0
	xNew = xNextFinal[:,:,1].copy()
	preVecNew = preVecFinal[:,:,1].copy()
	xNextFinal[:,:,-1] = xPath[:,:,t-1].copy()
	preVecFinal[:,:,-1] = xPathVec[:,:,t-1].copy()
	collidVec = np.zeros(amountBoats)
	while collidVar == 0:
		wind = WindMatrix(xNextFinal[:,:,1],preVecFinal[:,:,1])
		infiniteloop += 1
		collidVar = 1
		for j in range(amountBoats):
			#collidStart(xNewLoc, boat, iObs, xLoc, xs)
			if collidStart(xNextFinal[:,:,1], j, 0, xNextFinal[:,:,:], xNextFinal[j,:,1],preVecFinal[:,:,:]) <= 0: # if the boat collid
				collidVec[j] == 1
				nextStep(j,0,xNextFinal,preVecFinal,xNew,preVecNew, minVec[:,0], 0, infiniteloop, tmpMinSort[:,:,0], 0, sObs[:,:,0], wind)
		xNextFinal[:,:,1] = xNew.copy()
		preVecFinal[:,:,1] = preVecNew.copy()
		for j in range(amountBoats):
			collidVar = collidStart(xNextFinal[:,:,1], j, 0, xNextFinal[:,:,:], xNextFinal[j,:,1],preVecFinal[:,:,:])*collidVar # keeps being 1 if they don't collid

		if infiniteloop == 40:
			print("Crash happend second step")
			sys.exit()

	wind = WindMatrix(xNextFinal[:,:,1],preVecFinal[:,:,1]) #### FIX
	for i in range(amountBoats):
		if collidVec[i] == 1:
			windSelf = WindMatrixself(xNextFinal[:,:,1],preVecFinal[:,:,1],i)
			sObs[i,1,0] = sObs[i,1,0]*(1-(wind[xNextFinal[i,0,1],xNextFinal[i,1,1]]-windSelf[xNextFinal[i,0,1],xNextFinal[i,1,1]])/4)

########################################## ########################################## ##########################################
########################################## ########################################## ##########################################
###### end
	x0 = xNextFinal[:,:,1].copy()
	preVec = preVecFinal[:,:,1].copy()

	xPath[:,:,t+1] = x0.copy()
	xPathVec[:,:,t+1] = preVec.copy()
	s0 = sObs[:,1,0].copy()
	print(s0)

print("xPath", xPath)
#print("xPathVec", xPathVec)
print("xNextFinal", xNextFinal)
#print("preVecFinal", preVecFinal)
end = time.time()
print(end - start)
for t in range(T+1):
	print("t", t)
	fig = plt.figure(figsize=(5,2))
	plt.quiver(xPath[:,0,t]+0.5,xPath[:,1,t]+0.5, xPathVec[:,0,t], xPathVec[:,1,t], color = colorVec, scale=50)
	plt.plot([5,15],[15,15],'ro')
	plt.ylim(0, 20)
	plt.xlim(0,20)
	ax = fig.gca()
	ax.set_xticks(np.arange(0, 20, 1))
	ax.set_yticks(np.arange(0, 20, 1))
	plt.annotate(str(t),xy=(1,1))
	plt.grid()
	plt.savefig('imagesgif2/images' + str (t) + '.png')
	if t< 30:
		plt.pause(0.05)
	elif t < 50:
		plt.pause(0.2)
	else:
		plt.pause(3)
	plt.draw()
	plt.close()
#print(xPath)
