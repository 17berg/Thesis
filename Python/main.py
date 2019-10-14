# import library
#import tensorflow as tf
#from tensorflow_probability import edward2 as ed

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from mainafter import orderBoats
import constants
import os

# options
np.set_printoptions(threshold=sys.maxsize)

if 0: # print results
	print(np.load('saveStrat.npy'))
	print(np.load('saveAverage.npy'))
	sys.exit()


###### CONSTANTS ######
save = 0 # load utility space matrix or generate a new one
save2 = 0 # load utility field post start or generate a new one
save3 = 0 # load windBase or generate a new one

windSlowc = constants.windSlowc # constant for how the wind effect the boat
speedIncreasec = constants.speedIncreasec # constant for how the speed is increaded by wind
maxSpeed = constants.maxSpeed # constant for how the speed is increaded by wind (Max)
tackSlowc = constants.tackSlowc # constant how tacking effect speed
distanceSpeedc = constants.distanceSpeedc # constant how distance effect speed
d = constants.d # Avoid dividing by zero

xMax, xMin, yMax, yMin=constants.xMax ,constants.xMin, constants.yMax, constants.yMin #frame
xh, yh = constants.xh, constants.yh #stepsize
xMesh, yMesh= constants.xMesh, constants.yMesh # amount of nodes

legalMoveShiftc = 0.7 # constant for when the legal move change
legalMoveShiftc2 = 1 # constant for when the legal move change
degree1 = 1 # degree boats can tack with speed under legalMoveShiftc
degree2 = 2.2 # degree boats can tack with speed over legalMoveShiftc
degree3 = 1.7 # degree boats can tack with speed over legalMoveShiftc2

stillc = 0.2 # constant that determine how good it is to stand sill
avoidWindc = 0.2 # constant for how the sailors avoid the wind shadow
jumpy = 0.5 # y jump in value for initial utility function
jumpx= 0.15 # x jump in value for initial utility function
Kn = 3 # constant for finding the next step
KnPath = 2 # round for finding the best path

kEvolution = 3 # amount of rounds before the worst boat update strategy
evolutionStep = 10 # amount of evolutionary steps

sWidthMax = 24 # maximum amount of path evaluated
KobsMax = 5 # maximum amount of future steps seen
T = 25 # Time units
tsprintc = 7 # time units sprinting
uniformMovec = 0 # constant how uniformly the boats can move in all direction, in range [0,1]

riskc = 1 # constant on how much risk the boats are willing to take, in range [1,KobsMax), the higher the higher risk with incremental jumps
chancec = 5 # constant on how much chance the boats are willing to take , in range [0,infinity), the higher the lower chance


startBoxesnew = np.matrix([[3,27,2,14], # start boxes
				[3,27,2,14],
				[3, 27,2,14],
				[3,27,2,14],
				[3,27,2,14],
				[3,27,2,14]])

###### Stratergy ######
stratAmount = 6 # amount of Strategies
amountBoats = constants.amountBoats # amount of boats
strat= np.zeros(amountBoats, dtype=  np.int) # Starting Strategies
fbout = np.zeros((amountBoats, kEvolution), dtype = np.int) # position of boats during rounds

# strategy space
strat[0] = 2
strat[1] = 2
strat[2] = 3
strat[3] = 3
strat[4] = 3
strat[5] = 3
strat[6] = 4
strat[7] = 4
strat[8] = 4
strat[9] = 4
strat[10] = 4
strat[11] = 4

saveStrat = np.zeros((amountBoats,evolutionStep+1), dtype=  np.int) # Save Starting Strategies
saveAverage = np.zeros((6,evolutionStep), dtype=  np.float) # Save Starting Strategies
saveStrat[:,0] = strat.copy()



###### Pre start functions ######

def inbetween(xLoc, currentBoat,oppositeBoat,iObs, preVecObsLoc, betweenBoat):# Checks if a boat is inbetween two boats, used for right of way rules
	between =  (xLoc[betweenBoat,0,iObs] >= np.min([xLoc[currentBoat,0,iObs],xLoc[oppositeBoat,0,iObs]]) and xLoc[betweenBoat,0,iObs] <= np.max([xLoc[currentBoat,0,iObs],xLoc[oppositeBoat,0,iObs]])
	and xLoc[betweenBoat,1,iObs] >= np.min([xLoc[currentBoat,1,iObs],xLoc[oppositeBoat,1,iObs]]) and xLoc[betweenBoat,1,iObs] <= np.max([xLoc[currentBoat,1,iObs],xLoc[oppositeBoat,1,iObs]]))
	if between:
		return 1
	return -1

def overlapp(xLoc, currentBoat,oppositeBoat,iObs, preVecObsLoc): # Check if two boats are overlapping
	""" Returns if the boats overlapp or if one is clear ahead
	-1 oppositeBoat priorety (not overlapping)
	1 currentBoat priorety (not overlapping)
	0 overlapping
	"""
	if preVecObsLoc[oppositeBoat,1,iObs] == 0: # only clear astern
		if preVecObsLoc[oppositeBoat,0,iObs] > 0: # rule 12
			if xLoc[currentBoat,0,iObs] - xLoc[oppositeBoat,0,iObs] < 0:
				return -1
		else:
			if xLoc[currentBoat,0,iObs] - xLoc[oppositeBoat,0,iObs] > 0:
				return -1
		return xLoc[currentBoat,1,iObs] - xLoc[oppositeBoat,1,iObs] < 0 # lower boat is leeward boat rule 11

	if preVecObsLoc[currentBoat,1,iObs] == 0: # only clear astern
		if preVecObsLoc[currentBoat,0,iObs] > 0: # rule 12
			if xLoc[oppositeBoat,0,iObs] - xLoc[currentBoat,0,iObs] < 0:
				return 1
		else:
			if xLoc[oppositeBoat,0,iObs] - xLoc[currentBoat,0,iObs] > 0:
				return 1
		return xLoc[currentBoat,1,iObs] - xLoc[oppositeBoat,1,iObs] < 0 # lower boat is leeward boat rule 11


	xCenter = xLoc[currentBoat,:,iObs] - xLoc[oppositeBoat,:,iObs]# if current boat is clear astern, rule 12
	if preVecObsLoc[oppositeBoat,1,iObs] > 0:
		if xCenter[1] + xCenter[0] < -1 and preVecObsLoc[oppositeBoat,0,iObs] > 0:
			return -1

		elif xCenter[1] - xCenter[0] > 1  and preVecObsLoc[oppositeBoat,0,iObs] < 0:
			return -1



	elif preVecObsLoc[oppositeBoat,1,iObs] < 0:
		if xCenter[1] + xCenter[0] > 1 and preVecObsLoc[oppositeBoat,0,iObs] < 0:
			return -1
		elif xCenter[1] - xCenter[0] < -1 and preVecObsLoc[oppositeBoat,0,iObs] > 0:
			return -1

	# Current boat priorety
	xCenter = xLoc[oppositeBoat,:,iObs] - xLoc[currentBoat,:,iObs] # if opposite boat is clear astern, rule 12
	if preVecObsLoc[currentBoat,1,iObs] > 0:
		if  xCenter[1] + xCenter[0] < -1 and preVecObsLoc[currentBoat,0,iObs] > 0:
			return 1
		elif xCenter[1] - xCenter[0] > 1 and preVecObsLoc[currentBoat,0,iObs] < 0:
			return 1

	elif preVecObsLoc[currentBoat,1,iObs] < 0:
		if xCenter[1] + xCenter[0] > 0 and preVecObsLoc[currentBoat,0,iObs] < -1:
			return 1
		elif xCenter[1] - xCenter[0] < 0 and preVecObsLoc[currentBoat,0,iObs] > 1:
			return 1

	return 0 # boats are overlapping

def unit_vector(v1): # gives back unit vector
    """ Returns the unit vector of the vector. """
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

def WindMatrix(xLoc,preVecLoc): # Gives the wind shadow for all boats
	windRet = np.zeros((30,30))
	for boat in range(amountBoats):
		windRet[xLoc[boat,0],xLoc[boat,1]] += 5
		if preVecLoc[boat,0] < 0 and preVecLoc[boat,1] >= 0:
				windRet[xLoc[boat,0],xLoc[boat,1]-1] += 1
				windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]-1] += 0.6
				windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]] += 0.6
				windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]-1] += 0.4
				windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]] += 0.25
		elif preVecLoc[boat,0] < 0 and preVecLoc[boat,1] <= 0:
				windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]-1] += 0.6
				windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]] += 1
				windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]-1] += 0.6
				windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]] += 0.6

		elif preVecLoc[boat,0] > 0 and preVecLoc[boat,1] >= 0:
			windRet[xLoc[boat,0],xLoc[boat,1]-1] += 1
			windRet[xLoc[boat,0]-1,xLoc[boat,1]-1] += 0.6
			windRet[xLoc[boat,0]-1,xLoc[boat,1]] += 0.6
			windRet[xLoc[boat,0]-2,xLoc[boat,1]-1] += 0.4
			windRet[xLoc[boat,0]-2,xLoc[boat,1]] += 0.25

		elif preVecLoc[boat,0] < 0 and preVecLoc[boat,1] <= 0:
				windRet[(xLoc[boat,0]-1),xLoc[boat,1]-1] += 0.6
				windRet[(xLoc[boat,0]-1),xLoc[boat,1]] += 1
				windRet[(xLoc[boat,0]-2),xLoc[boat,1]-1] += 0.6
				windRet[(xLoc[boat,0]-2),xLoc[boat,1]] += 0.6
		else:
			windRet[xLoc[boat,0],xLoc[boat,1]-1] = 0.6

	return windRet

def WindMatrixself(xLoc,preVecLoc,boat): # Gives the wind shadow give boat
	windRet = np.zeros((30,30))
	windRet[xLoc[boat,0],xLoc[boat,1]] += 5
	if preVecLoc[boat,0] < 0 and preVecLoc[boat,1] >= 0:
			windRet[xLoc[boat,0]%30,xLoc[boat,1]-1] = 1
			windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]] = 0.6
			windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]-1] = 0.4
			windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]] = 0.25
	elif preVecLoc[boat,0] < 0 and preVecLoc[boat,1] <= 0:
			windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]+1)%30,xLoc[boat,1]] = 1
			windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]+2)%30,xLoc[boat,1]] = 0.6

	elif preVecLoc[boat,0] > 0 and preVecLoc[boat,1] >= 0:
		windRet[xLoc[boat,0],xLoc[boat,1]-1] = 1
		windRet[xLoc[boat,0]-1,xLoc[boat,1]-1] = 0.6
		windRet[xLoc[boat,0]-1,xLoc[boat,1]] = 0.6
		windRet[xLoc[boat,0]-2,xLoc[boat,1]-1] = 0.4
		windRet[xLoc[boat,0]-2,xLoc[boat,1]] = 0.25

	elif preVecLoc[boat,0] < 0 and preVecLoc[boat,1] <= 0:
			windRet[(xLoc[boat,0]-1),xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]-1),xLoc[boat,1]] = 1
			windRet[(xLoc[boat,0]-2),xLoc[boat,1]-1] = 0.6
			windRet[(xLoc[boat,0]-2),xLoc[boat,1]] = 0.6
	else:
		windRet[xLoc[boat,0],xLoc[boat,1]-1] = 1

	return windRet

def TVectorStart(): # Base travel vector for start
	VecRet = np.matrix([
						[1, 1, 0],
						[1, 0, 1],
						[1, -1, 2],
						[0, -1, 3],
						[-1,-1, 4],
						[-1, 0, 5],
						[-1, 1, 6],
						[0, 0, 0],
						[0, 0, 2],
						[0, 0, 4],
						[0, 0, 6],

						# two steps
						[2, 1, 0],
						[2, 0, 1],
						[2, -1, 2],
						[1, -2, 2],
						[-1,-2, 5],
						[-2,-1, 5],
						[-2, 0, 5],
						[-2, 1, 6]])

	return VecRet, 19

def prioretyFunction(xNewLoc, xLoc, currentBoat,oppositeBoat,iObs, preVecObsLoc): # return which boat have right of way

	if (np.linalg.norm(xNewLoc[oppositeBoat,:]-xLoc[oppositeBoat,:,iObs]) == 0 and
		np.linalg.norm(xLoc[oppositeBoat,:,iObs]-xLoc[oppositeBoat,:,iObs-1]) == 0):# standing still
		return 0
	elif np.linalg.norm(xNewLoc[currentBoat,:]-xLoc[currentBoat,:,iObs]) == 0 and np.linalg.norm(xLoc[currentBoat,:,iObs]-xLoc[currentBoat,:,iObs-1]) == 0:# standing still
		return 1


	if (preVecObsLoc[currentBoat,0,iObs]*preVecObsLoc[oppositeBoat,0,iObs] < 0 and
		preVecObsLoc[currentBoat,0,iObs-1]*preVecObsLoc[oppositeBoat,0,iObs-1] < 0 ): # Rule 10
		return  np.sign(preVecObsLoc[oppositeBoat,0,iObs])


	elif (preVecObsLoc[currentBoat,0,iObs] == 0 and
		preVecObsLoc[currentBoat,0,iObs-1] == 0 ): # Running
		return  np.sign(preVecObsLoc[oppositeBoat,0,iObs])
	elif (preVecObsLoc[oppositeBoat,0,iObs] == 0 and
		preVecObsLoc[oppositeBoat,0,iObs-1] == 0 ): # Running
		return  -np.sign(preVecObsLoc[currentBoat,0,iObs])


	if (preVecObsLoc[currentBoat,1,iObs] < 0 and preVecObsLoc[oppositeBoat,1,iObs] > 0
		and xLoc[currentBoat,1,iObs] - xLoc[oppositeBoat,1,iObs] > 0):# Rule 11 one down one upwind
		return -1
	elif (preVecObsLoc[currentBoat,1,iObs] > 0 and preVecObsLoc[oppositeBoat,1,iObs] < 0
		and xLoc[currentBoat,1,iObs] - xLoc[oppositeBoat,1,iObs] < 0):# Rule 11 one down one upwind
		return 1

	### RULE 11,12
	tmpReturn = overlapp(xLoc, currentBoat,oppositeBoat,iObs, preVecObsLoc)

	if (tmpReturn == 0): # overlapping rule 11
		xCenter = xLoc[currentBoat,:,iObs] - xLoc[oppositeBoat,:,iObs]
		if (preVecObsLoc[oppositeBoat,1,iObs] > 0 and preVecObsLoc[oppositeBoat,0,iObs] > 0):
			return xCenter[1]-xCenter[0] >= 0
		elif (preVecObsLoc[oppositeBoat,1,iObs] > 0 and preVecObsLoc[oppositeBoat,0,iObs] < 0):
			return xCenter[1]+xCenter[0] >= 0
		elif (preVecObsLoc[oppositeBoat,1,iObs] < 0 and preVecObsLoc[oppositeBoat,0,iObs] > 0):
			return xCenter[1]-xCenter[0] >= 0
		elif (preVecObsLoc[oppositeBoat,1,iObs] < 0 and preVecObsLoc[oppositeBoat,0,iObs] < 0):
			return xCenter[1]-xCenter[0] >= 0

	elif tmpReturn == - 1: #check if a boat inbetween overlapping
		if	max(xLoc[currentBoat,:,iObs] - xLoc[oppositeBoat,:,iObs])<= 1: # if next to each other no boat can overlapp
			return tmpReturn
		else:
			for boatIndex in range(amountBoats):
				if boatIndex != currentBoat and boatIndex != oppositeBoat:
					if (inbetween(xLoc, currentBoat,oppositeBoat,iObs, preVecObsLoc, boatIndex) == 1 and
					preVecObsLoc[oppositeBoat,0,iObs]*preVecObsLoc[boatIndex,0,iObs] >= 0 and
					overlapp(xLoc, boatIndex,oppositeBoat,iObs, preVecObsLoc) == 0 and
					overlapp(xLoc, currentBoat,boatIndex,iObs, preVecObsLoc) == 0): # if boat inbetween, overlapp, same tack, and inbetween boat and current boat overlapp, overlapp rules
						xCenter = xLoc[currentBoat,:,iObs] - xLoc[oppositeBoat,:,iObs]
						if (preVecObsLoc[oppositeBoat,1,iObs] > 0 and preVecObsLoc[oppositeBoat,0,iObs] > 0):
							return xCenter[1]-xCenter[0] >= 0
						elif (preVecObsLoc[oppositeBoat,1,iObs] > 0 and preVecObsLoc[oppositeBoat,0,iObs] < 0):
							return xCenter[1]+xCenter[0] >= 0
						elif (preVecObsLoc[oppositeBoat,1,iObs] < 0 and preVecObsLoc[oppositeBoat,0,iObs] > 0):
							return xCenter[1]-xCenter[0] >= 0
						elif (preVecObsLoc[oppositeBoat,1,iObs] < 0 and preVecObsLoc[oppositeBoat,0,iObs] < 0):
							return xCenter[1]-xCenter[0] >= 0
	return tmpReturn

def collidStart(xNewLoc, boat, iObs, xLoc, xs,preVecObsLoc): # check if the boats collid
	for i in range(amountBoats):
		if i != boat:
			if (xNewLoc[i,0]-xs[0])**2+(xNewLoc[i,1]-xs[1])**2 < 0.01:
				if prioretyFunction(xNewLoc, xLoc, boat, i, iObs,preVecObsLoc) <= 0:
					return 0
	return 1

def legalCrossing(xObsLoc,tTmp): # check if crossed the start line legally
	for i in range(Kobs+1):
		if tTmp+Kobs-i <= 25 and xObsLoc[1,Kobs-i] >= 20:
			return 0
		elif xObsLoc[1,Kobs-i] == 19:
			if xObsLoc[0,Kobs-i] <= 19 and xObsLoc[0,Kobs-i] >= 10:
				return 1
			else:
				return 0
		elif xObsLoc[1,Kobs-i] == 18:
			if xObsLoc[0,Kobs-i+1] >= 11 and xObsLoc[0,Kobs-i+1] <= 20 and xObsLoc[0,Kobs-i] >= 9 and xObsLoc[0,Kobs-i] <= 18:
				return 1
			elif xObsLoc[0,Kobs-i+1] >= 9 and xObsLoc[0,Kobs-i+1] <= 18 and xObsLoc[0,Kobs-i] >= 11 and xObsLoc[0,Kobs-i] <= 20:
				return 1
			else:
				return 0
	return 0

def nextStep(boat, iObs, xObsLoc, preVecObsLoc, xNewLoc, preVecNewLoc, minVecLoc, createIndicator, infiniteloop, tmpMinSortLoc, sIndexLoc, sObsLoc, wind,tTmp): # evaluates next viable steps for a boat
	minvalue = 10000000
	windSelf = WindMatrixself(xObsLoc[:,:,iObs+1],preVecObsLoc[:,:,iObs+1],boat)
	for k in range(sizeStart):
		xsx, xsy = vecStart[k,0], vecStart[k,1]
		distanceLoc = np.linalg.norm([xsx,xsy])
		vector1 = unit_vector(vecStart[vecStart[k,2],:2])
		degree = angle_between(vector1, preVecObsLoc[boat,:,iObs])

		if sObsLoc[boat, iObs] >= legalMoveShiftc2:
			Indextmp = degree < degree3 and distanceLoc != 0
		else:
			Indextmp = ((sObsLoc[boat, iObs] < legalMoveShiftc and (degree < degree1 or distance == 0) and distanceLoc <= 1.7) #(degree < degree1 or distanceLoc == 0)
				or (sObsLoc[boat, iObs] >= legalMoveShiftc and degree < degree2 and distanceLoc != 0 and distanceLoc <= 1.7))
		if Indextmp:
			sTmp = sObsLoc[boat, iObs].copy()*max((1-degree*tackSlowc-distanceSpeedc*(distanceLoc == 0)),0) + (maxSpeed-sObsLoc[boat, iObs])*speedIncreasec
			xs = [(xObsLoc[boat,0,iObs]+xsx),(xObsLoc[boat,1,iObs]+xsy)]

			if xs[0]<1 or xs[0] > 28 or xs[1]< 1 or xs[1] > 28: # if the boats go outside
				tmpMin = 10000
				if xs[0]<1:
					xs[0] = 1
				if xs[1]<1:
					xs[0] = 1
				if xs[0]>28:
					xs[0]= 28
				if xs[1]>28:
					xs[1]= 28

			elif T - tTmp - iObs - 1 == 0 and xs[1] >= 20:
				tmpMin = 10000
			elif T - tTmp - iObs - 1 < 0 and iObs+1 == Kobs and xs[1] >= 20 and legalCrossing(xObsLoc[boat,:,:],tTmp) == 0: # if the boats cross the start line ilegally
				tmpMin = 10000

			else:
				tmpMin = valuePosition (xs, degree, t, xsx, xsy, k, iObs, boat, sTmp, wind, windSelf)


			if infiniteloop > 5:
				tmpMin = tmpMin * np.abs(np.random.normal(1, 0.1*infiniteloop/5))
			if createIndicator == 1:
				tmpMinSortLoc[boat,k] = tmpMin
				speedWidth[k] = sTmp

			if tmpMin < minvalue and collidStart(xObsLoc[:,:,iObs+1],boat, iObs, xObsLoc[:,:,:], xs, preVecObsLoc): # determine which new step is best
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

def valuePosition (xPostion, degree, t, xsx, xsy, k, iObs, boat, s, wind, windSelf): # return the value of position traveld to
	wValue = avoidWindc*(wind[xs[0],xs[1]]-windSelf[xs[0],xs[1]])
	localStillc = 0
	if (T-(t+iObs)) > 10: # importance of speed change with time
		cs = 0
	elif (T-(t+iObs)) <= 5:
		cs = 1
		localStillc = stillc
	elif (T-(t+iObs)) <= 10:
		cs = 0.2
	standStilltmp = (stillc-localStillc)*(xsx == 0 and xsy == 0)
	return Z[xPostion[0],xPostion[1] ,T-(t+iObs)-1, vecStart[k,2],strat[boat]]*(2-(s-1)*cs+s* wValue - standStilltmp)

def potentialFuncStart0(fblr, stratIndex): # potential function at time 0
	zRet = np.zeros((30,30))
	init = 0

	if stratIndex == 0:

		for j in range(20): ## Middle
			for i in range(20-j):
				zRet[min(10+i+j,29),19-j] = init+jumpx*i+j*jumpy

				if fblr < 5:
					zRet[min(10+i+j,29),19-j] = zRet[min(10+i+j,29),19-j] + 0.4*(16-j)/15


		for j in range(20):# left
			for i in range(10+j):
				zRet[9+j-i,19-j] = j*jumpy + 5/(j+1) + i*jumpx

		for j in range(20): # right
			for i in range(10-j):
				zRet[20+j+i,19-j] += 2/(j+1)



	elif stratIndex == 1:
		for j in range(20): ## Middle
			for i in range(min(20-j,10)):

				zRet[10-min(j,10)+i,19-j] = 0.1+jumpx*i+j*jumpy

				if fblr > 1:
					zRet[10-min(j,10)+i,19-j] = zRet[10-min(j,10)+i,19-j] + 0.4*(16-j)/15


		for j in range(10): # left
			for i in range(10-j):
				zRet[9-i-j,19-j] = 2.0+jumpx*i+j*jumpy

		for j in range(20):# right
			for i in range(10+j):
				zRet[20-j+i,19-j] += 2/(j+1) + j*jumpy + (i+10)*jumpx


	elif stratIndex == 2 or stratIndex == 5:
		for j in range(20): ## Middle
			for i in range(15):

				zRet[14-i,19-j] = 0.1+jumpx*i+j*jumpy
				zRet[15+i,19-j] = 0.1+jumpx*i+j*jumpy

				if fblr < 5:
					zRet[14-i,19-j] = zRet[14-i,19-j] + 0.4*(16-j)/15
					zRet[15+i,19-j] = zRet[15+i,19-j] + 0.4*(16-j)/15

		for j in range(20): # left
			for i in range(10+j):
				zRet[9+j-i,19-j] += 2/(j+1)

		for j in range(20):#right
			for i in range(10-j):
				zRet[20+j+i,19-j] += 2/(j+1)


	elif stratIndex == 3:
		for j in range(20): ## Middle
			for i in range(15):

				zRet[14-i,19-j] = 0.1+jumpx*i+j*jumpy
				zRet[15+i,19-j] = 0.1+jumpx*i+j*jumpy

				if fblr > 1:
					zRet[14-i,19-j] = zRet[14-i,19-j] + 0.4*(16-j)/15
					zRet[15+i,19-j] = zRet[15+i,19-j] + 0.4*(16-j)/15

		for j in range(10): # left
			for i in range(10-j):
				zRet[9-i-j,19-j] +=  2/(j+1)

		for j in range(20): # right
			for i in range(10+j):
				zRet[20-j+i,19-j] += 2/(j+1)


	if stratIndex == 4:
		for j in range(20):
			for i in range(10): # Middle

				if 19-i+j <= 29:
					zRet[19-i+j,19-j] = 0.1+jumpx*i+j*jumpy
					if fblr < 5:
 						zRet[19-i+j,19-j] +=  0.4*(16-j)/15

		for j in range(20): # right
			for i in range(10):
				if 20+i+j <= 29:
					zRet[20+i+j,19-j] = 1+jumpx*i+j*jumpy

		for j in range(20): # left
			for i in range(10+j):
				zRet[9+j-i,19-j] += 2/(j+1) + j*jumpy + (i+10)*jumpx



	for j in range(10):
		for i in range(30):
			zRet[i,20+j] = 7+j*2

	return zRet

def potentialFuncStartPost(tPre,fblr,stratIndex): # potential function space, after start
	Z[:, :, -tPre-1, fblr, stratIndex] = Z[:, :, -tPre, fblr, stratIndex].copy()

	if stratIndex == 0 or stratIndex == 2 or stratIndex == 4 or stratIndex == 5:
		Z[(10-tPre-1):(19-tPre),19+tPre+1,-tPre-1,fblr,stratIndex] = Z[(10-tPre):(19-tPre+1),19+tPre,-tPre,fblr,stratIndex].copy() - jumpy

	elif stratIndex == 1 or stratIndex == 3:
		Z[(10+tPre+1):(19+tPre+2),19+tPre+1,-tPre-1,fblr,stratIndex] = Z[(10+tPre):(19+tPre+1),19+tPre,-tPre,fblr,stratIndex].copy() - jumpy

def potentialFuncStartpre(Zloc, fblr, t,stratIndex): # potential function space, before start. itterated backwards
	uniformMat = np.matrix([[1/8, 0, 1/8],
					[1/8, 1/8, 1/8],
					[1/8, 1/8, 1/8]])

	if fblr == 0:
		fMat = (1-uniformMovec)*np.matrix([[0.2, 0, 0.2],
						[0.15, 0.1, 0.15],
						[0.1, 0, 0.1]]) + uniformMovec*uniformMat

	elif fblr == 4:
		fMat = (1-uniformMovec)*np.matrix([[0.07 , 0, 0.07],
						[0.12, 0.1, 0.12],
						[0.2, 0.2, 0.2]]) + uniformMovec*uniformMat

	elif fblr == 1:
		fMat = (1-uniformMovec)*np.matrix([[0.25, 0, 0.3],
						[0.15, 0.1, 0.25],
						[0, 0.05, 0.15]])+ uniformMovec*uniformMat


	elif fblr == 7:
		fMat = (1-uniformMovec)*np.matrix([[0.3, 0, 0.25],
						[0.2, 0.1, 0.15],
						[0.15, 0.05, 0]])+ uniformMovec*uniformMat


	elif fblr == 3:
		fMat = (1-uniformMovec)*np.matrix([[0, 0, 0.15],
						[0.15, 0.1, 0.2],
						[0.2, 0.25, 0.25]])+ uniformMovec*uniformMat


	elif fblr == 5:
		fMat = (1-uniformMovec)*np.matrix([[0.15, 0, 0],
						[0.2, 0.1, 0.15],
						[0.25, 0.25, 0.2]])+ uniformMovec*uniformMat



	elif fblr == 6:
		fMat = (1-uniformMovec)*np.matrix([[0.1, 0, 1.5],
						[0, 0.1, 0.2],
						[0.1, 0.15, 0.15]])+ uniformMovec*uniformMat


	elif fblr == 2:
		fMat = (1-uniformMovec)*np.matrix([[0.15, 0, 0.1],
						[0.2, 0.1, 0],
						[0.15, 0.15, 0.1]])+ uniformMovec*uniformMat



	procentMat = np.matrix([[1, 0, 1],
					[1, 1, 1],
					[1, 1, 1]])


	procentMat = procentMat / np.sum(procentMat) * 8

	if stratIndex == 0: # sprint function
		sprintMat =np.matrix([[1, 0, 0],
		 					[0.3, 0, 0],
							[0, 0, 0]])
	elif stratIndex == 4 or stratIndex == 2:
		sprintMat =np.matrix([[1, 0, 0],
		 					[0.3, 0, 0],
							[0, 0, 0]])
	elif stratIndex == 3 or stratIndex == 1:
		sprintMat =np.matrix([[0, 0, 1],
		 					[0, 0, 0.3],
							[0, 0, 0]])
	else: # stratindex = 5
		sprintMat =np.matrix([[0, 0, 1],
		 					[0, 0, 0.3],
							[0, 0, 0]])
		sprintMatEnd =np.matrix([[1, 0, 0],
		 					[0.3, 0, 0],
							[0, 0, 0]])
		sprintMatEnd = (sprintMatEnd + fMat)/(np.sum(fMat+sprintMatEnd))
	sprintMat = (sprintMat + fMat)/(np.sum(fMat+sprintMat))

	fMat = fMat/np.sum(fMat)

	fAngel = np.matrix([[7, 0, 1],
		[6, 8, 2],
		[5, 4, 3]])

	zRet = np.zeros((30,30))
	for j in range(1,29): # itterating potential function backwards
		for i in range(1,29):
			tmp = 0
			for k in range(3):
				for l in range(3):
					if stratIndex != 5:
						if t < tsprintc:
							if j >= 0 and j <= 29 and i == 19:
								if fAngel[l,k] == 8:
									for centerIndex in range(8):
										tmp = tmp + Zloc[j,i,centerIndex]*(sprintMat[l,k])/8*procentMat[l,k]
								else:
									tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(sprintMat[l,k])*procentMat[l,k]
							else:
								if fAngel[l,k] == 8:
									for centerIndex in range(8):
										tmp = tmp + Zloc[j,i,centerIndex]*(sprintMat[l,k])/8
								else:
									tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(sprintMat[l,k])
						else:
							if fAngel[l,k] == 8:
								for centerIndex in range(8):
									tmp = tmp + Zloc[j,i,centerIndex]*fMat[l,k]/8
							else:
								tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*fMat[l,k]
					else: # stratIndex 5
						if t < 3:
							if j >= 0 and j <= 29 and i == 19:
								if fAngel[l,k] == 8:
									for centerIndex in range(8):
										tmp = tmp + Zloc[j,i,centerIndex]*(sprintMatEnd[l,k])/8*procentMat[l,k]
								else:
									tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(sprintMatEnd[l,k])*procentMat[l,k]
							else:
								if fAngel[l,k] == 8:
									for centerIndex in range(8):
										tmp = tmp + Zloc[j,i,centerIndex]*(sprintMatEnd[l,k])/8
								else:
									tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(sprintMatEnd[l,k])


						elif t < tsprintc:
							if j >= 0 and j <= 29 and i == 19:
								if fAngel[l,k] == 8:
									for centerIndex in range(8):
										tmp = tmp + Zloc[j,i,centerIndex]*(sprintMat[l,k])*procentMat[l,k]
								else:
									tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(sprintMat[l,k])*procentMat[l,k]
							else:
								if fAngel[l,k] == 8:
									for centerIndex in range(8):
										tmp = tmp + Zloc[j,i,centerIndex]*(sprintMat[l,k])/8
								else:
									tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*(sprintMat[l,k])

						else:
							if fAngel[l,k] == 8:
								for centerIndex in range(8):
									tmp = tmp + Zloc[j,i,centerIndex]*fMat[l,k]/8
							else:
								tmp = tmp + Zloc[j-1+k,i+1-l,fAngel[l,k]]*fMat[l,k]
			zRet[j,i] += tmp
	zRet[:,0]=zRet[:,1] # make the edges the same as one row/colomn inwards
	zRet[:,29]=zRet[:,28]
	zRet[0,:]=zRet[1,:]
	zRet[29,:]=zRet[28,:]
	return zRet.copy()

def nextStepSecond(boat, PositionNext, iObs, xObsLoc,preVecObsLoc,xNewLoc,preVecNewLoc, tmpMinSortLoc,sIndexLoc, sObs): # Finding second/third ... best step
	sortMinVec = np.argsort(tmpMinSortLoc)
	index = 0
	while sortMinVec[index] != minSortK[boat, sIndexLoc]:
		index += 1
	xs = xObsLoc[boat,:,iObs].copy
	if PositionNext > 0:
		for i in range(PositionNext):
			if index == 14:
				break
			index += 1
			xsx, xsy = vecStart[sortMinVec[index],0], vecStart[sortMinVec[index],1]
			xs = [(xObsLoc[boat,0,iObs]+xsx),(xObsLoc[boat,1,iObs]+xsy)]
			if xs[0]<1 or xs[0] > 28 or xs[1]< 1 or xs[1] > 28: # if the boats go outside
				tmpMin = 1000
				if xs[0]<1:
					xs[0] = 1
				if xs[1]<1:
					xs[0] = 1
				if xs[0]>28:
					xs[0]= 28
				if xs[1]>28:
					xs[1]= 28

			while collidStart(xObsLoc[:,:,iObs+1], boat, iObs, xObsLoc[:,:,:], xs, preVecObsLoc) <= 0 and index < 13:
				index += 1
				xsx, xsy = vecStart[sortMinVec[index],0], vecStart[sortMinVec[index],1]
				xs = [(xObsLoc[boat,0,iObs]+xsx),(xObsLoc[boat,1,iObs]+xsy)]
				if xs[0]<1 or xs[0] > 28 or xs[1]< 1 or xs[1] > 28: # if the boats go outside
					tmpMin = 1000
					if xs[0]<1:
						xs[0] = 1
					if xs[1]<1:
						xs[0] = 1
					if xs[0]>28:
						xs[0]= 28
					if xs[1]>28:
						xs[1]= 28
	else:
		xsx, xsy = vecStart[sortMinVec[index],0], vecStart[sortMinVec[index],1]
		xs = [(xObsLoc[boat,0,iObs]+xsx),(xObsLoc[boat,1,iObs]+xsy)]
		if xs[0]<1 or xs[0] > 28 or xs[1]< 1 or xs[1] > 28: # if the boats go outside
			tmpMin = 1000
			if xs[0]<1:
				xs[0] = 1
			if xs[1]<1:
				xs[0] = 1
			if xs[0]>28:
				xs[0]= 28
			if xs[1]>28:
				xs[1]= 28
	xNewLoc[boat,:] = xs.copy()
	preVecNewLoc[boat,:] =  unit_vector([vecStart[vecStart[sortMinVec[index],2],0], vecStart[vecStart[sortMinVec[index],2],1]])
	sObs[boat,iObs,sIndexLoc] = speedWidth[sortMinVec[index]]

def recurrsion(crash, xObsLoc, preVecObsLoc, sIndex, xNewLoc, preVecNewLoc, boat, iObs, collidVecLoc, minVecLoc, tmpMinSortLoc, sObsLoc, tTmp): # recurrsion to find the paths the boat can take
	if crash == 0 and iObs < Kobs: # make sure a crash hasn't happend
		for i in range(Kn): # observation before finding next position
			wind = WindMatrix(xObs[:,:,iObs+1,sIndex],preVecObs[:,:,iObs+1,sIndex])
			for j in range(amountBoats): # finding best next step
				if collidVecLoc[j] == 1:
					if j == boat:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 1, 0, tmpMinSortLoc[:,:,sIndex], sIndex, sObsLoc[:,:,sIndex], wind,tTmp)
					else:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 0, 0, tmpMinSortLoc[:,:,sIndex], sIndex, sObsLoc[:,:,sIndex], wind,tTmp)
				else:
					xNewLoc[j,:] = xObsLoc[j,:,iObs+1,0].copy()
					preVecNewLoc[j,:] = preVecObsLoc[j,:,iObs+1,0].copy()
			xObsLoc[:,:,iObs+1,sIndex] = xNewLoc.copy()
			preVecObsLoc[:,:,iObs+1,sIndex] = preVecNewLoc.copy()


		for tmpIndex in range(1, Kobsvec[iObs]):
			collidVecNew = collidVecLoc.copy()
			nextStepSecond(boat, tmpIndex, iObs, xObsLoc[:,:,:,sIndex], preVecObsLoc[:,:,:,sIndex], xNewLoc, preVecNewLoc, tmpMinSortLoc[boat,:,sIndex], sIndex, sObsLoc)
			sIndextmp = sWidthNew[iObs,sIndex] + tmpIndex
			xObsLoc[:,:,:iObs+1,sIndextmp] = xObsLoc[:,:,:iObs+1,sIndex].copy()
			preVecObsLoc[:,:,:iObs+1,sIndextmp] = preVecObsLoc[:,:,:iObs+1,sIndex].copy()
			xObsLoc[:,:,iObs+1,sIndextmp] = xNewLoc.copy()
			preVecObsLoc[:,:,iObs+1,sIndextmp] = preVecNewLoc.copy()

			sObsLoc[:,:iObs+1,sIndextmp] = sObsLoc[:,:iObs+1,sIndex].copy()

			crashtmp = crash
			collidVar = 0
			infiniteloop = 0
			while collidVar == 0:
				wind = WindMatrix(xObs[:,:,iObs+1,sIndextmp],preVecObs[:,:,iObs+1,sIndextmp])
				infiniteloop = infiniteloop + 1
				collidVar = 1
				for j in range(amountBoats):
					if collidStart(xObsLoc[:,:,iObs+1,sIndextmp],j, iObs, xObsLoc[:,:,:,sIndextmp], xObsLoc[j,:,iObs+1,sIndextmp], preVecObsLoc[:,:,:,sIndextmp]) <= 0: # if the boat collid
						if j == boat:
							nextStep(j, iObs,xObsLoc[:,:,:,sIndextmp],preVecObsLoc[:,:,:,sIndextmp],xNewLoc,preVecNewLoc, minVecLoc[:,sIndextmp], 1, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind,tTmp)
						else:
							nextStep(j, iObs,xObsLoc[:,:,:,sIndextmp],preVecObsLoc[:,:,:,sIndextmp],xNewLoc,preVecNewLoc, minVecLoc[:,sIndextmp], 0, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind,tTmp)
							collidVecNew[j] = 1
				xObsLoc[:,:,iObs+1,sIndextmp] = xNewLoc.copy()
				preVecObsLoc[:,:,iObs+1,sIndextmp] = preVecNewLoc.copy()
				for j in range(amountBoats):
					collidVar = collidStart(xObsLoc[:,:,iObs+1,sIndextmp],j, iObs, xObsLoc[:,:,:,sIndextmp], xObsLoc[j,:,iObs+1,sIndextmp], preVecObsLoc[:,:,:,sIndextmp])*collidVar # keeps being 1 if they don't collid

				if infiniteloop >= 10:
					minVecLoc[boat,sIndextmp] = 10000
					collidVar = 1
					crashtmp = 1

			wind = WindMatrix(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp])
			for i in range(amountBoats):
				if collidVecLoc[i] == 1:
					windSelf = WindMatrixself(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp],i)
					sObsLoc[i,iObs+1,sIndextmp] = sObs[i,iObs+1,sIndextmp]*(1-(wind[xObsLoc[i,0,iObs+1,sIndextmp],xObs[i,1,iObs+1,sIndextmp]]-windSelf[xObs[i,0,iObs+1,sIndextmp],xObs[i,1,iObs+1,sIndextmp]])/4)


			recurrsion(crashtmp, xObsLoc, preVecObsLoc, sIndextmp, xNewLoc, preVecNewLoc, boat, iObs + 1, collidVecNew, minVecLoc, tmpMinSortLoc, sObsLoc, tTmp)

		xNewLoc = xObsLoc[:,:,iObs+1,sIndex].copy()
		preVecNewLoc = preVecObsLoc[:,:,iObs+1,sIndex].copy()
		collidVecNew = collidVecLoc
		sIndextmp = sIndex
		collidVar = 0
		infiniteloop = 0
		while collidVar == 0:
			wind = WindMatrix(xObs[:,:,iObs+1,sIndextmp],preVecObs[:,:,iObs+1,sIndextmp])
			infiniteloop = infiniteloop + 1
			collidVar = 1
			for j in range(amountBoats):
				if collidStart(xObsLoc[:,:,iObs+1,sIndex],j,iObs, xObsLoc[:,:,:,sIndex], xObsLoc[j,:,iObs+1,sIndex], preVecObsLoc[:,:,:,sIndextmp]) <= 0: # if the boat collid
					if j == boat:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 1, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind , tTmp)
					else:
						nextStep(j, iObs,xObsLoc[:,:,:,sIndex],preVecObsLoc[:,:,:,sIndex],xNewLoc,preVecNewLoc, minVecLoc[:,sIndex], 0, infiniteloop, tmpMinSortLoc[:,:,sIndextmp], sIndextmp, sObsLoc[:,:,sIndextmp],wind , tTmp)
						collidVecNew[j] = 1
			xObsLoc[:,:,iObs+1,sIndex] = xNewLoc.copy()
			preVecObsLoc[:,:,iObs+1,sIndex] = preVecNewLoc.copy()
			for j in range(amountBoats):
				collidVar = collidStart(xObsLoc[:,:,iObs+1,sIndex],j,iObs, xObsLoc[:,:,:,sIndex], xObsLoc[j,:,iObs+1,sIndex], preVecObsLoc[:,:,:,sIndextmp])*collidVar # keeps being 1 if they don't collid

			if infiniteloop >= 10:
				minVecLoc[boat,sIndex] = 10000
				collidVar = 1
				crash = 1

		wind = WindMatrix(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp])
		for i in range(amountBoats):
			if collidVecLoc[i] == 1:
				windSelf = WindMatrixself(xObsLoc[:,:,iObs+1,sIndextmp],preVecObsLoc[:,:,iObs+1,sIndextmp],i)
				sObsLoc[i,iObs+1,sIndextmp] = sObs[i,iObs+1,sIndextmp]*(1-(wind[xObsLoc[i,0,iObs+1,sIndextmp],xObs[i,1,iObs+1,sIndextmp]]-windSelf[xObs[i,0,iObs+1,sIndextmp],xObs[i,1,iObs+1,sIndextmp]])*windSlowc)

		recurrsion(crash, xObsLoc, preVecObsLoc, sIndextmp, xNewLoc, preVecNewLoc, boat, iObs + 1, collidVecNew, minVecLoc, tmpMinSortLoc, sObsLoc,tTmp)

def branchweight(rot, futureBranchIndex, branchIndex, minVecLoc, minVecIndextmp, minValueTmp, boat): # the value of a path
	futureBranchIndex += 1
	if futureBranchIndex == Kobsvec.shape[0]-1-math.floor(riskc):
		minVecLoc[rot] += minVec[boat, branchIndex]

		if minVec[boat, branchIndex] < minValueTmp:
			minVecIndextmp[rot] = branchIndex
			minValueTmp = minVec[boat, branchIndex]
			return minVec[boat, branchIndex]
		else:
			return 0

	else:
		for addBranchIndex in range(Kobsvec[math.floor(riskc)+futureBranchIndex]):
			tmpValue = branchweight(rot, futureBranchIndex, sWidthNew[math.floor(riskc),branchIndex]+addBranchIndex+1, minVecLoc, minVecIndextmp, minValueTmp, boat)
			if tmpValue != 0:
				minValueTmp = tmpValue
		tmpValue = branchweight(rot, futureBranchIndex, branchIndex, minVecLoc, minVecIndextmp, minValueTmp, boat)

###### ###### ###### ###### ######
###### Post start functions ######
###### ###### ###### ###### ######

def TVector(): # Base travel vecotr after start
	T = np.zeros((18,4))

	T[0,0] = -10
	T[0,1] = 10
	T[1,0] = 10
	T[1,1] = 10
	T[2,0] = -14
	T[2,1] = 0
	T[5,0] = 14
	T[5,1] = 0
	T[3,0] = -8
	T[3,1] = 5
	T[4,0] = 8
	T[4,1] = 5
	T[6,0] = -10
	T[6,1] = -5
	T[7,0] = 10
	T[7,1] = -5
	T[8,0] = 0
	T[8,1] = -10


	T[9,0] = -10
	T[9,1] = 10
	T[10,0] = 10
	T[10,1] = 10
	T[11,0] = -14
	T[11,1] = 0
	T[14,0] = 14
	T[14,1] = 0
	T[12,0] = -12
	T[12,1] = 5
	T[13,0] = 12
	T[13,1] = 5
	T[15,0] = -10
	T[15,1] = -5
	T[16,0] = 10
	T[16,1] = -5
	T[17,0] = 0
	T[17,1] = -10

	for i in range(9,18):
		T[i,0] = T[i,0]/2
		T[i,1] = T[i,1]/2
	for i in range(T.shape[0]):
		T[i,2] = T[i,0]/np.sqrt(T[i,0]**2+T[i,1]**2)
		T[i,3] = T[i,1]/np.sqrt(T[i,0]**2+T[i,1]**2)
	T[:,0] = T[:,0]*4
	T[:,1] = T[:,1]*4
	return T

def potentialFunc(): #potential function after start
	zRet = np.zeros((yMesh,xMesh))
	shiftx = 1

	for i in range(xMesh): # potential is equal to distance from bouj placed at (0,10) + small distance to the right
		for j in range(yMesh):
			zRet[j,i] = np.sqrt((xMin-shiftx+i*xh)**2+(10-yMin-j*yh)**2)+5



	yrange = int((10-yMin)/xh)

	for i in range(yrange):
		distance = np.sqrt(2)*i*yh + 5
		for j in range(max(-i,-1000 - int(shiftx*100)),min(i+1,1000-int(shiftx*100))):
			zRet[1500-i,1000 + int(shiftx*100) + j] = distance


	for i in range(yrange):
		for j in range(1000+shiftx*100-i):
			zRet[1500-i,j] += (5-j/200)+2


	for i in range(1501,2001):
		for j in range(1000):
			zRet[i,1000-j] = -5-j/100 + np.abs((1600-i)/500)


	for i in range(1501,2001):
		for j in range(1000):
			zRet[i,1001+j] = -3+j*3/1000 + np.abs((1600-i)/500)



	for i in range(900):
		for j in range(700-i):
			zRet[749-i,1101+j+i] += j*5/750+2
			zRet[749-i,899-j-i] += j*5/750+2




	return zRet

def stratChange(fbout,strategy): # evolutionary changing the stratergies
	meanVec = fbout.mean(1, dtype=np.float)
	meanVecStrat = np.zeros((6,2))
	indexMinTmp = strat[0]
	indexMaxTmp = strat[0]
	indexMin = 0
	indexMax = 0
	for i in range(amountBoats):
		meanVecStrat[strat[i],0] += meanVec[i]
		meanVecStrat[strat[i],1] += 1
	for i in range(6):
		if meanVecStrat[i,1] > 0:
			meanVecStrat[i,0] = meanVecStrat[i,0]/(meanVecStrat[i,1])
	for i in range(6):
		if meanVecStrat[i,1] > 0:
			if meanVecStrat[i,0] > meanVecStrat[indexMaxTmp,0]:
				indexMaxTmp = i
			if meanVecStrat[i,0] < meanVecStrat[indexMinTmp,0]:
				indexMinTmp = i
	for i in range(amountBoats):
		if strat[i] == indexMinTmp:
			indexMin = i
		if strat[i] == indexMaxTmp:
			indexMax = i

	if int(min(meanVecStrat[:,1])) == 0:
		r = np.random.uniform(0,1,1)
		limit = 0.75
		if r > limit:
			ZeroSortVec = np.argsort(meanVecStrat[:,1])
			amountDeadStart = 0
			while meanVecStrat[ZeroSortVec[amountDeadStart],1] == 0:
				amountDeadStart += 1
			strategy[indexMax] = ZeroSortVec[np.random.randint(amountDeadStart, size = 1)]
		else:
			strategy[indexMax] = strat[indexMin]
	else:
		strategy[indexMax] = strat[indexMin]
	return meanVecStrat[:,0]

def WindPolly(vVector,travelPosition, xmesh,ymesh, xh): # wind base function depending on travel direction
	windRet = np.zeros((201,201), dtype = np.float) # wind mesh
	wideC, lengthC =1.5, 2.5


	for j in range(100):
		windRet[101-j,101] = ((lengthC*20)>j)
		for k in range(-99,0):
			if (lengthC*(wideC*k+20)>j):
				windRet[101-j,101+k] = 1
				windRet[101-j,101-k] = 1

	if travelPosition%9 < 2:
		shiftC = 0.1
	elif travelPosition%9>1 and travelPosition%9<4:
	 	shiftC=0.05
	elif travelPosition%9>3 and travelPosition%9<6:
		shiftC=0
	else:
		shiftC=1.2

	if travelPosition%9 != 8:
		shift = math.pi*(0.5+vVector[0]*shiftC)
		shiftMat=[[math.sin(shift),math.cos(shift)],[-math.cos(shift), math.sin(shift)]]
		rot = np.array([[vVector[1], -vVector[0]],[vVector[0],vVector[1]]])
		rot = np.matmul(rot,shiftMat)

		shadowSize = 25
		for j in range(90):
			xStep = np.int_(np.round_(np.matmul(rot,[0,j])))
			if windRet[101-xStep[1],101+xStep[0]] == 0 and (lengthC*shadowSize>j):
				windRet[101-xStep[1],101+xStep[0]] = 0.6

			for k in range(1,25):
				xStep = np.int_(np.round_(np.matmul(rot,[k,j])))
				xStep2 = np.int_(np.round_(np.matmul(rot,[-k,j])))
				if (-7*(k-j/3)+lengthC*(shadowSize)>j):
					for tmpi in range(-1,2):
						for tmpj in range(-1,2):
							if windRet[101-xStep[1]+tmpi,101+xStep[0]+tmpj] == 0:
								windRet[101-xStep[1]+tmpi,101+xStep[0]+tmpj] = 0.6
							if windRet[101-xStep2[1]+tmpi,101+xStep2[0]+tmpj] == 0:
								windRet[101-xStep2[1]+tmpi,101+xStep2[0]+tmpj] = 0.6



	for i in range(-20,21):
		for j in range(-20,21):
			if (np.sqrt(i*i+j*j)<= 12):
				windRet[101+i,101+j] = 5


	return windRet

def sWidthMake(KobsvecLoc,swidth):
	sWidthReturn = np.zeros((KobsvecLoc.shape[0],sWidth), dtype = np.int)
	currentbranches = 1
	for i in range(KobsvecLoc.shape[0]):
		for j in range(currentbranches):
			sWidthReturn[i,j] = (currentbranches-1)
			currentbranches += (KobsvecLoc[i]-1)

	return sWidthReturn

#start = time.time() # time to finish

Tpost = TVector()

if save2 == 1:
	Zpost = potentialFunc()
	np.save('potentialPost.npy', Zpost)
else:
	Zpost = np.load('potentialPost.npy')

if save3 == 1:
	windBase = np.zeros((201,201,Tpost.shape[0]))
	for i in range(Tpost.shape[0]):
		windBase[:,:,i] = WindPolly(Tpost[i,2:4],i, xMesh,yMesh, xh)
	np.save('windBase.npy', windBase)
else:
	windBase = np.load('windBase.npy')

xs, x0, xNew = np.zeros(2, dtype=np.int), np.zeros((amountBoats,2), dtype=np.int), np.zeros((amountBoats,2), dtype=np.int) # update step, value of update, start position
xObs = np.zeros((amountBoats,2,KobsMax+1, sWidthMax), dtype=np.int) # observation x

xNextFinal = np.ones((amountBoats,2,KobsMax+1), dtype=np.int)
preVecFinal = np.ones((amountBoats,2,KobsMax+1))
sVecFinal = np.ones((amountBoats,KobsMax+1))

###### START
yMeshstart, xMeshstart = int(yMesh/40) + 1 ,int(xMesh/10) + 1
xhStart, yhStart = xh*10, yh*10

vecStart , sizeStart = TVectorStart()
speedWidth = np.zeros(sizeStart)
preVec, preVecNew, preVecObs = np.zeros((amountBoats,2)), np.zeros((amountBoats,2)), np.zeros((amountBoats,2,KobsMax+1,sWidthMax))
xPath = np.zeros((amountBoats,2,T+1))
xPathVec = np.zeros((amountBoats,2,T+1))
tmpMinSort = np.ones((amountBoats, sizeStart, sWidthMax))
minSortK = np.ones((amountBoats,sWidthMax), dtype=np.int)
Z = np.zeros((30,30,T+3,8,stratAmount)) # potential function pre start

colorVec = []
shiftConstant = 100
#colorVec.append('#%02x%02x%02x' % (125, 0, 125)) # purple
#colorVec.append('#%02x%02x%02x' % (255, 255, 0)) # yellow
#colorVec.append('#%02x%02x%02x' % (0, 0, 255)) # blue
#colorVec.append('#%02x%02x%02x' % (0, 255, 0)) # green
#colorVec.append('#%02x%02x%02x' % (255, 0, 0)) # red

for i in range(amountBoats): # colors for boats
	if strat[i] == 0:
		colorVec.append('#%02x%02x%02x' % (255 - random.randrange(shiftConstant), random.randrange(shiftConstant), random.randrange(shiftConstant))) # red
	elif strat[i] == 1:
		colorVec.append('#%02x%02x%02x' % (255- random.randrange(shiftConstant), 255- random.randrange(shiftConstant), random.randrange(shiftConstant))) # yellow
	elif strat[i] == 2 or strat[i] == 5:
		colorVec.append('#%02x%02x%02x' % (random.randrange(shiftConstant), 255 -  random.randrange(shiftConstant),  random.randrange(shiftConstant))) # green
	elif strat[i] == 3:
		colorVec.append('#%02x%02x%02x' % (random.randrange(shiftConstant), random.randrange(shiftConstant), 255 - random.randrange(shiftConstant))) # blue
	elif strat[i] == 4:
		colorVec.append('#%02x%02x%02x' % (125-int(shiftConstant/2) + random.randrange(shiftConstant), random.randrange(shiftConstant), 125-int(shiftConstant/2)+random.randrange(shiftConstant))) # purple


s0 = np.ones(amountBoats)
sObs = np.ones((amountBoats,KobsMax+1, sWidthMax))

if save == 1:
	for stratIndex in range(0,stratAmount):
		for i in range(8):
			Z[:,:,0,i,stratIndex] = potentialFuncStart0(i,stratIndex)

	for tPre in range(0,4):
		for stratIndex in range(0,stratAmount):
			for i in range(8):
				potentialFuncStartPost(tPre, i, stratIndex)


	for stratIndex in range(0,stratAmount):
		for t in range(T-1):
			for tvec in range(8):
				Z[:,:,t+1,tvec,stratIndex] = potentialFuncStartpre(Z[:,:,t,:,stratIndex],tvec,t,stratIndex)



	np.save('potential.npy', Z)

else:
	Z = np.load('potential.npy')

#################### #################### #################### #################### ####################
#################### #################### #################### #################### ####################
#################### #################### #################### #################### ####################

for evolutionStepIndex in range(evolutionStep):
	sortPosition = np.argsort(strat)
	print(strat)
	for evolutionIndex in range(kEvolution):
		initialAngel = np.random.uniform(low = 0.0, high = math.pi/2, size = amountBoats) # initial angel of the boats

		for i in range(amountBoats): # initial position of the boats
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

		for t in range(T):
			print("t", t)
			if T-t == T: # UPDATE CONSTANTS WITH TIME
				Kobsvec = np.array([3,1,1])

				Kobs = Kobsvec.shape[0]
				sWidth = 1
				for i in range(Kobs):
					sWidth = sWidth*Kobsvec[i]

				sWidthNew = sWidthMake(Kobsvec, sWidth)
				minVec = np.zeros((amountBoats,sWidth))
			elif T-t == 7:
				Kobsvec = np.array([4,3,2,1])

				Kobs = Kobsvec.shape[0]
				sWidth = 1
				for i in range(Kobs):
					sWidth = sWidth*Kobsvec[i]
				sWidthNew = sWidthMake(Kobsvec, sWidth)
				minVec = np.zeros((amountBoats,sWidth))


			for tmpIndex in range(sWidth):
				sObs[:,0,tmpIndex] = s0.copy()
				xObs[:,:,0,tmpIndex] = x0.copy()
				xObs[:,:,-1,tmpIndex] = xPath[:,:,t-1].copy()
				preVecObs[:,:,0,tmpIndex] = preVec.copy()
				preVecObs[:,:,-1,tmpIndex] = xPathVec[:,:,t-1].copy()

			for iObs in range(Kobs): # observation rounds, greedy paths for the boats
				wind = WindMatrix(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0])
				for i in range(Kn): # observation before finding next position
					for j in range(amountBoats): # finding best next step
						nextStep(j, iObs,xObs[:,:,:,0],preVecObs[:,:,:,0],xNew,preVecNew ,minVec[:,0], iObs+1, 0, tmpMinSort[:,:,0], 0, sObs[:,:,0],wind, t)
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
						if collidStart(xObs[:,:,iObs+1,0], j, iObs, xObs[:,:,:,0], xObs[j,:,iObs+1,0],preVecObs[:,:,:,0]) <= 0: # if the boat collid
							nextStep(j,iObs,xObs[:,:,:,0],preVecObs[:,:,:,0],xNew,preVecNew, minVec[:,0], iObs+1, infiniteloop, tmpMinSort[:,:,0], 0, sObs[:,:,0], wind, t)

					xObs[:,:,iObs+1,0] = xNew.copy()
					preVecObs[:,:,iObs+1,0] = preVecNew.copy()
					for j in range(amountBoats):
						collidVar = collidStart(xObs[:,:,iObs+1,0], j, iObs, xObs[:,:,:,0], xObs[j,:,iObs+1,0],preVecObs[:,:,:,0])*collidVar # keeps being 1 if they don't collid
					if infiniteloop == 10:
						collidVar = 1

				wind = WindMatrix(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0])
				for i in range(amountBoats):
					windSelf = WindMatrixself(xObs[:,:,iObs+1,0],preVecObs[:,:,iObs+1,0],i)
					sObs[i,iObs+1,0] = sObs[i,iObs+1,0]*(1-(wind[xObs[i,0,iObs+1,0],xObs[i,1,iObs+1,0]]-windSelf[xObs[i,0,iObs+1,0],xObs[i,1,iObs+1,0]])*windSlowc)

		########################################## ########################################## ##########################################
		########################################## ########################################## ##########################################
		########################################## Step 2

			for iObsMax in range(KnPath):
				for boat in range(amountBoats):
					collidVec = np.zeros(amountBoats)
					collidVec[boat] = 1
					xNew = xObs[:,:,0,0].copy()
					preVecNew = preVecObs[:,:,0,0].copy()

					recurrsion(0, xObs, preVecObs, 0, xNew, preVecNew, boat, 0, collidVec, minVec, tmpMinSort, sObs,t)
					if iObsMax + 1 == KnPath:

						########################################## ########################################## ########################################## Finding new path
						########################################## ########################################## ##########################################

						riskMaxLoc = min(math.floor(riskc),Kobsvec.shape[0]-1)

						if Kobsvec[riskMaxLoc] == 1 or riskMaxLoc == Kobsvec.shape[0]-1:
							powerVec = 1/np.power(minVec[boat,:]-np.min(minVec[boat,:])+1,chancec)
							vectorSum = np.sum(np.absolute(powerVec))
							vectorCumSum = np.cumsum(powerVec/vectorSum)
							u = random.uniform(0, 1)
							nextPositionindex = 0
							while vectorCumSum[nextPositionindex] <= u:
								nextPositionindex += 1

							xNextFinal[boat,:,:] = xObs[boat,:,:,nextPositionindex].copy()
							preVecFinal[boat,:,:] = preVecObs[boat,:,:,nextPositionindex].copy()
							sVecFinal[boat,:] = sObs[boat,:,nextPositionindex].copy()
						else:

							amountBranches = 1 # amount of branches
							for branchIndex in range(riskMaxLoc):
								amountBranches = amountBranches*Kobsvec[branchIndex]

							minVecTmp = np.zeros(amountBranches)
							minVecIndextmp =   np.ones(amountBranches, dtype = np.int)

							for amountBranchesIndex in range(amountBranches):
								branchweight(amountBranchesIndex, 0, amountBranchesIndex, minVecTmp, minVecIndextmp, 10000, boat)

							powerVec = 1/np.power(minVecTmp-np.min(minVecTmp)+1,chancec)
							vectorSum = np.sum(np.absolute(powerVec))
							vectorCumSum = np.cumsum(powerVec/vectorSum)
							u = random.uniform(0, 1)
							nextPositionindex = 0
							while vectorCumSum[nextPositionindex] <= u:
								nextPositionindex += 1

							xNextFinal[boat,:,:] = xObs[boat,:,:,minVecIndextmp[nextPositionindex]].copy()
							preVecFinal[boat,:,:] = preVecObs[boat,:,:,minVecIndextmp[nextPositionindex]].copy()
							sVecFinal[boat,:] = sObs[boat,:,minVecIndextmp[nextPositionindex]].copy()


					########################################## ########################################## ##########################################
					########################################## ########################################## ##########################################
					else:
						riskMaxLoc = min(math.floor(riskc),Kobsvec.shape[0]-1)

						if Kobsvec[riskMaxLoc] == 1 or riskMaxLoc == Kobsvec.shape[0]-1:
							minimumArgument = np.argsort(minVec[boat,:])
							xNextFinal[boat,:,:] = xObs[boat,:,:,minimumArgument[0]].copy()
							preVecFinal[boat,:,:] = preVecObs[boat,:,:,minimumArgument[0]].copy()
							sVecFinal[boat,:] = sObs[boat,:,minimumArgument[0]].copy()

						else:
							amountBranches = 1 # amount of branches
							for branchIndex in range(riskMaxLoc):
								amountBranches = amountBranches*Kobsvec[branchIndex]

							minVecTmp = np.zeros(amountBranches)
							minVecIndextmp =   np.ones(amountBranches, dtype = np.int)

							for amountBranchesIndex in range(amountBranches):
								branchweight(amountBranchesIndex, 0, amountBranchesIndex, minVecTmp, minVecIndextmp, 10000, boat)

							minimumArgument = np.argsort(minVecTmp)
							xNextFinal[boat,:,:] = xObs[boat,:,:,minVecIndextmp[minimumArgument[0]]].copy()
							preVecFinal[boat,:,:] = preVecObs[boat,:,:,minVecIndextmp[minimumArgument[0]]].copy()
							sVecFinal[boat,:] = sObs[boat,:,minVecIndextmp[minimumArgument[0]]].copy()




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
					if collidStart(xNextFinal[:,:,1], j, 0, xNextFinal[:,:,:], xNextFinal[j,:,1],preVecFinal[:,:,:]) <= 0: # if the boat collid
						collidVec[j] == 1
						nextStep(j,0,xNextFinal,preVecFinal,xNew,preVecNew, minVec[:,0], 0, infiniteloop, tmpMinSort[:,:,0], 0, sObs[:,:,0], wind, t)
				xNextFinal[:,:,1] = xNew.copy()
				preVecFinal[:,:,1] = preVecNew.copy()
				for j in range(amountBoats):
					collidVar = collidStart(xNextFinal[:,:,1], j, 0, xNextFinal[:,:,:], xNextFinal[j,:,1],preVecFinal[:,:,:])*collidVar # keeps being 1 if they don't collid

				if infiniteloop == 10:
					collidVar = 1

			wind = WindMatrix(xNextFinal[:,:,1],preVecFinal[:,:,1])
			for i in range(amountBoats):
				if collidVec[i] == 1:
					windSelf = WindMatrixself(xNextFinal[:,:,1],preVecFinal[:,:,1],i)
					sObs[i,1,0] = sObs[i,1,0]*(1-(wind[xNextFinal[i,0,1],xNextFinal[i,1,1]]-windSelf[xNextFinal[i,0,1],xNextFinal[i,1,1]])*windSlowc)

			x0 = xNextFinal[:,:,1].copy()
			preVec = preVecFinal[:,:,1].copy()

			xPath[:,:,t+1] = x0.copy()
			xPathVec[:,:,t+1] = preVec.copy()
			s0 = sObs[:,1,0].copy()
		########################################## ########################################## ##########################################
		########################################## ########################################## ##########################################
		###### end


		print("xNextFinal", xNextFinal)
		print("preVecFinal", preVecFinal)

		if os.path.isdir('./imagesgif/' + str (evolutionStepIndex) + 'E' + str (evolutionIndex) + 'E') == 0:
			os.mkdir('./imagesgif/' + str (evolutionStepIndex) + 'E' + str (evolutionIndex) + 'E')


		#end = time.time()
		#print(end - start)

		for t in range(T+1):

			fig = plt.figure(figsize=(5,4))
			plt.quiver(xPath[:,0,t]+0.5,xPath[:,1,t]+0.5, xPathVec[:,0,t], xPathVec[:,1,t], color = colorVec, scale=50)
			plt.plot([10,20],[20,20],'ro')
			plt.ylim(0, 30)
			plt.xlim(0,30)
			ax = fig.gca()

			# Major ticks every 4, minor ticks every 1
			major_ticks = np.arange(0, 31, 5)
			minor_ticks = np.arange(0, 31, 1)
			major_ticksx = np.arange(0, 31, 3)

			ax.set_xticks(major_ticksx)
			ax.set_xticks(minor_ticks, minor=True)
			ax.set_yticks(major_ticks)
			ax.set_yticks(minor_ticks, minor=True)

			plt.annotate("t " + str(4*(T-t)),xy=(28,29))
			plt.grid(which='both')
			for i in range(amountBoats):
				plt.quiver(i+0.5, 29+0.5, 1, 0, color = colorVec[sortPosition[i]], scale=50)
				plt.annotate(str(strat[sortPosition[i]]+1),xy=(i+0.5,30))
				plt.annotate("s ",xy=(0,30))


			plt.savefig( './imagesgif/' + str (evolutionStepIndex) + 'E' + str (evolutionIndex) + 'E/image'+ str (t) + '.png')
			if t == T:
				plt.pause(0.2)
			elif t< 30:
				plt.pause(0.05)
			elif t < 50:
				plt.pause(0.2)
			else:
				plt.pause(3)
			plt.draw()
			plt.close()


		fbout[:,evolutionIndex] = orderBoats(Zpost, Tpost, x0, s0, strat, colorVec, windBase, preVec, sortPosition, evolutionStepIndex, evolutionIndex)
		print(evolutionIndex, " evolutionIndex")
	print(fbout.mean(1))
	print(fbout)
	saveAverage[:,evolutionStepIndex] = stratChange(fbout,strat)
	print("evolutionStepIndex",evolutionStepIndex)
	saveStrat[:,evolutionStepIndex+1] = strat.copy()
	np.save('saveStrat.npy', saveStrat)
	np.save('saveAverage.npy', saveAverage)


print(saveStrat)
