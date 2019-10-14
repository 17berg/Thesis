# import library

import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import constants
import random
import time
import os

# options
np.set_printoptions(threshold=sys.maxsize)

###### CONSTANTS ######
d = constants.d # avoid dividing by zero
windSlowc = constants.windSlowc # constant for how the wind effect the boat
tackSlowc = constants.tackSlowc # constant how tacking effect speed
distanceSpeedc = constants.distanceSpeedc # constant how distance effect speed
amountBoats = constants.amountBoats # amount of boats
speedIncreasec = constants.speedIncreasec # constant for how the speed is increaded by wind
maxSpeed = constants.maxSpeed # constant for how the speed is increaded by wind

xMax, xMin, yMax, yMin=constants.xMax ,constants.xMin, constants.yMax, constants.yMin #frame
xh, yh = constants.xh, constants.yh #stepsize
xMesh, yMesh= constants.xMesh, constants.yMesh # amount of nodes
avoidWindc = 0.25 # constant for how the sailors avoid the wind
speedImpc = 0.13 # constant of how speed is important
Kn = 2 #observation rounds, Rounds before finding position
a = np.ones(amountBoats) # tactical variables
a *= avoidWindc # constant for how the sailors avoid the wind
clearAsternc = 25
rot = np.array([[0,1],[-1,0]]) # rotation matrix
distnaceLocUpper = 70
tackDecreaseDistans = 1/3.16
degreec = 1.8 # boat able to tack
pastbuoy = 750

chancec = 3 # constant on how much the boats are willing to take a chance, in range [0,infinity), the higher the lower chance
KnPath = 1 # round for finding the best path

Kobsvec = np.array([3,2,1,1])

Kobs = Kobsvec.shape[0] # maximum amount of future steps seen
sWidth = 1 # maximum amount of path evaluated
for i in range(Kobs):
	sWidth = sWidth*Kobsvec[i]

speedWidth = np.zeros(18) # 18 here is the amount of aviable moves
tmpMinSort = np.zeros((amountBoats,18,sWidth))
minVec = np.zeros((amountBoats,sWidth))
minSortK = np.zeros((amountBoats,sWidth))

def sWidthMake(KobsvecLoc,swidth):
	sWidthReturn = np.zeros((KobsvecLoc.shape[0],sWidth), dtype = np.int)
	currentbranches = 1
	for i in range(KobsvecLoc.shape[0]):
		for j in range(currentbranches):
			sWidthReturn[i,j] = (currentbranches-1)
			currentbranches += (KobsvecLoc[i]-1)

	return sWidthReturn

sWidthNew = sWidthMake(Kobsvec,sWidth)

###### CONSTANTS ######


def inbetween(xLoc, currentBoat,oppositeBoat, betweenBoat):# Checks if a boat is inbetween two boats, used for right of way rules
	betweenc =  (xLoc[betweenBoat,0] >= np.min([xLoc[currentBoat,0],xLoc[oppositeBoat,0]]) and xLoc[betweenBoat,0] <= np.max([xLoc[currentBoat,0],xLoc[oppositeBoat,0]])
	and xLoc[betweenBoat,1] >= np.min([xLoc[currentBoat,1],xLoc[oppositeBoat,1]]) and xLoc[betweenBoat,1] <= np.max([xLoc[currentBoat,1],xLoc[oppositeBoat,1]]))
	if betweenc:
		return 1
	return -1

def overlapp(xLoc, currentBoat,oppositeBoat, preVecObs, T): # Check if two boats are overlapping
	""" Returns if the boats overlapp or one is clear ahead
	-1 oppositeBoat priorety (not overlapping)
	1 currentBoat priorety (not overlapping)
	0 overlapping
	"""

	## oppositeBoat
	if T[preVecObs[oppositeBoat],0] > 0:
		travelVecRot = np.matmul(rot,T[preVecObs[oppositeBoat],2:4])
		if T[preVecObs[oppositeBoat],1] > 0:
			kslope = (travelVecRot[1])/(travelVecRot[0])
			m = xLoc[oppositeBoat,1]-kslope * xLoc[oppositeBoat,0]
			if  xLoc[currentBoat,1] + clearAsternc*np.abs(travelVecRot[0]) < kslope * xLoc[currentBoat,0] + clearAsternc*np.abs(travelVecRot[1]) + m:
				return -1
		elif T[preVecObs[oppositeBoat],1] == 0:
			if  xLoc[currentBoat,0] + clearAsternc < xLoc[oppositeBoat,0]:
				return -1
		else:
			kslope = (travelVecRot[1])/(travelVecRot[0])
			m = xLoc[oppositeBoat,1]-kslope * xLoc[oppositeBoat,0]
			if  xLoc[currentBoat,1] - clearAsternc*np.abs(travelVecRot[0]) > kslope * xLoc[currentBoat,0] + clearAsternc*np.abs(travelVecRot[1]) + m:
				return -1


	elif T[preVecObs[oppositeBoat],0] < 0:
		travelVecRot = np.matmul(rot,T[preVecObs[oppositeBoat],2:4])
		if T[preVecObs[oppositeBoat],1] > 0:
			kslope = (travelVecRot[1])/(travelVecRot[0])
			m = xLoc[oppositeBoat,1]-kslope * xLoc[oppositeBoat,0]
			if  xLoc[currentBoat,1] + clearAsternc*np.abs(travelVecRot[0]) < kslope * xLoc[currentBoat,0] - clearAsternc*np.abs(travelVecRot[1]) + m:
				return -1
		elif T[preVecObs[oppositeBoat],1] == 0:
			if  xLoc[currentBoat,0] - clearAsternc > xLoc[oppositeBoat,0]:
				return -1
		else:
			kslope = (travelVecRot[1])/(travelVecRot[0])
			m = xLoc[oppositeBoat,1]-kslope * xLoc[oppositeBoat,0]
			if  xLoc[currentBoat,1] - clearAsternc*np.abs(travelVecRot[0]) > kslope * xLoc[currentBoat,0] - clearAsternc*np.abs(travelVecRot[1]) + m:
				return -1


	else:
		if  xLoc[currentBoat,1] - clearAsternc > xLoc[oppositeBoat,1]:
			return -1

	## currentBoat
	if T[preVecObs[currentBoat],0] > 0:
		travelVecRot = np.matmul(rot,T[preVecObs[currentBoat],2:4])
		if T[preVecObs[currentBoat],1] > 0:
			kslope = (travelVecRot[1]+d)/(travelVecRot[0]+d)
			m = xLoc[currentBoat,1]-kslope * xLoc[currentBoat,0]
			if  xLoc[currentBoat,1] + clearAsternc*np.abs(travelVecRot[0]) < kslope * xLoc[currentBoat,0] + clearAsternc*np.abs(travelVecRot[1]) + m:
				return 1
		elif T[preVecObs[currentBoat],1] == 0:
			if  xLoc[currentBoat,0] + clearAsternc < xLoc[currentBoat,0]:
				return 1
		else:
			kslope = (travelVecRot[1]+d)/(travelVecRot[0]+d)
			m = xLoc[currentBoat,1]-kslope * xLoc[currentBoat,0]
			if  xLoc[currentBoat,1] - clearAsternc*np.abs(travelVecRot[0]) > kslope * xLoc[currentBoat,0] + clearAsternc*np.abs(travelVecRot[1]) + m:
				return 1


	elif T[preVecObs[currentBoat],0] < 0:
		travelVecRot = np.matmul(rot,T[preVecObs[currentBoat],2:4])
		if T[preVecObs[currentBoat],1] > 0:
			kslope = (travelVecRot[1]+d)/(travelVecRot[0]+d)
			m = xLoc[currentBoat,1]-kslope * xLoc[currentBoat,0]
			if  xLoc[currentBoat,1] + clearAsternc*np.abs(travelVecRot[0]) < kslope * xLoc[currentBoat,0] - clearAsternc*np.abs(travelVecRot[1]) + m:
				return 1
		elif T[preVecObs[currentBoat],1] == 0:
			if  xLoc[currentBoat,0] > xLoc[currentBoat,0] + clearAsternc:
				return 1
		else:
			kslope = (travelVecRot[1]+d)/(travelVecRot[0]+d)
			m = xLoc[currentBoat,1]-kslope * xLoc[currentBoat,0]
			if  xLoc[currentBoat,1] - clearAsternc*np.abs(travelVecRot[0]) > kslope * xLoc[currentBoat,0] - clearAsternc*np.abs(travelVecRot[1]) + m:
				return 1


	else:
		if  xLoc[currentBoat,1] > xLoc[currentBoat,1] + clearAsternc:
			return 1

	return 0 # overlapp

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

def WindPollyadd(xLoc, amountBoats, vNew, windBase): # wind shadow from all boats
	windRet = np.zeros((xMesh,yMesh))
	for i in range(amountBoats):
		x1 = np.maximum(np.minimum(xLoc[i,1],1900),100)
		x0 = np.maximum(np.minimum(xLoc[i,0],1900),100)
		windRet[x1-100:x1+101, x0-100 : x0+101] += windBase[:,:,vNew[i]%9].copy()
	return windRet

def WindPollyaddself(xLoc, amountBoats, vNew, currentBoat, windBase):# wind shadow from given boats
	windRet = np.zeros((yMesh,yMesh))
	x1 = np.maximum(np.minimum(xLoc[currentBoat,1],1900),100)
	x0 = np.maximum(np.minimum(xLoc[currentBoat,0],1900),100)
	windRet[x1-100:x1+101, x0-100 : x0+101] += windBase[:,:,vNew[currentBoat]%9].copy()
	return windRet

def prioretyFunction(xNew, xLoc, currentBoat,oppositeBoat, preVecObs, T):# right of way between two boats
	if (T[preVecObs[currentBoat],0]*T[preVecObs[oppositeBoat],0] < 0): # Rule 10
		return  np.sign(T[preVecObs[oppositeBoat],0])

	if (T[preVecObs[currentBoat],1] < 0 and T[preVecObs[oppositeBoat],1] > 0
		and xLoc[currentBoat,1] - xLoc[oppositeBoat,1] > 0):# Rule 11 one down one upwind
		return -1
	elif (T[preVecObs[currentBoat],1] > 0 and T[preVecObs[oppositeBoat],1] < 0
		and xLoc[currentBoat,1] - xLoc[oppositeBoat,1] < 0):# Rule 11 one down one upwind
		return 1

	overlappvar = overlapp(xLoc, currentBoat,oppositeBoat, preVecObs, T)

	if overlappvar == 0:
		kslope = (T[preVecObs[oppositeBoat],1]+d)/(T[preVecObs[oppositeBoat],0]+d)
		m = xLoc[oppositeBoat,1]-kslope * xLoc[oppositeBoat,0]
		if  xLoc[currentBoat,1]  > kslope * xLoc[currentBoat,0] + m:
			return -1
		return 1
	else:

		for boatIndex in range(amountBoats):
			if boatIndex != oppositeBoat and boatIndex != currentBoat:
				if (inbetween(xLoc, currentBoat,oppositeBoat, boatIndex) == 1 and
				T[preVecObs[oppositeBoat],0]*T[preVecObs[boatIndex],0] >= 0 and
				overlapp(xLoc, boatIndex,oppositeBoat, preVecObs,T) == 0 and
				overlapp(xLoc, currentBoat,boatIndex, preVecObs,T) == 0): # if boat inbetween, overlapp, same tack, and inbetween boat and current boat overlapp, overlapp rules
					kslope = (T[preVecObs[oppositeBoat],1]+d)/(T[preVecObs[oppositeBoat],0]+d)
					m = xLoc[oppositeBoat,1]-kslope * xLoc[oppositeBoat,0]
					if  xLoc[currentBoat,1]  > kslope * xLoc[currentBoat,0] + m:
						return -1
					return 1
		return overlappvar

	return 0 # default

def startPosition(x0Return,vNew,amountBoats,strat):# getting startingpositions
	x0Return[:,0] = x0Return[:,0]*25+920-250
	for i in range(amountBoats):
		if x0Return[i,1] <= 19:
			x0Return[i,1] = x0Return[i,1]*25+240
		else:
			x0Return[i,1] = 100 # if boat start in front of starting line, teleports and starts far behind everybody else

	vRet = np.zeros(amountBoats, dtype=np.int)
	for i in range(amountBoats):
		if vNew[i,0] < -0.7 and vNew[i,1] > 0.7:
			vRet[i] = 0
		elif vNew[i,0] > 0.7 and vNew[i,1] > 0.7:
			vRet[i] = 1
		elif vNew[i,0] > 0.7 and vNew[i,1] == 0:
			vRet[i] = 5
		elif vNew[i,0] < -0.7 and vNew[i,1] == 0:
			vRet[i] = 2
		elif vNew[i,0] < -0.7 and vNew[i,1] < -0.7:
			vRet[i] = 6
		elif vNew[i,0] > 0.7 and vNew[i,1] < -0.7:
			vRet[i] = 7

		elif vNew[i,0] < 0 and vNew[i,1] == 0:
			vRet[i] = 8

	return vRet

def WindType(x,wind): # Wind area that effet sailor, in regard to windshadow
	return np.sum(wind[x[1]-1:x[1]+2,x[0]-1] + wind[x[1]-1:x[1]+2,x[0]] + wind[x[1]-1:x[1]+2,x[0]+1])/9

def LegalCross(x0,xNew): # check if crossed start line legally
	if x0[0] < 1000:
		if xNew[0] > x0[0]:
			kslope = (xNew[1] - x0[1])/(xNew[0] - x0[0] + d)
			m = x0[1]-x0[0]*kslope
			if 750 >= 900*kslope+m:
				return 1
			else:
				return 0
		else:
			kslope = (xNew[1] - x0[1])/(xNew[0] - x0[0] + d)
			m = x0[1]-x0[0]*kslope
			if 750 <= 900*kslope+m:
				return 1
			else:
				return 0
	else:
		if xNew[0] > x0[0]:
			kslope = (xNew[1] - x0[1])/(xNew[0] - x0[0]+ d)
			m = x0[1]-x0[0]*kslope
			if 750 <= 1100*kslope+m:
				return 1
			else:
				return 0
		else:
			kslope = (xNew[1] - x0[1])/(xNew[0] - x0[0]+ d)
			m = xNew[1]-xNew[0]*kslope
			if 750 >= 1100*kslope+m:
				return 1
			else:
				return 0
	return 0

def LegalCrossbouy(x0,xNew): # check if crossed start line legally
	if x0[0] <= 1000:
		return 0
	else:
		kslope = (x0[1] - xNew[1])/(x0[0] - xNew[0])
		m = x0[1]-x0[0]*kslope
		if 1500 <= 1000*kslope+m:
			return 1
		else:
			return 0

def LegalCrossbouy2(x0,xNew): # check if crossed start line legally
	kslope = (xNew[1] - x0[1])/(xNew[0] - x0[0])
	m = x0[1]-x0[0]*kslope
	if 1500 >= 1000*kslope+m:
		return 1
	else:
		return 0

def collid(currentBoat, xs ,x0,xNew, v,amountBoats, T): # check if collid
	collisionc = 15
	endcollisionc = 17
	closeToBouy = np.sqrt((xNew[currentBoat,0]-1000)**2+(xNew[currentBoat,1]-1500)**2) <= 200
	for i in range(amountBoats):
		if i != currentBoat:
			if  closeToBouy and T[v[i],0]*T[v[currentBoat],0] <= 0:
				if T[v[i],0] <= 0:
					if (np.sqrt((xNew[i,0]-xs[0])**2+(xNew[i,1]-xs[1])**2) <= endcollisionc or np.sqrt(((xNew[i,0]+T[v[i],0]/4)-xs[0])**2+((xNew[i,1]+T[v[i],1]/4)-xs[1])**2) <= endcollisionc):
						if prioretyFunction(xNew, x0, currentBoat, i , v, T) <= 0:
							return 0
				elif T[v[currentBoat],0] <= 0:
					if (np.sqrt((xNew[i,0]-xs[0])**2+(xNew[i,1]-xs[1])**2) <= endcollisionc or np.sqrt((xNew[i,0]-(xs[0]+T[v[currentBoat],0]/4))**2+(xNew[i,1]-(xs[1]+T[v[currentBoat],1]/4))**2) <= endcollisionc):

						if prioretyFunction(xNew, x0, currentBoat, i , v, T) <= 0:
							return 0
			elif np.sqrt((xNew[i,0]-xs[0])**2+(xNew[i,1]-xs[1])**2) <= collisionc:
				if prioretyFunction(xNew, x0, currentBoat, i , v, T) <= 0:
					return 0
	return 1 # dont collid or have priorety

def nextStep(fboutloc, j, xNew, vNew, v, windBase, T, s, x0, wind, Zpost, crossedStart, xtmp, vtmp, sIndexLoc,  createIndicator, iObs, infiniteloop): # evaluating aviable steps

	windself = WindPollyaddself(xNew, amountBoats, vNew, j, windBase)
	minvalue = 1000
	minimumk = 8
	if (fboutloc[j] >= 1 and x0[j,0] < pastbuoy):
		xtmp[j,:] = [0,2000-j*30]

	else:
		for k in range(T.shape[0]):
			xsx, xsy = int(T[k,0]*s[j,iObs]), int(T[k,1]*s[j,iObs])
			vector1 = unit_vector([xsx,xsy])
			degree = np.abs(angle_between(vector1, T[v[j],2:4]))
			xsx, xsy = int(xsx/(1+degree*tackDecreaseDistans)), int(xsy/(1+degree*tackDecreaseDistans))

			xs = [(x0[j,0]+xsx)%2000,(x0[j,1]+xsy)%2000] # new step
			windslow = WindType(xs,wind)-WindType(xs,windself)
			distanceLoc = np.linalg.norm([xsx,xsy])
			stmp = s[j,iObs]*max((1-degree*tackSlowc-distanceSpeedc*(k >= 8)),0) + (maxSpeed-s[j,iObs])*speedIncreasec
			tmpmin = Zpost[xs[1],xs[0]]*(1+((windslow)*a[j]+(1.2-stmp)*speedImpc)*0.2)


			if crossedStart[j] == 0 and xs[1] >= 750: # cross line legally
				if LegalCross(x0[j,:],xs) == 0:
					tmpmin = 100000000

			elif fboutloc[j] == 0 and xs[0] >= 1000 and xs[1] >= 1500 and x0[j,0] <= 1000 and x0[j,1] <= 1500: # crossing bouy legally
				if LegalCrossbouy2(x0[j,:],xs) == 0:
					tmpmin = 100000000

			elif fboutloc[j] == 0 and ((xs[0] <= 1000 and xs[1] >= 1500) or (xs[0] <= 1000 and xs[1] <= 1500 and x0[j,0] >= 1000 and x0[j,1] >= 1500)): # crossing bouy legally
				if LegalCrossbouy(x0[j,:],xs) == 0:
					tmpmin = 100000000

			elif degree > degreec:
				tmpmin = 100000000

			elif 3 <= infiniteloop:
				tmpmin = tmpmin * np.abs(np.random.normal(loc = 1,scale = infiniteloop/20, size = 1))

			if createIndicator == 1:
				tmpMinSort[j,k,sIndexLoc] = tmpmin
				speedWidth[k] = stmp

				if tmpmin < minvalue and (collid(j, xs, x0, xNew, v, amountBoats, T) or infiniteloop >= 3):
					s[j,iObs+1] = stmp
					xtmp[j,:] = xs.copy()
					minvalue = tmpmin
					minimumk = k
					minVec[j,sIndexLoc] = tmpmin
			else:
				if tmpmin < minvalue and (collid(j, xs, x0, xNew, v, amountBoats, T) or infiniteloop >= 3):
					s[j,iObs+1] = stmp
					xtmp[j,:] = xs.copy()
					minvalue = tmpmin
					minimumk = k

		minSortK[j,sIndexLoc] = minimumk
		vtmp[j] = minimumk

def nextStepSecond(boat, PositionNext, iObs, xObs, preVecObs, xNew, preVecNew, sIndexLoc, sIndextmpLoc, s, tmpMinSortLoc, T): # Finding second/third ... best step

	sortMinVec = np.argsort(tmpMinSortLoc[boat,:,sIndexLoc])
	index = 0
	while sortMinVec[index] != minSortK[boat, sIndexLoc]:
		index += 1
	if PositionNext > 0:
		for i in range(PositionNext):
			if index == 17:
				xs = xNew[boat,:].copy()
				break
			index += 1
			xsx, xsy = int(T[sortMinVec[index],0]*s[boat,iObs, sIndexLoc]), int(T[sortMinVec[index],1]*s[boat,iObs, sIndexLoc])
			vector1 = unit_vector([xsx,xsy])
			degree = np.abs(angle_between(vector1, T[preVecObs[boat,iObs],2:4]))
			xsx, xsy = int(xsx/(1+degree*tackDecreaseDistans)), int(xsy/(1+degree*tackDecreaseDistans))


			xs = [(xObs[boat,0,iObs]+xsx),(xObs[boat,1,iObs]+xsy)]

			while collid(boat,xObs[boat,:,iObs+1], xObs[:,:,iObs], xObs[:,:,iObs+1], preVecObs[:,iObs],amountBoats, T) <= 0 and index < 17:
				index += 1

				xsx, xsy = int(T[sortMinVec[index],0]*s[boat,iObs, sIndexLoc]), int(T[sortMinVec[index],1]*s[boat,iObs, sIndexLoc])
				vector1 = unit_vector([xsx,xsy])
				degree = np.abs(angle_between(vector1, T[preVecObs[boat,iObs],2:4]))
				xsx, xsy = int(xsx/(1+degree*tackDecreaseDistans)), int(xsy/(1+degree*tackDecreaseDistans))

				xs = [(xObs[boat,0,iObs]+xsx),(xObs[boat,1,iObs]+xsy)]
	else:
		xsx, xsy = int(T[sortMinVec[index],0]*s[boat,iObs, sIndexLoc]), int(T[sortMinVec[index],1]*s[boat,iObs, sIndexLoc])
		vector1 = unit_vector([xsx,xsy])
		degree = np.abs(angle_between(vector1, T[preVecObs[boat,iObs],2:4]))
		xsx, xsy = int(xsx/(1+degree*tackDecreaseDistans)), int(xsy/(1+degree*tackDecreaseDistans))
		xs = [(xObs[boat,0,iObs]+xsx),(xObs[boat,1,iObs]+xsy)]

	s[boat,iObs+1,sIndextmpLoc] = speedWidth[sortMinVec[index]]
	xNew[boat,:] = xs.copy()
	preVecNew[boat] = sortMinVec[index]

def recurrsion(crash, xObs, preVecObs, sIndex, xtmp, vtmp, boat, iObs, collidVecLoc, sObs, windBase, fboutloc, T, Zpost, crossedStart): # recurrsion to find paths the boat can take
	if crash == 0 and iObs < Kobs: # make sure a crash hasn't happend
		for i in range(Kn): # observation before finding next position
			wind = WindPollyadd(xObs[:,:,iObs+1,sIndex], amountBoats, preVecObs[:,iObs+1,sIndex], windBase)
			for j in range(amountBoats): # finding best next step
				if collidVecLoc[j] == 1:
					if j == boat:
						nextStep(fboutloc, j, xObs[:,:,iObs+1,sIndex], preVecObs[:,iObs+1,sIndex],preVecObs[:,iObs,sIndex],windBase,T,sObs[:,:,sIndex],xObs[:,:,iObs,sIndex],wind,Zpost,crossedStart,xtmp,vtmp,sIndex, 1, iObs, 0)
					else:
						nextStep(fboutloc, j, xObs[:,:,iObs+1,sIndex], preVecObs[:,iObs+1,sIndex],preVecObs[:,iObs,sIndex],windBase,T,sObs[:,:,sIndex],xObs[:,:,iObs,sIndex],wind,Zpost,crossedStart,xtmp,vtmp,sIndex, 0, iObs, 0)
				else:
					xtmp[j,:] = xObs[j,:,iObs+1,0].copy()
					vtmp[j] = preVecObs[j,iObs+1,0].copy()
			xObs[:,:,iObs+1,sIndex] = xtmp.copy()
			preVecObs[:,iObs+1,sIndex] = vtmp.copy()


		for tmpIndex in range(1, Kobsvec[iObs]):
			collidVecNew = collidVecLoc.copy()
			sIndextmp = sWidthNew[iObs,sIndex] + tmpIndex
			nextStepSecond(boat, tmpIndex, iObs, xObs[:,:,:,sIndex], preVecObs[:,:,sIndex], xtmp, vtmp, sIndex, sIndextmp, sObs, tmpMinSort, T)
			xObs[:,:,:iObs+1,sIndextmp] = xObs[:,:,:iObs+1,sIndex].copy()
			preVecObs[:,:iObs+1,sIndextmp] = preVecObs[:,:iObs+1,sIndex].copy()
			xObs[:,:,iObs+1,sIndextmp] = xtmp.copy()
			preVecObs[:,iObs+1,sIndextmp] = vtmp.copy()
			sObs[:,:iObs+1,sIndextmp] = sObs[:,:iObs+1,sIndex].copy()

			fboutloctmp = fboutloc.copy()
			crossedStartTmp = crossedStart.copy()

			collidVar = 0
			infiniteloop = 0
			crashtmp = crash

			while collidVar == 0: # make sure no collid
				wind = WindPollyadd(xObs[:,:,iObs+1,sIndextmp], amountBoats, preVecObs[:,iObs+1,sIndextmp], windBase)
				infiniteloop += 1
				collidVar = 1
				for j in range(amountBoats):
					if collid(j,xObs[j,:,iObs+1,sIndextmp],xObs[:,:,iObs,sIndextmp], xObs[:,:,iObs+1,sIndextmp], preVecObs[:,iObs,sIndextmp], amountBoats, T) <= 0: # if the boat collid
						if j == boat:
							nextStep(fboutloctmp,j,xObs[:,:,iObs+1,sIndextmp],preVecObs[:,iObs+1,sIndextmp],preVecObs[:,iObs,sIndextmp],windBase,T,sObs[:,:,sIndextmp],xObs[:,:,iObs,sIndextmp],wind,Zpost,crossedStartTmp,xtmp,vtmp,sIndex,1, iObs, infiniteloop)
						else:
							nextStep(fboutloctmp,j,xObs[:,:,iObs+1,sIndextmp],preVecObs[:,iObs+1,sIndextmp],preVecObs[:,iObs,sIndextmp],windBase,T,sObs[:,:,sIndextmp],xObs[:,:,iObs,sIndextmp],wind,Zpost,crossedStartTmp,xtmp,vtmp,sIndex,0, iObs, infiniteloop)
							collidVecNew[j] = 1
				xObs[:,:,iObs+1,sIndextmp] = xtmp.copy()
				preVecObs[:,iObs+1,sIndextmp] = vtmp.copy()
				for j in range(amountBoats):
					collidVar = collid(j,xObs[j,:,iObs+1,sIndextmp],xObs[:,:,iObs,sIndextmp], xObs[:,:,iObs+1,sIndextmp], preVecObs[:,iObs,sIndextmp], amountBoats, T)*collidVar # keeps being 1 if they don't collid

				if infiniteloop == 10: # crash
					collidVar = 1
					minVec[j,sIndextmp] = 10000
					crashtmp = 1


			wind = WindPollyadd(xObs[:,:,iObs+1,sIndextmp], amountBoats, preVecObs[:,iObs,sIndextmp], windBase)
			for boatIndex in range(amountBoats):
				if collidVecLoc[j] == 1:
					windself = WindPollyaddself(xObs[:,:,iObs+1,sIndextmp], amountBoats, preVecObs[:,iObs+1,sIndextmp], boatIndex, windBase)
					sObs[boatIndex,iObs+1,sIndextmp] = sObs[boatIndex,iObs+1,sIndextmp]*(1-(WindType(xObs[boatIndex,:,iObs+1,sIndextmp],wind)-WindType(xObs[boatIndex,:,iObs+1,sIndextmp],windself))*windSlowc)


				if fboutloctmp[boatIndex] == 0 and ((xObs[boatIndex,0,iObs+1,sIndextmp] <= 1000 and xObs[boatIndex,1,iObs+1,sIndextmp] >= 1500) or (xObs[boatIndex,0,iObs+1,sIndextmp] <= 1000 and xObs[boatIndex,1,iObs+1,sIndextmp] <= 1500 and xObs[boatIndex,0,iObs,sIndextmp] >= 1000 and xObs[boatIndex,1,iObs,sIndextmp] >= 1500) ): # crossing bouy legally
					if LegalCrossbouy(xObs[boatIndex,:,iObs,sIndextmp],xObs[boatIndex,:,iObs+1,sIndextmp]) == 1:
						fboutloctmp[boatIndex] = 1

				if crossedStartTmp[boatIndex] == 0 and xObs[boatIndex,1,iObs+1,sIndextmp] >= 750 and xObs[boatIndex,1,iObs,sIndextmp] <= 750 and LegalCross(xObs[boatIndex,:,iObs,sIndextmp], xObs[boatIndex,:,iObs+1,sIndextmp]):
					crossedStartTmp[boatIndex] = 1

			recurrsion(crashtmp, xObs, preVecObs, sIndextmp, xtmp, vtmp, boat, iObs+1, collidVecNew, sObs, windBase, fboutloctmp, T, Zpost, crossedStartTmp)

		fboutloctmp = fboutloc.copy()
		crossedStartTmp = crossedStart.copy()
		xtmp = xObs[:,:,iObs+1,sIndex].copy()
		vtmp = preVecObs[:,iObs+1,sIndex].copy()
		collidVecNew = collidVecLoc
		sIndextmp = sIndex
		collidVar = 0
		infiniteloop = 0

		collidVar = 0
		infiniteloop = 0
		while collidVar == 0: # make sure no collid
			wind = WindPollyadd(xObs[:,:,iObs+1,sIndextmp], amountBoats, preVecObs[:,iObs+1,sIndextmp], windBase)
			infiniteloop += 1
			collidVar = 1
			for j in range(amountBoats):

				if collid(j,xObs[j,:,iObs+1,sIndextmp],xObs[:,:,iObs,sIndextmp], xObs[:,:,iObs+1,sIndextmp], preVecObs[:,iObs,sIndextmp], amountBoats, T) <= 0: # if the boat collid
					if j == boat:
						nextStep(fboutloctmp,j,xObs[:,:,iObs+1,sIndextmp],preVecObs[:,iObs+1,sIndextmp],preVecObs[:,iObs,sIndextmp],windBase,T,sObs[:,:,sIndextmp],xObs[:,:,iObs,sIndextmp],wind,Zpost,crossedStart,xtmp,vtmp,sIndex,1, iObs, collidVar)
					else:
						nextStep(fboutloctmp,j,xObs[:,:,iObs+1,sIndextmp],preVecObs[:,iObs+1,sIndextmp],preVecObs[:,iObs,sIndextmp],windBase,T,sObs[:,:,sIndextmp],xObs[:,:,iObs,sIndextmp],wind,Zpost,crossedStart,xtmp,vtmp,sIndex,0, iObs, collidVar)
						collidVecNew[j] = 1
			xObs[:,:,iObs+1,sIndextmp] = xtmp.copy()
			preVecObs[:,iObs+1,sIndextmp] = vtmp.copy()
			for j in range(amountBoats):
				collidVar = collid(j,xObs[j,:,iObs+1,sIndextmp],xObs[:,:,iObs,sIndextmp], xObs[:,:,iObs+1,sIndextmp], preVecObs[:,iObs,sIndextmp], amountBoats, T)*collidVar # keeps being 1 if they don't collid

			if infiniteloop == 10: # crash
				collidVar = 1
				minVec[j,sIndextmp] = 10000
				crash = 1

		wind = WindPollyadd(xObs[:,:,iObs+1,sIndextmp], amountBoats, preVecObs[:,iObs,sIndextmp], windBase)
		for boatIndex in range(amountBoats):
			if collidVecLoc[j] == 1:
				windself = WindPollyaddself(xObs[:,:,iObs+1,sIndextmp], amountBoats, preVecObs[:,iObs+1,sIndextmp], boatIndex, windBase)
				sObs[boatIndex,iObs+1,sIndextmp] = sObs[boatIndex,iObs+1,sIndextmp]*(1-(WindType(xObs[boatIndex,:,iObs+1,sIndextmp],wind)-WindType(xObs[boatIndex,:,iObs+1,sIndextmp],windself))*windSlowc)

				if fboutloctmp[boatIndex] == 0 and ((xObs[boatIndex,0,iObs+1,sIndextmp] <= 1000 and xObs[boatIndex,1,iObs+1,sIndextmp] >= 1500) or (xObs[boatIndex,0,iObs+1,sIndextmp] <= 1000 and xObs[boatIndex,1,iObs+1,sIndextmp] <= 1500 and xObs[boatIndex,0,iObs,sIndextmp] >= 1000 and xObs[boatIndex,1,iObs,sIndextmp] >= 1500) ): # crossing bouy legally
					if LegalCrossbouy(xObs[boatIndex,:,iObs,sIndextmp],xObs[boatIndex,:,iObs+1,sIndextmp]) == 1:
						fboutloctmp[boatIndex] = 1

					if crossedStartTmp[boatIndex] == 0 and xObs[boatIndex,1,iObs+1,sIndextmp] >= 750 and xObs[boatIndex,:,iObs,sIndextmp] <= 750 and LegalCross(xObs[boatIndex,:,iObs,sIndextmp], xObs[boatIndex,:,iObs+1,sIndextmp]):
						crossedStartTmp[boatIndex] = 1

		recurrsion(crash, xObs, preVecObs, sIndextmp, xtmp, vtmp, boat, iObs+1, collidVecNew, sObs, windBase, fboutloctmp, T, Zpost, crossedStartTmp)

def orderBoats(Zpost, T, finalPosition, sFinal, strat, colorVec, windBase, vNew, sortPosition, evolutionStep, evolutionIndex): # one round, returning the order the boats crossed the buoy

	global Kobsvec
	global Kobs
	global sWidth

	crossedStart = np.zeros(amountBoats, dtype = np.int)

	position, ploti = 0, 0
	x0 = finalPosition.copy() # x position
	xs = np.zeros(2, dtype=np.int) # update step

	s = sFinal.copy() # speed
	fbout = np.zeros(amountBoats, dtype=np.int) # finish positions

	v = startPosition(x0,vNew, amountBoats,strat) # start position, travel angel vector
	vtmp = v.copy()
	xtmp =  np.zeros((amountBoats,2), dtype=np.int)


	xPath = np.zeros((amountBoats,2,100), dtype=np.int) # plot
	xPathVec = np.zeros((amountBoats,100), dtype=np.int) # plot

	xPath[:,:,ploti] = x0.copy()
	xPathVec[:,ploti] = v.copy()

	######### FUTURE UPDATE #########
	xObs = np.ones((amountBoats,2,Kobs+1, sWidth), dtype=np.int) # observation
	preVecObs =  np.ones((amountBoats,Kobs+1,sWidth), dtype=np.int)
	sObs = np.ones((amountBoats,Kobs+1, sWidth))

	xNextFinal = np.ones((amountBoats,2,Kobs+1), dtype=np.int) # final
	preVecFinal = np.ones((amountBoats,Kobs+1), dtype=np.int)
	sVecFinal = np.ones((amountBoats,Kobs+1))

	firstcrossing = 0
	while np.sort(fbout)[1] == 0: # while atleast one boat haven't reached the bouy
		if ploti == 5:
			Kobsvec = np.array([3,2,1])

			Kobs = Kobsvec.shape[0] # maximum amount of future steps seen
			sWidth = 1 # maximum amount of path evaluated
			for i in range(Kobs):
				sWidth = sWidth*Kobsvec[i]
		if max(fbout) > 0:
			Kobsvec = np.array([3,2,1,1])

			Kobs = Kobsvec.shape[0] # maximum amount of future steps seen
			sWidth = 1 # maximum amount of path evaluated
			for i in range(Kobs):
				sWidth = sWidth*Kobsvec[i]

		######### FUTURE UPDATE #########
		#########	#########	######### Step 1 #########	#########	#########
		for tmpIndex in range(sWidth):
			sObs[:,0,tmpIndex] = s.copy()
			xObs[:,:,1,tmpIndex] = x0.copy()
			xObs[:,:,0,tmpIndex] = x0.copy()
			preVecObs[:,0,tmpIndex] = vtmp.copy()

		for iObs in range(Kobs): # observation rounds, greedy paths for the boats
			wind = WindPollyadd(xObs[:,:,iObs+1,0], amountBoats, preVecObs[:,iObs+1,0], windBase)
			for i in range(Kn): # observation before finding next position
				for j in range(amountBoats): # finding best next step
					nextStep(fbout,j,xObs[:,:,iObs+1,0],preVecObs[:,iObs+1,0], preVecObs[:,iObs,0],windBase,T,sObs[:,:,0],xObs[:,:,iObs,0],wind,Zpost,crossedStart,xtmp,vtmp,0, 0, iObs, 0)
				xObs[:,:,iObs+1,0] = xtmp.copy()
				preVecObs[:,iObs+1,0] = vtmp.copy()

			collidVar = 0
			infiniteloop = 0
			while collidVar == 0: # make sure no collid
				wind = WindPollyadd(xObs[:,:,iObs+1,0], amountBoats, preVecObs[:,iObs+1,0], windBase)
				infiniteloop += 1
				collidVar = 1
				for j in range(amountBoats):
					if collid(j,xObs[j,:,iObs+1,0],xObs[:,:,iObs,0], xObs[:,:,iObs+1,0], preVecObs[:,iObs,0], amountBoats, T) <= 0: # if the boat collid
						nextStep(fbout,j,xObs[:,:,iObs+1,0],preVecObs[:,iObs+1,0],preVecObs[:,iObs,0],windBase,T,sObs[:,:,0],xObs[:,:,iObs,0],wind,Zpost,crossedStart,xtmp,vtmp,0, 0, iObs, infiniteloop)
				xObs[:,:,iObs+1,0] = xtmp.copy()
				preVecObs[:,iObs+1,0] = vtmp.copy()
				for j in range(amountBoats):
					collidVar = collid(j,xObs[j,:,iObs+1,0],xObs[:,:,iObs,0], xObs[:,:,iObs+1,0], preVecObs[:,iObs,0], amountBoats, T)*collidVar # keeps being 1 if they don't collid

				if infiniteloop == 10: # crash
					collidVar = 1

			wind = WindPollyadd(xObs[:,:,iObs+1,0], amountBoats, preVecObs[:,iObs+1,0], windBase)
			for boatIndex in range(amountBoats):
				windself = WindPollyaddself(xObs[:,:,iObs+1,0], amountBoats, preVecObs[:,iObs+1,0], boatIndex, windBase)
				sObs[boatIndex,iObs+1,0] = sObs[boatIndex,iObs+1,0]*(1-(WindType(xObs[boatIndex,:,iObs+1,0],wind)-WindType(xObs[boatIndex,:,iObs+1,0],windself))*windSlowc)



		#########	#########	######### Step 2 #########	#########	#########

		for iObsMax in range(KnPath):
			for boat in range(amountBoats):
				if (fbout[boat] >= 1 and xObs[boat,0,0,0] < pastbuoy): # removes irrelevent boats from simmulation
					xNextFinal[boat,:,1] = [0,2000-boat*30]
					preVecFinal[boat,1] = 0
					sVecFinal[boat,1] = 1
					sVecFinal[boat,0] = 1

				else:
					collidVec = np.zeros(amountBoats)
					collidVec[boat] = 1
					xtmp = xObs[:,:,0,0].copy()
					vtmp = preVecObs[:,0,0].copy()

					fboutTmp = fbout.copy()
					crossedStartTmp = crossedStart.copy()

					recurrsion(0, xObs, preVecObs, 0, xtmp, vtmp, boat, 0, collidVec, sObs,windBase, fboutTmp, T, Zpost, crossedStartTmp)


					if T[preVecObs[boat,0,0],0] <= 0 and xObs[boat,0,0,0] <= 1000 and xObs[boat,1,0,0] <= 1400 and iObsMax + 1 == KnPath:
						minimumArgument = np.argsort(minVec[boat,:])
						secondMinIndex = -1

						tmpMinVec = np.zeros(2)
						tmpMinVec[0] = minVec[boat,minimumArgument[0]]


						for tmpLoopIndex in minimumArgument:
							if T[preVecObs[boat,1,tmpLoopIndex],0]*T[preVecObs[boat,1,minimumArgument[0]],0] < 0 and tmpLoopIndex != minimumArgument[0]:
								tmpMinVec[1] = minVec[boat,tmpLoopIndex]
								secondMinIndex = tmpLoopIndex
								break

						if secondMinIndex == -1:

							xNextFinal[boat,:,:] = xObs[boat,:,:,minimumArgument[0]].copy()
							preVecFinal[boat,:] = preVecObs[boat,:,minimumArgument[0]].copy()
							sVecFinal[boat,:] = sObs[boat,:,minimumArgument[0]].copy()

						else:

							powerVec = 1/np.power(tmpMinVec-np.min(tmpMinVec)+1,chancec)
							vectorSum = np.sum(np.absolute(powerVec))
							vectorCumSum = np.cumsum(powerVec/vectorSum)
							u = random.uniform(0, 1)
							if vectorCumSum[0] >= u:
								xNextFinal[boat,:,:] = xObs[boat,:,:,minimumArgument[0]].copy()
								preVecFinal[boat,:] = preVecObs[boat,:,minimumArgument[0]].copy()
								sVecFinal[boat,:] = sObs[boat,:,minimumArgument[0]].copy()
							else:
								xNextFinal[boat,:,:] = xObs[boat,:,:,secondMinIndex].copy()
								preVecFinal[boat,:] = preVecObs[boat,:,secondMinIndex].copy()
								sVecFinal[boat,:] = sObs[boat,:,secondMinIndex].copy()

					else:

						minimumArgument = np.argsort(minVec[boat,:])
						xNextFinal[boat,:,:] = xObs[boat,:,:,minimumArgument[0]].copy()
						preVecFinal[boat,:] = preVecObs[boat,:,minimumArgument[0]].copy()
						sVecFinal[boat,:] = sObs[boat,:,minimumArgument[0]].copy()

			xObs[:,:,:,0] = xNextFinal[:,:,:].copy()
			preVecObs[:,:,0] = preVecFinal[:,:].copy()
			sObs[:,:,0] = sVecFinal[:,:].copy()
		#########	#########	######### Step 3 #########	#########	#########

		collidVar = 0
		infiniteloop = 0
		xtmp = xNextFinal[:,:,1].copy()
		vtmp = preVecFinal[:,1].copy()
		xNextFinal[:,:,-1] = xPath[:,:,ploti-1].copy()
		preVecFinal[:,-1] = xPathVec[:,ploti-1].copy()
		collidVec = np.zeros(amountBoats)
		while collidVar == 0: # make sure no collid
			wind = WindPollyadd(xNextFinal[:,:,1], amountBoats, preVecFinal[:,1], windBase)
			infiniteloop += 1
			collidVar = 1
			for j in range(amountBoats):
				if collid(j,xNextFinal[j,:,1], xNextFinal[:,:,0], xNextFinal[:,:,1], preVecFinal[:,0], amountBoats, T) <= 0: # if the boat collid
					nextStep(fbout,j,xNextFinal[:,:,1], preVecFinal[:,1], preVecFinal[:,0], windBase,T, sObs[:,:,0], xNextFinal[:,:,0],wind,Zpost,crossedStart,xtmp,vtmp,0, 0, 0, infiniteloop)
			xNextFinal[:,:,1] = xtmp.copy()
			preVecFinal[:,1] = vtmp.copy()
			for j in range(amountBoats):
				collidVar = collid(j,xNextFinal[j,:,1], xNextFinal[:,:,0], xNextFinal[:,:,1], preVecFinal[:,0], amountBoats, T)*collidVar # keeps being 1 if they don't collid

			if infiniteloop == 10: # crash
				collidVar = 1

		wind = WindPollyadd(xNextFinal[:,:,1], amountBoats, preVecFinal[:,1], windBase)
		tmpcrosslineindex = 0
		for boatIndex in range(amountBoats):
			windself = WindPollyaddself(xNextFinal[:,:,1], amountBoats, preVecFinal[:,1], boatIndex, windBase)
			sVecFinal[boatIndex,1] = sVecFinal[boatIndex,1]*(1-(WindType(xNextFinal[boatIndex,:,1],wind)-WindType(xNextFinal[boatIndex,:,1],windself))*windSlowc)
			if fbout[boatIndex] == 0 and ((xNextFinal[boatIndex,0,1] <= 1000 and xNextFinal[boatIndex,1,1] >= 1500) or (xNextFinal[boatIndex,0,1] <= 1000 and xNextFinal[boatIndex,1,1] <= 1500 and xNextFinal[boatIndex,0,0] >= 1000 and xNextFinal[boatIndex,1,0] >= 1500)): # crossing bouy legally
				if LegalCrossbouy(xNextFinal[boatIndex,:,0],xNextFinal[boatIndex,:,1]) == 1 and crossedStart[boatIndex] == 1:
					if tmpcrosslineindex == 0:
						position = position + 1
						tmpcrosslineindex = 1
					fbout[boatIndex] = position

			if crossedStart[boatIndex] == 0 and xNextFinal[boatIndex,1,1] >= 750 and xNextFinal[boatIndex,1,0] <= 750 and LegalCross(xNextFinal[boatIndex,:,0], xNextFinal[boatIndex,:,1]):
				crossedStart[boatIndex] = 1


		x0 = xNextFinal[:,:,1].copy()
		v = preVecFinal[:,1].copy()
		for sIndexTmp in range(amountBoats):
			s[sIndexTmp] = max(sVecFinal[sIndexTmp,1],0.55)

		xPath[:,:,ploti+1] = x0.copy()
		xPathVec[:,ploti+1] = v.copy()
		print(xPath[:,:,ploti+1])
		print(ploti, fbout)
		print(crossedStart)

		ploti = ploti +1

		if ploti == 80:
			position = position + 1
			for boatIndex in range(amountBoats):
				if fbout[boatIndex] == 0:
					fbout[boatIndex] = position
			break

		#########	#########	#########
	position = position + 1
	for boatIndex in range(amountBoats):
		if fbout[boatIndex] == 0:
			fbout[boatIndex] = position



	if os.path.isdir('./gifafter/' + str (evolutionStep) + 'E' + str (evolutionIndex) + 'E') == 0:
		os.mkdir('./gifafter/' + str (evolutionStep) + 'E' + str (evolutionIndex) + 'E')

	for plotIndex in range(ploti+1):
		for pl in range(amountBoats):
			plt.quiver(xPath[pl,0,plotIndex],xPath[pl,1,plotIndex], T[xPathVec[pl,plotIndex],2], T[xPathVec[pl,plotIndex],3], color = colorVec[pl], scale=70)
		plt.plot([900,1100], [750,750], 'ro')
		plt.plot([1000], [1500], 'g^')

		for i in range(amountBoats):
			plt.quiver(i*50+350, 250, 1, 0, color = colorVec[sortPosition[i]], scale=50)
			plt.annotate(str(strat[sortPosition[i]]+1), xy=(i*50+350,300))
		plt.annotate("s", xy=(300,300))
		plt.annotate("t" + str(plotIndex), xy=(300,350))


		plt.ylim((200,1800))
		plt.xlim((300,1700))



		plt.savefig('gifafter/' + str (evolutionStep) + 'E' + str (evolutionIndex) + 'E/image'+ str (plotIndex) + '.png')
		plt.pause(0.1)
		plt.draw()
		plt.close()


	return fbout
