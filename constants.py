amountBoats = 12 # amount of boats
tackSlowc = 1/3.6 # constant how tacking effect speed
distanceSpeedc = 0.2 # constant how travel distance effect speed
speedIncreasec = 0.25 # constant for how the speed is increased by wind
maxSpeed = 1.2 # constant for how the speed is increaded by wind
windSlowc = 0.2 # constant for how the wind shadow effect the boat
d = 10 ** -4 # Avoid dividing by zeros

xMax, xMin, yMax, yMin = 10 , -10, 15, -5 #frame
xh, yh = 0.01, 0.01 #stepsize
xMesh, yMesh= int((xMax-xMin)/xh+1), int((yMax-yMin)/yh+1)# amount of nodes
