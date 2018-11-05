def dist(A,B):
    return sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)

def dist2(A,B):
    return (B[0]-A[0])**2 + (B[1]-A[1])**2

def sarea(A,B,C):
    return (1/2)*((B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0]))

def orientation(A,B,C):
    if sarea(A,B,C) > 0:
        return 1
    elif sarea(A,B,C) < 0:
        return -1
    return 0

def midPoint(A,B):
    return [(A[0]+B[0])/2, (A[1]+B[1])/2]

def inSegment(P,s):
    maxX = 0 if s[0][0] >= s[1][0] else 1
    maxY = 0 if s[0][1] >= s[1][1] else 1
    if P[0] >= s[maxX-1][0] and P[0] <= s[maxX][0] and P[1] >= s[maxY-1][1] and P[1] <= s[maxY][1] and sarea(s[0],s[1],P) == 0:
        return 1
    else:
        return 0

def inTriangle(P,t):
    orion = 0
    for i in range(len(t)):
        orion += orientation(t[i-1], t[i], P)
    if orion == 3 or orion == -3:
        return 1
    else:
        j = 0
        inSeg = 0
        while j < len(t) and inSeg != 1:
            inSeg = inSegment(P,[t[j-1],t[j]])
            j+=1
        return 1 if inSeg == 1 else 0

def segmentIntersectionTest(a,b):
    return (orientation(a[0],a[1],b[0])*orientation(a[0],a[1],b[1]) != 1) and (orientation(b[0],b[1],a[0])*orientation(b[0],b[1],a[1]) != 1)

def gradient(r):
    return (r[1][1]-r[0][1])/(r[1][0]-r[0][0])

def getYFrom(line,x):
    return gradient(line)*(x-line[0][0]) + line[0][1]

def falseLineIntersection(r,s):
    if not segmentIntersectionTest(r,s):
        return "Estas dos rectas no intersecan"
    rGradient = gradient(r)
    sGradient = gradient(s)
    commonX = (rGradient*r[0][0]-sGradient*s[0][0] + s[0][1] - r[0][1])/(rGradient-sGradient)
    return [commonX, getYFrom(r,commonX)]

def lineIntersection(r,s):
    x1 = r[0][0]
    y1 = r[0][1]
    x2 = r[1][0]
    y2 = r[1][1]
    x12 = s[0][0]
    y12 = s[0][1]
    x22 = s[1][0]
    y22 = s[1][1]
    #determinante de los parámetros
    deter = ((y2 - y1) * (x12 - x22)) - ((x1 - x2) * (y22 - y12))
    if deter == 0:
    #Las rectas son paralelas o coincidentes.
        return
    p = x1 * (y2 - y1) - y1 * (x2 - x1)
    q = x12 * (y22 - y12) - y12 * (x22 - x12)
    x = (p * (x12 - x22) - q * (x1 - x2)) / (deter)
    y = (q * (y2 - y1) - p * (y22 - y12)) / (deter)
    return [x,y] 

def circumcenter(a,b,c):
    if sarea(a,b,c) == 0:
        return []
    mab = midPoint(a,b)
    mbc = midPoint(b,c)
    vmab = [b[0]-a[0], b[1]-a[1]]
    vmbc = [c[0]-b[0],c[1]-b[1]]
    return lineIntersection([vmab,[vmab[1], -vmab[0]]], [vmbc,[vmbc[1], -vmbc[0]]]) 

def inCircle(a,b,c,d):
    if inTriangle(d, [a,b,c]) == 1:
        return 1
    center = circumcenter(a,b,c)
    if center == []:
        return
    radio = dist2(center, a)
    distanceD = dist2(center, d)
    if distanceD > radio:
        return -1
    elif distanceD == radio:
        return 0
    else:
        return 1 

def maxAbcisa(P):
    indice = 0
    for i in range(1,len(P)):
        if P[i][0] > P[indice][0] or (P[i][0] == P[indice][0] and P[i][1] > P[indice][1]):
            indice = i
    return indice

def maxOrdenada(P):
    indice = 0
    for i in range(1,len(P)):
        if P[i][1] > P[indice][1] or (P[i][1] == P[indice][1] and P[i][0] > P[indice][0]):
            indice = i
    return indice

def minAbcisa(P):
    indice = 0
    for i in range(1,len(P)):
        if P[i][0] < P[indice][0] or (P[i][0] == P[indice][0] and P[i][1] < P[indice][1]):
            indice = i
    return indice

def minOrdenada(P):
    indice = 0
    for i in range(1,len(P)):
        if P[i][1] < P[indice][1] or (P[i][1] == P[indice][1] and P[i][0] < P[indice][0]):
            indice = i
    return indice

def prodEscalar(u,v):
    if len(u) != len(v):
        return "VECTORES DEL MISMO TAMAÑO"
    prod = 0
    for i in range(len(u)):
        prod += u[i]*v[i]
    return prod

def maxDir(v,P):
    indice = 0
    dist = prodEscalar(v, P[0])
    for i in range(1,len(P)):
        tempProd = prodEscalar(v, P[i])
        if tempProd > dist:
            indice = i
            dist = tempProd
    return indice

def minDir(v,P):
    indice = 0
    dist = prodEscalar(v, P[0])
    for i in range(1,len(P)):
        tempProd = prodEscalar(v, P[i])
        if tempProd < dist:
            indice = i
            dist = tempProd
    return indice

def maxAngleVectorWithX(P):
    indice = 0
    for i in range(1,len(P)):
        if (P[i][0] == 0 and P[indice][1] < P[i][1]) or P[i][0] < P[indice][0]:
            indice = i
    return indice

def minAngleVectorWithX(P):
    indice = 0
    for i in range(1,len(P)):
        if (P[i][1] == 0 and P[indice][0] < P[i][0]) or P[i][1] < P[indice][1]:
            indice = i
    return indice

def splitHorizList(P):
    def vertSort(x,y):
        if x[1] > y[1]:
            return int(1)
        elif x[1] == y[1] and x[0] > y[0]:
            return int(1)
        elif x == y:
            return int(0)
        elif x[1] == y[1] and x[0] < y[0]:
            return int(-1)
        else:
            return int(-1)

    def vertSortWithMaxOrdenada(x,y):
        if maxOrdenada([x,y]) == 0:
            return int(1)
        else:
            return int(-1)

    sortedList = sorted(P, cmp = vertSort)
    medPoint = int(len(sortedList)/2)
    subList = 0
    list = [[],[]]
    for i in range(len(sortedList)):
        if i == medPoint:
            subList = 1
        list[subList].append(sortedList[i])
    return list

def splitVerticList(P):
    def vertSortWithMaxAbcisa(x,y):
        if maxAbcisa([x,y]) == 0:
            return int(1)
        else:
            return int(-1)

    sortedList = sorted(P, cmp = vertSortWithMaxAbcisa)
    medPoint = int(len(sortedList)/2)
    subList = 0
    list = [[],[]]
    for i in range(len(sortedList)):
        if i == medPoint:
            subList = 1
        list[subList].append(sortedList[i])
    return list

def angularSort(l, p):
    def compare(x,y):
        if x == y:
            return int(0)
        elif x[1] >= p[1] > y[1]:
            return int(-1)
        elif y[1] >= p[1] > x[1]:
            return int(1)
        orient = orientation(p,x,y)
        if orient == 1:
            return int(-1)
        elif orient == -1:
            return int(1)
        elif x[0] > p[0] > y[0]:
            return int(1)
        elif y[0] > p[0] > x[0]:
            return int(-1)
        distanceX = dist2(p,x)
        distanceY = dist2(p,y)
        if distanceX > distanceY:
            return int(-1)
        elif distanceY > distanceX:
            return int(1)
        else: 
            return int(1) if x == p else int(-1)
        
    return sorted(l,cmp = compare)

def splitAnyDirectionList(v,P):
    perpendicularV = [-v[1],v[0]]

    def sortInAnyDirection(x,y):
        if maxDir(perpendicularV,[x,y]) == 0:
            return int(1)
        else:
            return int(-1)

    sortedList = sorted(P, cmp = sortInAnyDirection)
    medPoint = int(len(sortedList)/2)
    subList = 0
    list = [[],[]]
    for i in range(len(sortedList)):
        if i == medPoint:
            subList = 1
        list[subList].append(sortedList[i])
    return list

def boundingBox(P):
    maxRightIndex = maxAbcisa(P); maxRight = P[maxRightIndex]
    maxLeftIndex = minAbcisa(P); maxLeft = P[maxLeftIndex]
    maxUpIndex = maxOrdenada(P); maxUp = P[maxUpIndex]
    maxDownIndex = minOrdenada(P); maxDown = P[maxDownIndex]
    
    rightLine = [maxRight, [maxRight[0], maxRight[1] + 1]]
    leftLine = [maxLeft, [maxLeft[0], maxLeft[1] + 1]]
    topLine = [maxUp, [maxUp[0] + 1, maxUp[1]]]
    botLine = [maxDown, [maxDown[0] + 1, maxDown[1]]]
    return [lineIntersection(rightLine, topLine), lineIntersection(topLine, leftLine), lineIntersection(leftLine, botLine), lineIntersection(botLine, rightLine)]
