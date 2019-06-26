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
    if P[0] >= s[maxX-1][0] and P[0] <= s[maxX][0] and P[1] >= s[maxY-1][1] and P[1] <= s[maxY][1]:
        return True if sarea(s[0],s[1],P) == 0 else False
    else:
        return False

def inSegmentNumbers(P,s):
    maxX = 0 if s[0][0] >= s[1][0] else 1
    maxY = 0 if s[0][1] >= s[1][1] else 1
    if P[0] >= s[maxX-1][0] and P[0] <= s[maxX][0] and P[1] >= s[maxY-1][1] and P[1] <= s[maxY][1]:
        return 1 if sarea(s[0],s[1],P) == 0 else 0
    else:
        return 0

def inTriangle(P,t):
    orion = sum(orientation(t[i-1], t[i], P) for i in range(len(t)))
    if orion == 3 or orion == -3:
        return True
    else:
        i = 0
        inSeg = 0
        while i < len(t) and not inSeg:
            inSeg = inSegment(P,[t[i-1],t[i]])
            i+=1
        return inSeg

def inTriangleNumbers(P,t):
    orion = 0
    for i in range(len(t)):
       orion += orientation(t[i-1], t[i], P)
    if orion == 3 or orion == -3:
        return 1
    else:
        i = 0
        inSeg = 0
        while i < len(t) and inSeg != 1:
            inSeg = inSegmentNumbers(P,[t[i-1],t[i]])
            i+=1
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
        return []
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
    vectorAB = [b[0]-a[0], b[1]-a[1]]
    vectorBC = [c[0]-b[0], c[1]-b[1]]
    perpAB = [vectorAB[1], -vectorAB[0]]
    perpBC = [vectorBC[1], -vectorBC[0]]
    pointInDirectionAB = [mab[0] + perpAB[0], mab[1] + perpAB[1]]
    pointInDirectionBC = [mbc[0] + perpBC[0], mbc[1] + perpBC[1]]
    return lineIntersection([mab, pointInDirectionAB], [mbc, pointInDirectionBC])

def inCircle(a,b,c,d):
    if inTriangle(d, [a,b,c]):
        return 1
    center = circumcenter(a,b,c)
    if center == []:
        print("NO CENTER")
        return -2
    radio = dist2(center, a)
    distanceD = dist2(center, d)
    if distanceD > radio:
        return -1
    elif distanceD == radio:
        return 0
    else:
        return 1

def inCircleTest(a,b,c,d):
    return inCircle(a,b,c,d) > 0

def generatePoints(n):
    return [[random(), random()] for i in range(n)]

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

def polygonization(p):
    def greaterX(x,y):
        if x[0] > y[0]:
            return int(1)
        elif x[0] == y[0] and x[1] > y[1]:
            return int(1)
        else:
            return int(-1)

    distributedPoints = splitHorizList(p)
    orderedList1 = sorted(distributedPoints[0], cmp= greaterX)
    orderedList2 = sorted(distributedPoints[1], cmp= greaterX)
    orderedList2.reverse()
    return orderedList1 + orderedList2

def starPolygonization(p):
    if len(p) < 3:
        return
    rotationalCenter = midPoint(p[0], p[1])
    #rotationalCenter = circumcenter(p[0], p[1], p[2])
    return angularSort(p, rotationalCenter)

def minPolygonization(p):
    if len(p) < 3:
        return
    rotationalCenterIndex = minAbcisa(p)
    return angularSort(p, p[rotationalCenterIndex])

def clipping(P,r):
    result = []
    if P == []:
        return result
    list = deepcopy(P) + [P[0]]
    capturingPolygon = sarea(r[0], r[1], P[0]) >= 0
    for i in range(len(list)):
        if capturingPolygon:
            capturedArea = sarea(r[0], r[1], list[i])
            if capturedArea >= 0:
                result.append(list[i])
            elif capturedArea < 0:
                # We've moved outside the new polygon.
                capturingPolygon = False
                intersection = lineIntersection(r,[list[i-1], list[i]])
                result.append(intersection)
        else:
            capturedArea = sarea(r[0], r[1], list[i])
            if capturedArea > 0:
                # We've moved inside the new polygon.
                capturingPolygon = True
                intersection = lineIntersection(r,[list[i-1], list[i]])
                result.append(intersection)
                result.append(list[i])
            elif capturedArea == 0:
                result.append(list[i])
                #No mode switching because maybe next time we'll move outside the polygon.
    return result

def kernel(p):
    C=deepcopy(p)
    for i in range(len(p)):
        C=clipping(C,[p[i-1],p[i]])
    return C

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

def convexHullPointsBF(P):
    L = deepcopy(P)
    for triangle in Combinations(P,3):
        elim = []
        for point in L:
            if point not in triangle and inTriangle(point, triangle):
                elim.append(point)
        L = [x for x in L if x not in elim]
    return angularSort(L, midPoint(L[0], L[1]))

def convexHullEdgesBF(P):
    L = deepcopy(P)
    result = []
    for edge in Combinations(P, 2):
        i = 1
        filteredList = [x for x in L if x not in edge]
        orien = orientation(edge[0], edge[1], filteredList[0])
        allValid = True
        while i < len(filteredList) and allValid:
            allValid = orien != 0 and orien == orientation(edge[0], edge[1], filteredList[i])
            orien = orientation(edge[0], edge[1], filteredList[i])
            i+= 1
        if allValid:
            if edge[0] not in result:
                result.append(edge[0])
            if edge[1] not in result:
                result.append(edge[1])
    return angularSort(result, midPoint(L[0], L[1]))

def graham(P):
    minPoint = min(P)
    L = angularSort(P, minPoint)
    i = 0
    while i < len(L):
        orien = orientation(L[i-2], L[i-1], L[i])
        if orien < 0:
            L.pop(i-1)
            i -= 1
        else:
            i+=1
    return L

def jarvis(P):
    initial = min(P)
    hull = [initial]
    nextOneIsInitial = False
    lastHullPoint = initial
    while not nextOneIsInitial:
        pivot = P[0] if lastHullPoint != P[0] else P[1]
        for point in P:
            if sarea(lastHullPoint, pivot, point) < 0:
                pivot = point
        lastHullPoint = pivot
        nextOneIsInitial = lastHullPoint == initial
        if not nextOneIsInitial:
            hull.append(pivot)
    return hull

# funcion que crea un DCEL, para un poligono
def dcel(P):
    n=len(P)
    V=[[P[i],i] for i in range(len(P))]
    e=[[i,n+i,(i-1)%n,(i+1)%n,1]for i in range(n)]+[[(i+1)%n,i,n+(i+1)%n,n+(i-1)%n,0]for i in range(n)]
    f=[n,0]
    return [V,e,f]

# funciones para referirse a los elementos asociados a un elemento del DCEL

# indice del origen de una arista e
def origin(e,D):
    return D[1][e][0]

# coordenadas del origen de la arista e
def originCoords(e,D):
    return D[0][origin(e,D)][0]

# arista gemela de la arista e
def twin(e,D):
    return D[1][e][1]

# arista previa de la arista e
def prev(e,D):
    return D[1][e][2]

# arista siguiente de la arista e
def next(e,D):
    return D[1][e][3]

# indice de la cara de cuyo borde forma parte la arista e
def face(e,D):
    return D[1][e][4]

# indice de una de las aristas del borde de la cara c
def edge(c,D):
    return D[2][c]

# funcion para dibujar las aristas de un DCEL

def plotDCEL(D):
    return sum(line([originCoords(i,D),originCoords(twin(i,D),D)],aspect_ratio=1) for i in range(len(D[1])))

# funcion para colorear una cara de un DCEL

def plotFace(c,D,col):
    f=D[2][c]
    C=[f]
    f=next(f,D)
    while f <> C[0]:
        C.append(f)
        f=next(f,D)

    P=[originCoords(j,D) for j in C]
    return polygon(P,color=col, alpha=.5)

# funcion para colorear las caras de un DCEL
def colorDCEL(D):
    return sum(plotFace(i,D,(random(),random(),random())) for i in range(1,len(D[2])))

# funcion para dividir una cara del DCEL D por una diagonal
# e1 y e2 son las aristas cuyos orígenes son los extremos de la diagonal que divide la cara
def splitFace(e1,e2,D):
    # si no son aristas de la misma cara o si son adyacentes sus origenes no definen una diagonal
    if face(e1,D) <> face(e2,D) or origin(e2,D) == origin(twin(e1,D),D) or origin(e1,D) == origin(twin(e2,D),D):
        print "no diagonal"
        return

    nv, ne, nf = len(D[0]), len(D[1]), len(D[2])
    preve1 = prev(e1,D)
    preve2 = prev(e2,D)
    k=face(e1,D)

    # añadimos las aristas nuevas
    D[1].append([origin(e1,D),ne+1,preve1,e2,k])
    D[1].append([origin(e2,D),ne,preve2,e1,nf])

    # modificamos aristas afectadas
    D[1][preve1][3]=ne
    D[1][e1][2]=ne+1
    D[1][preve2][3]=ne+1
    D[1][e2][2]=ne
    i=e1
    while i<>ne+1:
        D[1][i][4]=nf
        i=next(i,D)

    #modificamos la cara afectada
    D[2][k]=ne

    # añadimos la nueva cara
    D[2].append(ne+1)

def earTestDCEL(edgesList, edge, D):
    n = len(edgesList)
    if sarea(originCoords(prev(edge, D), D), originCoords(edge, D), originCoords(next(edge, D), D)) <= 0:
        return False

    notInTriangle = True
    j = 0
    while notInTriangle and j < n:
        if (edgesList[j] == prev(edge, D)):
            j += 2
        elif edgesList[j] == edge:
            j += 1
        else:
            notInTriangle = not inTriangle(originCoords(edgesList[j], D), [originCoords(prev(edge, D), D), originCoords(edge, D), originCoords(next(edge, D), D)])

        j += 1
    return notInTriangle

# funcion para buscar una diagonal en una cara acotada de un DCEL

def diagonalDCEL(c,D):
    edgesList = faceEdges(c, D)

    if len(edgesList) == 3:
        return []

    actualEdge = edge(c, D)

    n = len(edgesList)
    chosenFirstEdge = -1
    chosenSecondEdge = -1
    chosenArea = oo

    i = 0
    while i < n and chosenFirstEdge == chosenSecondEdge:
        point1 = originCoords(prev(actualEdge, D), D)
        point2 = originCoords(actualEdge, D)
        point3 = originCoords(next(actualEdge, D), D)

        if sarea(point1, point2, point3) > 0:
            if earTestDCEL(edgesList, actualEdge, D):
                chosenFirstEdge = prev(actualEdge, D)
                chosenSecondEdge = next(actualEdge, D)
                continue
            for j in range(n):
                if edgesList[j] in [prev(actualEdge, D), actualEdge, next(actualEdge, D)]:
                    continue
                calculatedArea = sarea(point1, point3, originCoords(next(edgesList[j], D), D))
                if calculatedArea > 0 and chosenArea > calculatedArea:
                    chosenFirstEdge = actualEdge
                    chosenSecondEdge = edgesList[j]
                    chosenArea = calculatedArea
                elif calculatedArea < chosenArea:
                    chosenFirstEdge = actualEdge
                    chosenSecondEdge = edgesList[j]
                    chosenArea = calculatedArea
        i+=1
        actualEdge = next(actualEdge, D)

    return [chosenFirstEdge, chosenSecondEdge]

# funcion para triangular las caras de un DCEL
def triangulateDCEL(D):
    i = 0
    while i < len(D[2]):
        possibleDiagonal = diagonalDCEL(i, D)
        if possibleDiagonal == [] or possibleDiagonal == [-1, -1]:
            i+=1
        else:
            splitFace(possibleDiagonal[0], possibleDiagonal[1], D)

def simpleTriangulateDCEL(D):
    indexMin = minAbcisa(faceVerticesCoords(len(D[2])-1, D))
    firstEdge = D[0][indexMin][1]
    i = len(D[0]) - 1
    while i > 0:
        second = next(next(firstEdge, D), D)
        if D[1][second][0] <> D[1][D[1][firstEdge][1]][0] and D[1][firstEdge][0] <> D[1][D[1][second][1]][0]:
            splitFace(second, firstEdge, D)
        firstEdge = D[2][-1]
        i-=1

def faceEdges(c, D):
    last = edge(c,D)
    list = []
    while last not in list:
        list.append(last)
        last = next(last,D)
    return list

def faceVertices(c, D):
    vertices = []
    for edge in faceEdges(c, D):
        vertices.append(origin(edge, D))
    return vertices

def faceVerticesCoords(c,D):
    vertices = []
    for edge in faceEdges(c, D):
        vertices.append(originCoords(edge, D))
    return vertices

def faceNeighbors(c,D):
    faces = []
    for edge in faceEdges(c, D):
        newFace = face(twin(edge, D), D)
        if newFace not in faces:
            faces.append(newFace)
    return faces

def vertexEdgesPoint(v,D):
    vertexList = D[0]

    # Search for vertex edge.
    searching = true
    firstEdge = None
    i = 0
    while (i < len(vertexList)) and searching:
        if vertexList[i][0] == v:
            firstEdge = vertexList[i][1]
            searching = false
        i+=1

    if firstEdge == None: return []

    lastEdge = twin(prev(firstEdge, D), D)
    twins = [lastEdge]
    while lastEdge != firstEdge:
        lastEdge = twin(prev(lastEdge, D), D)
        twins.append(lastEdge)

    return twins

def vertexEdges(v,D):
    firstEdge = D[0][v][1]
    lastEdge = twin(prev(firstEdge, D), D)
    twins = [firstEdge]
    while lastEdge != firstEdge:
        twins.append(lastEdge)
        lastEdge = twin(prev(lastEdge, D), D)

    return twins

def vertexFanPoint(v,D):
    vertexList = D[0]

    # Search for vertex edge.
    searching = true
    firstEdge = None
    i = 0
    while (i < len(vertexList)) and searching:
        if vertexList[i][0] == v:
            firstEdge = vertexList[i][1]
            searching = false
        i+=1

    lastEdge = twin(prev(firstEdge, D), D)
    twins = [face(lastEdge, D)]
    while lastEdge != firstEdge:
        lastEdge = twin(prev(lastEdge, D), D)
        twins.append(face(lastEdge, D))
    return twins

def vertexFan(v,D):
    firstEdge = D[0][v][1]
    lastEdge = twin(prev(firstEdge, D), D)
    twins = [face(lastEdge, D)]
    while lastEdge != firstEdge:
        lastEdge = twin(prev(lastEdge, D), D)
        twins.append(face(lastEdge, D))

    return twins

def convexHullDCEL(D):
    actualEdge = edge(0, D)
    first = next(actualEdge, D)
    second = next(first, D)
    lonExtFace = len(D[0])
    i = 0
    while i <= lonExtFace:
        orient = orientation(originCoords(actualEdge, D), originCoords(first, D), originCoords(second, D))
        if orient > 0:
            splitFace(actualEdge, second, D)
            actualEdge = prev(len(D[1])-2, D)
            first = next(actualEdge, D)
            second = next(first, D)
            i -= 1
            lonExtFace -= 1
        else:
            actualEdge = first
            first = second
            second = next(first, D)
            i += 1

def triangulation(p):
    dcel3 = dcel(minPolygonization(p))
    print("creado")
    simpleTriangulateDCEL(dcel3)
    print("triangulado")
    convexHullDCEL(dcel3)
    print("cerrando")
    return dcel3

def flip(a,D):
    oa=D[1][a][0]
    ga=D[1][a][1]
    ca=D[1][a][4]
    aa=D[1][a][2]
    pa=D[1][a][3]
    cb=D[1][ga][4]
    ab=D[1][ga][2]
    pb=D[1][ga][3]
    oga=D[1][ga][0]
    D[1][a]=[D[1][aa][0],ga,pa,ab,ca]
    D[1][ga]=[D[1][ab][0],a,pb,aa,cb]
    D[1][pa][2]=ab
    D[1][pa][3]=a
    D[1][aa][2]=ga
    D[1][aa][3]=pb
    D[1][aa][4]=cb
    D[1][pb][2]=aa
    D[1][pb][3]=ga
    D[1][ab][2]=a
    D[1][ab][3]=pa
    D[1][ab][4]=ca
    D[2][ca]=a
    D[2][cb]=ga
    D[0][oa][1]=pb
    D[0][oga][1]=pa

def flipable(e, D):
    return face(e, D) <> 0 and face(twin(e, D), D) <> 0

def isLegal(e, D):
    firstVertexEdge = e
    secondVertexEdge = twin(e, D)
    externalEdge1 = prev(firstVertexEdge, D)
    externalEdge2 = prev(secondVertexEdge, D)
    return not inCircleTest(originCoords(firstVertexEdge,D), originCoords(secondVertexEdge,D), originCoords(externalEdge1,D), originCoords(externalEdge2,D))

def legalize(T):
    i = 0
    while i < len(T[1]):
        if flipable(i, T) and not isLegal(i, T):
            flip(i, T)
            i = 0
        else:
            i+= 1

def delone(p):
    newDcel = triangulation(p)
    legalize(newDcel)
    return newDcel

def voronoiRegion(v,D):
    centers = []
    vertexEdgesList = vertexEdges(v, D)
    for selectedEdge in vertexEdgesList:
        if face(selectedEdge, D) == 0:
            newVertex = getNewPoint(selectedEdge, True, D)
            centers.append(newVertex)
        elif face(twin(selectedEdge, D), D) == 0:
            newVertex = getNewPoint(selectedEdge, False, D)
            centers.append(newVertex)
            centers.append(circumcenter(originCoords(selectedEdge, D), originCoords(prev(selectedEdge,D), D), originCoords(next(selectedEdge, D), D)))
        else:
            centers.append(circumcenter(originCoords(selectedEdge, D), originCoords(prev(selectedEdge, D), D), originCoords(next(selectedEdge, D), D)))
    return centers

def voronoi(p):
    newDcel = delone(p)
    lines = []
    for point in range(len(newDcel[0])):
        lines.append(voronoiRegion(point, newDcel))
    return lines
