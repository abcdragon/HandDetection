import math

def distance(x1, y1, x2, y2):
    temp = ( math.pow(float(x2) - float(x1), 2), math.pow(float(y2) - float(y1), 2) )
    return int(math.sqrt(temp[0] + temp[1]))

def middlePoint(Data):
    x = y = 0
    #print(len(Data))
    for i in range(len(Data)):
        x += Data[i][0]
        y += Data[i][1]

    middle_x, middle_y = x/len(Data), y/len(Data)
    #print(x, middle_x, y, middle_y)
    return int(middle_x), int(middle_y)

def FindMiddlePoint(a):
    SaveData = []
    tempSave = []

    for i in range(len(a) - 1):
        #print(a[i], end=' ')
        #print(a[i + 1])
        tempSaveValue = distance(int(a[i][0]), int(a[i][1]), int(a[i + 1][0]), int(a[i + 1][1]))
        tempLen = len(tempSave)
        if (tempSaveValue <= 50):
            if tempLen == 0:
                tempSave.append([a[i], a[i + 1]])
            else:
                tempSave[0].append(a[i + 1])
        else:
            if tempLen == 0:
                continue
            else:
                SaveData.append(tempSave[0])
                tempSave.clear()

    returnData = []

    for i in range(len(SaveData)):
        #print(SaveData[i])
        returnData.append(middlePoint(SaveData[i]))

    #print(returnData)

    return returnData
