import random

dimension = 1
numOfElements = 20
predefineRatio = 3
#elementPool = [2147483647, -2147483648, 1, 0, -1, -2, 0x3F3F3F3F]
elementPool = [2147483647, 1, 0x3F3F3F3F]
listRes = []
def buildList():
    if(dimension == 1):
        for x in range(numOfElements):
            rDice = random.randrange(0, 9)
            if(rDice < predefineRatio):
                listRes.append(elementPool[random.randrange(0,2)])
            else:
                listRes.append(random.randrange(0,99))
        random.shuffle(listRes)
    print(listRes)

buildList()
