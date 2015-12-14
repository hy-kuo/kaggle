import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

DATA_SRC = "data/gifts.csv"
SUBMISSION = "submission/1.csv"
LO, LA, W = "Longitude", "Latitude", "Weight"
GROUP = "Group"
TRIP = "TripId"
GIFT = "GiftId"
ASGN = "Assigned"
MAX_W = 1000-10


def read_dataset(path):
    ds = pd.read_csv(path)
    return ds
    

def visualize(ds, group_trip):
    fig = plt.figure()
    clusters = len(set(ds[group_trip]))
    print("{1} number {0}".format(clusters, group_trip))
    color = plt.cm.rainbow(np.linspace(0, 1, clusters))
    for i, c in zip(range(clusters), color):
        tds = ds.loc[ds[group_trip] == i]
        plt.scatter(tds[LO], tds[LA], color=c)
        plt.hold(True)
    plt.title(group_trip)
    plt.gca().set_xlabel(LO)
    plt.gca().set_ylabel(LA)
    # plt.show()
    fig.savefig("visual/{0}.png".format(group_trip))


def findInGroup(ds, g, w):
    ds = ds.loc[ds[ASGN] == 0]
    ret = []
    same = ds.loc[ds[GROUP] == g]
    if len(same) == 0:
        return (ret, -1)
    tw = w
    for i in same.index:
        if same.ix[i][W]+tw <= MAX_W:
            tw += same.ix[i][W]
            ret.append(i)
    if len(ret) > 0:
        return (ret, tw)
    return ([], -1)


def getNext(ds, g, w):
    ds = ds.loc[ds[ASGN] == 0]
    min_ele = min(ds[W])
    if min_ele + w > MAX_W:
        return (-1, -1, MAX_W+1)
    ret = []
    # find in the same group first
    (ret, tw) = findInGroup(ds, g, w)
    if tw > 0:
        return (ret, g, tw)
    # find in other group
    for i in set(ds[GROUP]):
        (rt, tw) = findInGroup(ds, i, w)
        if tw > 0:
            return (rt, i, tw)
    return (-1, -1, MAX_W+1)


def genTripId(ds): # heuristic
    n, tn, g, w, curT = len(ds), 0, 0, 0, []
    trip = np.zeros(len(ds), dtype=np.int)
    ds[ASGN] = np.zeros(len(ds), dtype=np.int)
    while n > 0:
        print("trip {0} - weight: {1}, group: {2}".format(tn, w, g))
        nxt, g, tw = getNext(ds, g, w)
        if tw+w <= MAX_W:
            w += tw
            curT.append(nxt)
            for x in nxt:
                ds[ASGN].iloc[x] = 1
            n -= len(nxt)
        elif len(curT) > 0:
            w = 0
            for x in curT:
                trip[x] = tn
            tn += 1
            curT = []
        else:
            w = 0
            curT = []
    return trip


def checkTrip(trip):
    n = len(set(trip))
    for i in range(n):
        w = sum(trip[trip == i])
        if w > MAX_W:
            print("Exceed Maximum Weight")
            return
    print("Everything's OK")
    return


def main():
    ds = read_dataset(DATA_SRC)
    db = DBSCAN(min_samples=5).fit(ds[[LO, LA]])
    ds[GROUP] = db.labels_
    visualize(ds, GROUP)
    trip = genTripId(ds)
    checkTrip(trip)
    ds[TRIP] = trip
    visualize(ds, TRIP)
    submission = pd.DataFrame({GIFT: ds[GIFT], TRIP: ds[TRIP]})
    submission.to_csv(SUBMISSION, index=False)
    

if __name__ == '__main__':
    main()
