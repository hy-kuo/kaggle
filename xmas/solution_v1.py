import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

SMALL_SRC = "data/small.csv"
DATA_SRC = "data/gifts.csv"
SUBMISSION = "submission/1.csv"
LO, LA, W = "Longitude", "Latitude", "Weight"
TRIP = "TripId"
GIFT = "GiftId"
ASGN = "Assigned"
MAX_W = 1000


def read_dataset(path):
    ds = pd.read_csv(path)
    return ds


def visualize(ds, group_trip, src):
    fig = plt.figure()
    clusters = len(set(ds[group_trip]))
    print("{1} number {0}".format(clusters, group_trip))
    color = plt.cm.rainbow(np.linspace(0, 1, clusters))
    for i, c in zip(range(clusters), color):
        tds = ds.loc[ds[group_trip] == i]
        plt.scatter(tds[LO], tds[LA], color=c)
        plt.plot(tds[LO], tds[LA], color=c)
        plt.hold(True)
    plt.title(group_trip)
    plt.gca().set_xlabel(LO)
    plt.gca().set_ylabel(LA)
    #plt.show()
    fig.savefig("visual/{0}.svg".format(group_trip))


def bucketSort(ds):
    ds.sort_values(by=[LO, LA], ascending=[True, True])
    Long360 = dict([i, []] for i in range(360)) # sort into 360 longitude bucket
    salt = 180
    for cur in range(360):
        for index, row in ds.iterrows():
            if (row[LO]+salt)%360 >= cur and (row[LO]+salt)%360 < cur+1:
                Long360[cur].append(index)
    return Long360


def genTripId(ds, buckets):
    trips = [-1 for x in range(len(ds))]
    curT = 0
    curLa = 90
    w = 10
    for i in range(360):
        ids = buckets[i]
        for x in ids:
            if (w+ds.ix[x][W]) <= MAX_W and ds.ix[x][LA] <= curLa:
                w += ds.ix[x][W]
                trips[x] = curT
                curLa = ds.ix[x][LA]
            else:
                curT += 1
                trips[x] = curT
                w = 10+ds.ix[x][W]
                curLa = ds.ix[x][LA]
    return trips


def checkTrip(ds):
    if len(ds.loc[ds[TRIP] == -1]) > 0:
        print("Oops, {0} points have no trip assigned!".format(len(trip[trip == -1])))
        return
    n = len(set(ds[TRIP]))
    for i in range(n):
        w = sum(ds[W].ix[ds.loc[ds[TRIP] == i].index.values])
        if w+10 > MAX_W:
            print("Trip {0} Exceed Maximum Weight {1}>{2}!".format(i, w, MAX_W))
            return
    print("Everything's OK")
    return


def main():
    src = DATA_SRC
    ds = read_dataset(src)
    buckets = bucketSort(ds)
    trip = genTripId(ds, buckets)
    ds[TRIP] = trip
    checkTrip(ds)
    # visualize(ds, TRIP, src)
    submission = pd.DataFrame({GIFT: ds[GIFT], TRIP: ds[TRIP]})
    submission.to_csv(SUBMISSION, index=False)
    

if __name__ == '__main__':
    main()
