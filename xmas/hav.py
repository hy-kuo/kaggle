from math import sqrt, asin, sin, cos, pi

def haversine(phi1, th1, phi2, th2):
    "calculate haversine distance between two points"
    r = 100000
    phi1, phi2 = phi1*pi/180, phi2*pi/180
    th1, th2 = th1*pi/180, th2*pi/180
    return 2*r*asin(sqrt(sin((phi1-phi2)/2)**2+cos(phi1)*cos(phi2)*(sin((th1-th2)/2))**2))