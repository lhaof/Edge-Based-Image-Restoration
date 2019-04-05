import numpy as np
import math
from scipy import optimize

def AngleDiff(alpha_1, alpha_2):
    ## compute directed angle difference, range of [-pi, pi]
    if abs(alpha_1-alpha_2)<=math.pi:
        return alpha_1-alpha_2
    else:
        signflag = 1.0
        if alpha_1-alpha_2<0:
            signflag = -1.0
        return -1.0*signflag*(math.pi*2.0 - abs(alpha_1-alpha_2))

def PointOnArc(p, Cir, h1, t1, h2, t2):
    ## tell if p on h1-h2, an arc of Cir
    ## t1-h1 and t2-h2 point to the arc h1-h2
    q = Cir[0]
    alpha_p = math.atan2(p[1]-q[1],p[0]-q[0])
    alpha_h1 = math.atan2(h1[1]-q[1],h1[0]-q[0])
    alpha_h2 = math.atan2(h2[1]-q[1],h2[0]-q[0])
    alpha_h1p = AngleDiff(alpha_h1,alpha_p)
    alpha_h2p = AngleDiff(alpha_h2,alpha_p)
    qp = p-q
    if abs(alpha_h1p)<abs(alpha_h2p):
        alpha_t1 = math.atan2(t1[1]-q[1],t1[0]-q[0])
        alpha_t1p = AngleDiff(alpha_t1,alpha_p)
        qh1 = h1-q
        qt1 = t1-q
        crosspro = np.cross(qp,qh1)*np.cross(qp,qt1)
        if abs(alpha_h1p)<abs(alpha_t1p) and crosspro>=0:
            return True
    else:
        alpha_t2 = math.atan2(t2[1]-q[1],t2[0]-q[0])
        alpha_t2p = AngleDiff(alpha_t2,alpha_p)
        qh2 = h2-q
        qt2 = t2-q
        crosspro = np.cross(qp,qh2)*np.cross(qp,qt2)
        if abs(alpha_h2p)<abs(alpha_t2p) and crosspro>=0:
            return True
    return False

def Point2Arc(p, Cir, h1, t1, h2, t2):
    q = Cir[0]
    r = Cir[1]
    qp = p-q
    qt1 = t1-q
    qt2 = t2-q
    alpha_qp = math.atan2(qp[1],qp[0])
    alpha_qt1 = math.atan2(qt1[1],qt1[0])
    alpha_qt2 = math.atan2(qt2[1],qt2[0])
    alpha_t1_p = AngleDiff(alpha_qt1, alpha_qp)
    alpha_t2_p = AngleDiff(alpha_qt2, alpha_qp)
    withinArc = False
    if abs(alpha_t1_p)<abs(alpha_t2_p):
        qh1 = h1-q
        alpha_qh1 = math.atan2(qh1[1],qh1[0])
        alpha_h1_p = AngleDiff(alpha_qh1, alpha_qp)
        sameside_h1t1 = np.cross(qp,qh1)*np.cross(qp,qt1)
        if abs(alpha_t1_p)>abs(alpha_h1_p) and sameside_h1t1>0:
            withinArc = True
    else:
        qh2 = h2-q
        alpha_qh2 = math.atan2(qh2[1],qh2[0])
        alpha_h2_p = AngleDiff(alpha_qh2, alpha_qp)
        sameside_h2t2 = np.cross(qp,qh2)*np.cross(qp,qt2)
        if abs(alpha_t2_p)>abs(alpha_h2_p) and sameside_h2t2>0:
            withinArc = True
    d_qp = math.sqrt(float(np.sum(np.power(qp,2))))
    if withinArc and d_qp>1e-6:
        d_parc = abs(d_qp-r)
        t = r/d_qp
        inter = q + qp*t
        return inter, d_parc
    else:
        d_h1p = math.sqrt(float(np.sum(np.power(h1-p,2))))
        d_h2p = math.sqrt(float(np.sum(np.power(h2-p,2))))
        if d_h1p < d_h2p:
            return h1, d_h1p
        else:
            return h2, d_h2p

def Point2LineSegment(p, lineseg):
    q1 = lineseg[0]
    q2 = lineseg[1]
    d_q1q2 = math.sqrt(float(np.sum(np.power(q2-q1,2))))
    if d_q1q2>1e-6 and np.dot(p-q1)*np.dot(q2-q1) > 0 and np.dot(p-q2)*np.dot(q1-q2) > 0:
        d_p2L = abs(np.cross(q2-q1,p-q1)) / d_q1q2
        t = np.dot(p-q1,p2-q1) / d_q1q2 / d_q1q2
        proj = q1 + (q2-q1)*t
        return proj, d_p2L
    else:
        d_q1p = math.sqrt(float(np.sum(np.power(p-q1,2))))
        d_q2p = math.sqrt(float(np.sum(np.power(p-q2,2))))
        if d_q1p<d_q2p:
            return q1,d_q1p
        else:
            return q2,d_q2p

def CircleFitting(x, y):

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt(np.square(x-xc) + np.square(y-yc))
    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    def f_1(c):
        Ri = calc_R(*c)
        return Ri - 1e4

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center_, ier = optimize.leastsq(f_2, center_estimate)
    xc_, yc_ = center_
    Ri_       = calc_R(*center_)
    R_        = Ri_.mean()
    residu_   = sum((Ri_ - R_)**2)
    if R_>1e4:
        center_, ier = optimize.leastsq(f_1, center_estimate)
        xc_, yc_ = center_
        Ri_       = calc_R(*center_)
        R_        = 1e4
        residu_   = sum((Ri_ - 1e4)**2)
    return xc_, yc_, R_, Ri_

def LineFitting(xdata, ydata):
    ## line fitting
    ## suppose a line is Ax + By + C = 0
    ymean = float(ydata.mean())
    xmean = float(xdata.mean())
    y_ = ydata-ymean
    x_ = xdata-xmean
    a_ = np.sum(np.power(x_,2))-np.sum(np.power(y_,2))
    b_ = np.sum(x_*y_)
    if abs(float(b_)) > 1e-6:
        A = 2.0*b_
        B = -(a_ + math.sqrt(float(a_*a_ + 4.0*b_*b_)))
        print A, B, xmean, ymean
        C =  -(A*xmean + B*ymean)
        assert(abs(B) > 0)
        return A, B, C
        A = A/B
        C = C/B
        return A, 1, C
    else:
        mse1 = np.mean(np.power(x_,2))
        mse2 = np.mean(np.power(y_,2))
        if mse1 < mse2:
            ## choose x = xmean
            return 1, 0, -xmean
        else:
            ## choose y = ymean
            return 0, 1, -ymean

def LineCircleIntersect(line, circle):
    p1 = line[0]
    p2 = line[1]
    q = circle[0]
    r = circle[1]
    v = p2-p1
    a = np.dot(v,v)
    b = 2.0 * np.dot(v,p1-q)
    c = np.dot(p1,p1) + np.dot(q,q) - 2.0 * np.dot(p1,q) - r * r
    disc = b*b - 4.0*a*c
    if disc < 0:
        return False, None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    if not ((t1>=0 and t1<=1) or (t2>=0 and t2<=1)):
        return False, None
    ret = []
    if t1>=0 and t1<=1:
        ret.append(p1 + v*t1)
    if t2>=0 and t2<=1:
        ret.append(p1 + v*t2)
    return True, ret

def LineSegmentArcIntersect(lineseg, Cir, h1, t1, h2, t2):
    state, inter = LineCircleIntersect(lineseg, Cir)
    if not state:
        return False, None, None
    ret = []
    dist = []
    p = lineseg[0]
    for i in inter:
        if PointOnArc(i, Cir, h1, t1, h2, t2):
            d = math.sqrt(float(np.sum(np.power(i-p,2))))
            ret.append(i)
            dist.append(d)
    if len(ret)==0:
        return False, None, None
    elif len(ret)==1 or dist[0]<dist[1]:
        return True, ret[0], dist[0]
    else:
        return True, ret[1], dist[1]

def LineSegmentsIntersect(lineseg1, lineseg2):
    p1 = lineseg1[0]
    p2 = lineseg1[1]
    q1 = lineseg2[0]
    q2 = lineseg2[1]
    ## see if q1,q2 on the same side of line1
    p1p2 = p2-p1
    p1q1 = q1-p1
    p1q2 = q2-p1
    if np.cross(p1p2,p1q1)*np.cross(p1p2,p1q2) > 0:
        return False, None
    ## see if p1,p2 on the same side of line2
    q1q2 = q2-q1
    q1p1 = p1-q1
    q1p2 = p2-q1
    if np.cross(q1q2,q1p1)*np.cross(q1q2,q1p2) > 0:
        return False, None
    area1 = abs(float(np.cross(q1p1,q1q2)))
    area2 = abs(float(np.cross(p1q2,p1p2))) + abs(float(np.cross(p1p2,p1q1)))
    t = area1*1.0/area2
    if t<0 or t>1:
        return False, None
    return True, p1 + p1p2*t

def SortEdge(h0,w0,h_ind,w_ind,segments_map,seg,maxn):
    fxh = [0, 0, 1,-1, 1, 1,-1,-1]
    fxw = [1,-1, 0, 0,-1, 1,-1, 1]
    cnt = 0
    for j in range(h_ind.shape[0]):
        h = h_ind[j]
        w = w_ind[j]
        isboundary = False
        for k in range(len(fxh)):
            h_ = h+fxh[k]
            w_ = w+fxw[k]
            if h_>=0 and h_<segments_map.shape[0] and w_>=0 and w_<segments_map.shape[1]:
                if segments_map[h_,w_]!=seg:
                    isboundary = True
        if not isboundary:
            continue
        mindist = 1e6
        mi = -1
        for k in range(j,h_ind.shape[0]):
            dist = 1e6
            if j==0:
                dist = math.sqrt(math.pow(h_ind[k]-h0,2)+math.pow(w_ind[k]-w0,2))
            else:
                dist = math.sqrt(math.pow(h_ind[k]-h_ind[j-1],2)+math.pow(w_ind[k]-w_ind[j-1],2))
            if dist<mindist or mi==-1:
                mindist = dist
                mi = k
        if j!=mi and mi!=-1 and mindist<3:
            tmp = h_ind[j]
            h_ind[j] = h_ind[mi]
            h_ind[mi] = tmp
            tmp = w_ind[j]
            w_ind[j] = w_ind[mi]
            w_ind[mi] = tmp
        if j+1>=maxn or mindist>=3:
            cnt = j+1
            break
    return cnt, h_ind, w_ind

def DrawCircle(rebuild, Cir, h1,t1,h2,t2, val):
    x0 = round(Cir[0][0])
    y0 = round(Cir[0][1])
    r = round(Cir[1])
    x = r-1
    y = 0
    dx = 1
    dy = 1
    err = dx - (r*2)
    while (x >= y):
        pixels = []
        pixels.append((x0 + x, y0 + y))
        pixels.append((x0 + y, y0 + x))
        pixels.append((x0 - y, y0 + x))
        pixels.append((x0 - x, y0 + y))
        pixels.append((x0 - x, y0 - y))
        pixels.append((x0 - y, y0 - x))
        pixels.append((x0 + y, y0 - x))
        pixels.append((x0 + x, y0 - y))
        for pix in pixels:
            if PointOnArc(np.array(pix),(np.array((x0,y0)),r),h1,t1,h2,t2):
                if pix[0]<0 or pix[0]>=512 or pix[1]<0 or pix[1]>=512:
                    print pix, x0,y0,r,h1,t1,h2,t2
                rebuild[int(pix[0]),int(pix[1]),:] = val
        if (err <= 0):
            y = y+1
            err = err+dy
            dy = dy+2
        if (err > 0):
            x = x-1
            dx = dx+2
            err = err+dx - (r*2)
    return rebuild

def GetCircle(rebuild, Cir, h1,t1,h2,t2):
    ret = []
    x0 = round(Cir[0][0])
    y0 = round(Cir[0][1])
    r = round(Cir[1])
    x = r-1
    y = 0
    dx = 1
    dy = 1
    err = dx - (r*2)
    while (x >= y):
        pixels = []
        pixels.append((x0 + x, y0 + y))
        pixels.append((x0 + y, y0 + x))
        pixels.append((x0 - y, y0 + x))
        pixels.append((x0 - x, y0 + y))
        pixels.append((x0 - x, y0 - y))
        pixels.append((x0 - y, y0 - x))
        pixels.append((x0 + y, y0 - x))
        pixels.append((x0 + x, y0 - y))
        for pix in pixels:
            if PointOnArc(np.array(pix),(np.array((x0,y0)),r),h1,t1,h2,t2):
                #if pix[0]<0 or pix[0]>=rebuild.shape[0] or pix[1]<0 or pix[1]>=rebuild.shape[1]:
                #print pix, x0,y0,r,h1,t1,h2,t2
                #else:
                if pix[0]>=0 and pix[0]<rebuild.shape[0] and pix[1]>=0 and pix[1]<rebuild.shape[1]:
                    ret.append(  np.array((int(pix[0]),int(pix[1])),dtype=np.int32).reshape(1,2) )
        if (err <= 0):
            y = y+1
            err = err+dy
            dy = dy+2
        if (err > 0):
            x = x-1
            dx = dx+2
            err = err+dx - (r*2)
    if len(ret)==0:
        return np.ndarray(shape=(0,1))
    ret = np.concatenate(ret,axis=0)
    ## sorting
    angle = np.ndarray(shape=(len(ret),),dtype=np.float32)
    for i in range(angle.shape[0]):
        angle[i] =math.atan2(ret[i][1]-y0,ret[i][0]-x0)
    ind = np.argsort(angle)
    ret = ret[ind,:]
    cen = np.array((x0,y0),dtype=np.float32).reshape(1,2)
    if np.linalg.norm(cen-ret[0],axis=1) > np.linalg.norm(cen-ret[-1],axis=1):
        ret = np.flip(ret,axis=0)
    return ret

def DrawCircleBackground(rebuild, Cir, h1,t1,h2,t2, val):
    ofsh = [0, 0, 1,-1, 1, 1,-1,-1]
    ofsw = [1,-1, 0, 0,-1, 1,-1, 1]
    pixels = GetCircle(rebuild, Cir, h1,t1,h2,t2)
    end_i = pixels.shape[0]
    for i in range(pixels.shape[0]):
        h = pixels[i][0]
        w = pixels[i][1]
        istouch = False
        for j in range(len(ofsh)):
            h_ = h+ofsh[j]
            w_ = w+ofsw[j]
            if h_<0 or h_>=rebuild.shape[0] or w_<0 or w_>=rebuild.shape[1]:
                continue
            if rebuild[h_,w_,0]!=0:
                istouch = True
                break
        if istouch:
            end_i = i+1
            break
    h_ind = pixels[:end_i,0]
    w_ind = pixels[:end_i,1]
    rebuild[h_ind,w_ind,:] = val
    start_i = 0
    for i in range(pixels.shape[0]-1,-1,-1):
        h = pixels[i][0]
        w = pixels[i][1]
        istouch = False
        for j in range(len(ofsh)):
            h_ = h+ofsh[j]
            w_ = w+ofsw[j]
            if h_<0 or h_>=rebuild.shape[0] or w_<0 or w_>=rebuild.shape[1]:
                continue
            if rebuild[h_,w_,0]!=0:
                istouch = True
                break
        if istouch:
            start_i = i
            break
    if start_i<pixels.shape[0]-1 and start_i>end_i+1:
        h_ind = pixels[start_i:,0]
        w_ind = pixels[start_i:,1]
        rebuild[h_ind,w_ind,:] = val
    return rebuild

def plotLineLow(x0,y0,x1,y1):
    ret = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2*dy - dx
    y = y0
    for x in range(int(x0), int(x1)+1):
        ret.append((int(x),int(y)))
        if D > 0:
            y = y + yi
            D = D - 2*dx
        D = D + 2*dy
    return ret

def plotLineHigh(x0,y0, x1,y1):
    ret = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2*dx - dy
    x = x0
    for y in range(int(y0),int(y1)+1):
        ret.append((int(x),int(y)))
        if D > 0:
            x = x + xi
            D = D - 2*dy
        D = D + 2*dx
    return ret

def DrawLine(x0,y0, x1,y1):
    ret = []
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            ret = plotLineLow(x1, y1, x0, y0)
        else:
            ret = plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            ret = plotLineHigh(x1, y1, x0, y0)
        else:
            ret = plotLineHigh(x0, y0, x1, y1)
    return ret