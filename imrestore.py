import numpy as np
from PIL import Image
from skimage import measure
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.io import imread, imsave
from scipy.misc import imresize
import scipy.io as sio
import time
import math
import matlab.engine
eng = matlab.engine.start_matlab()
from util import AngleDiff, PointOnArc, Point2Arc, Point2LineSegment, CircleFitting, \
LineFitting, LineCircleIntersect, LineSegmentArcIntersect, LineSegmentsIntersect, SortEdge, \
DrawCircleBackground, DrawCircle, DrawLine, GetCircle
from progressbar import ProgressBar

def imrestore_gray(imname, channel):
	pbar = ProgressBar()
    ofs = 128
    boxminh = ofs
    boxmaxh = ofs+255
    boxminw = ofs
    boxmaxw = ofs+255
    fxh = [0, 0, 1,-1, 1, 1,-1,-1]
    fxw = [1,-1, 0, 0,-1, 1,-1, 1]
    maxnpix = 20
    img = imread(imname)
    if len(img.shape)==3:
        img = np.squeeze(img[:,:,channel])
    img = imresize(img, (512,512))

    ################
    ##   upside   ##
    ################

    img_up = img[:boxminh,:]
    imgdisp = Image.fromarray(img_up,'L')
    display(imgdisp)

    imname_up = imname[:-4]+'_up'+imname[-4:]
    imsave(imname_up,img_up)
    eng.sobelwatershed(imname_up)
    matname_up = imname_up[:-3]+'mat'
    segments_up = sio.loadmat(matname_up)
    segments_up = segments_up['label']

    img_up = img_up.astype(np.float32)
    img_up = img_up / 255.0

    maxsid = np.max(segments_up)
    segmap_up = np.zeros(shape=(img_up.shape[0],img_up.shape[1],3), dtype=np.uint8)
    for i in range(maxsid):
        h_ind, w_ind = np.where(segments_up==i)
        segmap_up[h_ind,w_ind,:] = np.random.choice(256,3)
    imgdisp = Image.fromarray(segmap_up,'RGB')
    display(imgdisp)

    ## segment to edge

    segarr_up = segments_up[-1,boxminw-1:boxmaxw+2]
    segarr_up_uni, segidx = np.unique(segarr_up,return_index=True)
    segidx = np.sort(segidx)
    segarr_up_uni = segarr_up[segidx]
    boundary_up = np.zeros(shape=segments_up.shape,dtype=np.bool)
    segbound_hdict = {}
    segbound_wdict = {}
    for i in range(segarr_up_uni.shape[0]):
        si = segarr_up_uni[i]
        segmask = np.zeros(shape=segments_up.shape,dtype=np.bool)
        segmask[np.where(segments_up==si)] = True
        segboundary = find_boundaries(segmask,1,'inner')
        h_ind,w_ind = np.where(segboundary==True)
        boundary_up[h_ind,w_ind] = True
        segbound_hdict[si] = h_ind
        segbound_wdict[si] = w_ind

    edgelist_up = []
    for i in range(boxminw-1,boxmaxw+1):
        si = segments_up[-1,i]
        si_ = segments_up[-1,i+1]
        if si!=si_:
            h_ind = segbound_hdict[si].copy()
            w_ind = segbound_wdict[si].copy()
            h0 = segments_up.shape[0]-1
            w0 = i
            cnt,h_ind,w_ind = SortEdge(h0,w0,h_ind,w_ind,segments_up,si,maxnpix)
            if cnt>0:
                h_ind = h_ind[:cnt]
                w_ind = w_ind[:cnt]
                edgelist_up.append((h_ind,w_ind))

    edgemap_up = np.zeros(shape=segmap_up.shape,dtype=np.uint8)
    for i in range(len(edgelist_up)):
        h_ind = edgelist_up[i][0]
        w_ind = edgelist_up[i][1]
        edgemap_up[h_ind,w_ind,:] = 255
    imgdisp = Image.fromarray(edgemap_up,'RGB')
    display(imgdisp)

    ##################
    ##   downside   ##
    ##################

    img_down = img[-boxminh:,:]
    imgdisp = Image.fromarray(img_down,'L')
    display(imgdisp)

    imname_down = imname[:-4]+'_down'+imname[-4:]
    imsave(imname_down,img_down)
    eng.sobelwatershed(imname_down)
    matname_down = imname_down[:-3]+'mat'
    segments_down = sio.loadmat(matname_down)
    segments_down = segments_down['label']

    img_down = img_down.astype(np.float32)
    img_down = img_down / 255.0
    maxsid_down = np.max(segments_down)

    segmap_down = np.zeros(shape=(img_down.shape[0],img_down.shape[1],3), dtype=np.uint8)
    for i in range(maxsid_down):
        h_ind, w_ind = np.where(segments_down==i)
        segmap_down[h_ind,w_ind,:] = np.random.choice(256,3)
    imgdisp = Image.fromarray(segmap_down,'RGB')
    display(imgdisp)

    ## segment to edge

    segarr_down = segments_down[0,boxminw-1:boxmaxw+2]
    segarr_down_uni, segidx = np.unique(segarr_down,return_index=True)
    segidx = np.sort(segidx)
    segarr_down_uni = segarr_down[segidx]
    boundary_down = np.zeros(shape=segments_down.shape,dtype=np.bool)
    bound_hdict_down = {}
    bound_wdict_down = {}
    for i in range(segarr_down_uni.shape[0]):
        si = segarr_down_uni[i]
        segmask = np.zeros(shape=segments_down.shape,dtype=np.bool)
        segmask[np.where(segments_down==si)] = True
        segboundary = find_boundaries(segmask,1,'inner')
        h_ind,w_ind = np.where(segboundary==True)
        boundary_down[h_ind,w_ind] = True
        bound_hdict_down[si] = h_ind
        bound_wdict_down[si] = w_ind

    edgelist_down = []
    for i in range(boxmaxw+1,boxminw-1,-1):
        si = segments_down[0,i]
        si_ = segments_down[0,i-1]
        if si!=si_:
            h_ind = bound_hdict_down[si].copy()
            w_ind = bound_wdict_down[si].copy()
            h0 = 0
            w0 = i
            cnt,h_ind,w_ind = SortEdge(h0,w0,h_ind,w_ind,segments_down,si,maxnpix)
            if cnt>0:
                h_ind = h_ind[:cnt]
                w_ind = w_ind[:cnt]
                edgelist_down.append((h_ind,w_ind))

    edgemap_down = np.zeros(shape=segmap_down.shape,dtype=np.uint8)
    for i in range(len(edgelist_down)):
        h_ind = edgelist_down[i][0]
        w_ind = edgelist_down[i][1]
        edgemap_down[h_ind,w_ind,:] = 255
        print i
        print h_ind,w_ind
    imgdisp = Image.fromarray(edgemap_down,'RGB')
    display(imgdisp)

    ##################
    ##   leftside   ##
    ##################

    img_left = img[:,:boxminw]
    imgdisp = Image.fromarray(img_left,'L')
    display(imgdisp)

    imname_left = imname[:-4]+'_left'+imname[-4:]
    imsave(imname_left,img_left)
    eng.sobelwatershed(imname_left)
    matname_left = imname_left[:-3]+'mat'
    segments_left = sio.loadmat(matname_left)
    segments_left = segments_left['label']

    img_left = img_left.astype(np.float32)
    img_left = img_left / 255.0
    maxsid_left = np.max(segments_left)

    segmap_left = np.zeros(shape=(img_left.shape[0],img_left.shape[1],3), dtype=np.uint8)
    for i in range(maxsid_left):
        h_ind, w_ind = np.where(segments_left==i)
        segmap_left[h_ind,w_ind,:] = np.random.choice(256,3)
    imgdisp = Image.fromarray(segmap_left,'RGB')
    display(imgdisp)

    ## segment to edge

    segarr_left = segments_left[boxminh-1:boxmaxh+2,-1]
    segarr_left_uni, segidx = np.unique(segarr_left,return_index=True)
    segidx = np.sort(segidx)
    segarr_left_uni = segarr_left[segidx]
    boundary_left = np.zeros(shape=segments_left.shape,dtype=np.bool)
    bound_hdict_left = {}
    bound_wdict_left = {}
    for i in range(segarr_left_uni.shape[0]):
        si = segarr_left_uni[i]
        segmask = np.zeros(shape=segments_left.shape,dtype=np.bool)
        segmask[np.where(segments_left==si)] = True
        segboundary = find_boundaries(segmask,1,'inner')
        h_ind,w_ind = np.where(segboundary==True)
        boundary_left[h_ind,w_ind] = True
        bound_hdict_left[si] = h_ind
        bound_wdict_left[si] = w_ind

    edgelist_left = []
    for i in range(boxmaxh+1,boxminh-1,-1):
        si = segments_left[i,-1]
        si_ = segments_left[i-1,-1]
        if si!=si_:
            h_ind = bound_hdict_left[si].copy()
            w_ind = bound_wdict_left[si].copy()
            h0 = i
            w0 = segments_left.shape[1]-1
            cnt,h_ind,w_ind = SortEdge(h0,w0,h_ind,w_ind,segments_left,si,maxnpix)
            if cnt>0:
                h_ind = h_ind[:cnt]
                w_ind = w_ind[:cnt]
                edgelist_left.append((h_ind,w_ind))

    edgemap_left = np.zeros(shape=segmap_left.shape,dtype=np.uint8)
    for i in range(len(edgelist_left)):
        h_ind = edgelist_left[i][0]
        w_ind = edgelist_left[i][1]
        edgemap_left[h_ind,w_ind,:] = 255
        print i
        print h_ind,w_ind
    imgdisp = Image.fromarray(edgemap_left,'RGB')
    display(imgdisp)

    ###################
    ##   rightside   ##
    ###################

    img_right = img[:,-boxminw:]
    imgdisp = Image.fromarray(img_right,'L')
    display(imgdisp)

    imname_right = imname[:-4]+'_right'+imname[-4:]
    imsave(imname_right,img_right)
    eng.sobelwatershed(imname_right)
    matname_right = imname_right[:-3]+'mat'
    segments_right = sio.loadmat(matname_right)
    segments_right = segments_right['label']

    img_right = img_right.astype(np.float32)
    img_right = img_right / 255.0
    maxsid_right = np.max(segments_right)

    segmap_right = np.zeros(shape=(img_right.shape[0],img_right.shape[1],3), dtype=np.uint8)
    for i in range(maxsid_right):
        h_ind, w_ind = np.where(segments_right==i)
        segmap_right[h_ind,w_ind,:] = np.random.choice(256,3)
    imgdisp = Image.fromarray(segmap_right,'RGB')
    display(imgdisp)

    ## segment to edge

    segarr_right = segments_right[boxminh-1:boxmaxh+2,0]
    segarr_right_uni, segidx = np.unique(segarr_right,return_index=True)
    segidx = np.sort(segidx)
    segarr_right_uni = segarr_right[segidx]
    boundary_right = np.zeros(shape=segments_right.shape,dtype=np.bool)
    bound_hdict_right = {}
    bound_wdict_right = {}
    for i in range(segarr_right_uni.shape[0]):
        si = segarr_right_uni[i]
        segmask = np.zeros(shape=segments_right.shape,dtype=np.bool)
        segmask[np.where(segments_right==si)] = True
        segboundary = find_boundaries(segmask,1,'inner')
        h_ind,w_ind = np.where(segboundary==True)
        boundary_right[h_ind,w_ind] = True
        bound_hdict_right[si] = h_ind
        bound_wdict_right[si] = w_ind

    edgelist_right = []
    for i in range(boxminh-1,boxmaxh-1):
        si = segments_right[i,0]
        si_ = segments_right[i+1,0]
        if si!=si_:
            h_ind = bound_hdict_right[si].copy()
            w_ind = bound_wdict_right[si].copy()
            h0 = i
            w0 = 0
            cnt,h_ind,w_ind = SortEdge(h0,w0,h_ind,w_ind,segments_right,si,maxnpix)
            if cnt>0:
                h_ind = h_ind[:cnt]
                w_ind = w_ind[:cnt]
                edgelist_right.append((h_ind,w_ind))

    edgemap_right = np.zeros(shape=segmap_right.shape,dtype=np.uint8)
    for i in range(len(edgelist_right)):
        h_ind = edgelist_right[i][0]
        w_ind = edgelist_right[i][1]
        edgemap_right[h_ind,w_ind,:] = 255
        print i
        print h_ind,w_ind
    imgdisp = Image.fromarray(edgemap_right,'RGB')
    display(imgdisp)

    edgelist = []
    edgelist.extend(edgelist_up)
    for i in range(len(edgelist_right)):
        h_ind = edgelist_right[i][0]
        w_ind = edgelist_right[i][1].copy()
        w_ind = w_ind + img.shape[1] - boxminw
        edgelist.append((h_ind,w_ind))
    for i in range(len(edgelist_down)):
        h_ind = edgelist_down[i][0].copy()
        w_ind = edgelist_down[i][1]
        h_ind = h_ind + img.shape[0] - boxminh
        edgelist.append((h_ind,w_ind))
    edgelist.extend(edgelist_left)

    edgehead = []
    edgetail = []
    for e in edgelist:
        h_ind = e[0]
        w_ind = e[1]
        head_e = np.array([h_ind[0],w_ind[0]],dtype=np.float32)
        tail_e = np.array([h_ind[-1],w_ind[-1]],dtype=np.float32)
        edgehead.append(head_e)
        edgetail.append(tail_e)

    ## finish edge detection

    testmask = img.copy()
    testmask = np.tile(np.expand_dims(testmask,axis=2),(1,1,3))
    testmask[boxminh:boxmaxh+1,boxminw:boxmaxw+1,0] = 0
    testmask[boxminh:boxmaxh+1,boxminw:boxmaxw+1,1] = 0
    testmask[boxminh:boxmaxh+1,boxminw:boxmaxw+1,2] = 255
    for i in range(len(edgelist)):
        ind = edgelist[i]
        testmask[ind[0],ind[1],0] = 255
        testmask[ind[0],ind[1],1] = 0
        testmask[ind[0],ind[1],2] = 0
        print i,ind[0],ind[1]
    imgdisp = Image.fromarray(testmask,'RGB')
    display(imgdisp)

    ##################################
    ##   local feature: intensity   ##
    ##################################

    intensity = []
    inten = np.zeros(shape=(1,))
    bdsz = 10 # border_size
    border_mask = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.bool)
    border_mask[boxminh-bdsz:boxmaxh+bdsz+1,boxminw-bdsz:boxmaxw+bdsz+1] = True
    border_mask[boxminh:boxmaxh+1,boxminw:boxmaxw+1] = False
    for i in range(len(edgelist)):
        j = (i+len(edgelist)+1)%len(edgelist)
        hi = edgelist[i][0][0]
        wi = edgelist[i][1][0]
        hj = edgelist[j][0][0]
        wj = edgelist[j][1][0]
        seg_j = -1
        h_ind = np.ndarray(shape=(1,))
        w_ind = np.ndarray(shape=(1,))
        if hj<boxminh and wj>=boxminw-1 and wj<boxmaxw+1:
            seg_j = segments_up[hj,wj]
            h_ind,w_ind = np.where(segments_up==seg_j)
            ind = np.where(border_mask[h_ind,w_ind]==True)
            h_ind = h_ind[ind]
            w_ind = w_ind[ind]
        elif hj>=boxminh-1 and hj<boxmaxh+1 and wj>boxmaxh:
            wj_ = wj - (img.shape[1] - boxminw)
            seg_j = segments_right[hj,wj_]
            h_ind,w_ind = np.where(segments_right==seg_j)
            w_ind = w_ind + (img.shape[1] - boxminw)
            ind = np.where(border_mask[h_ind,w_ind]==True)
            h_ind = h_ind[ind]
            w_ind = w_ind[ind]
        elif hj>boxmaxh and wj>=boxminw and wj<boxmaxw+2:
            hj_ = hj - (img.shape[0] - boxminh)
            seg_j = segments_down[hj_,wj]
            h_ind,w_ind = np.where(segments_down==seg_j)
            h_ind = h_ind + (img.shape[0] - boxminh)
            ind = np.where(border_mask[h_ind,w_ind]==True)
            h_ind = h_ind[ind]
            w_ind = w_ind[ind]
        elif hj>=boxminh and hj<boxmaxh+2 and wj<boxminw:
            seg_j = segments_left[hj,wj]
            h_ind,w_ind = np.where(segments_left==seg_j)
            ind = np.where(border_mask[h_ind,w_ind]==True)
            h_ind = h_ind[ind]
            w_ind = w_ind[ind]
        inten = np.median(img[h_ind,w_ind],axis=0)
        intensity.append(inten)
    print intensity

    bandsz = 20
    band = np.ndarray(shape=(bandsz,bandsz*len(intensity)),dtype=np.uint8)
    for i in range(len(intensity)):
        band[:,i*bandsz:(i+1)*bandsz] = intensity[i]
        band[:,i*bandsz] = 0
    imgdisp = Image.fromarray(band,'L')
    display(imgdisp)

    ##################################
    ##   local feature: gradient    ##
    ##################################

    gradient_up = sobel(img_up)
    gradient_down = sobel(img_down)
    gradient_left = sobel(img_left)
    gradient_right = sobel(img_right)
    gradmag = []
    for i in range(len(edgelist)):
        h_ind = edgelist[i][0]
        w_ind = edgelist[i][1]
        npix = h_ind.shape[0]
        weights = np.flip(np.arange(maxnpix)+1,axis=0)
        weights = weights[:npix]
        h = h_ind[0]
        w = w_ind[0]
        grads = np.ndarray(shape=(1,))
        if h<boxminh and w>=boxminw-1 and w<boxmaxw+1:
            grads = gradient_up[h_ind,w_ind].copy()
        elif h>=boxminh-1 and h<boxmaxh+1 and w>boxmaxw:
            w_ind_ = w_ind - (img.shape[1] - boxminw)
            grads = gradient_right[h_ind,w_ind_].copy()
        elif h>boxmaxh and w>=boxminw and w<=boxmaxw+1:
            h_ind_ = h_ind - (img.shape[0] - boxminh)
            grads = gradient_down[h_ind_,w_ind].copy()
        elif h>=boxminh and h<=boxmaxh+1 and w<boxminw:
            grads = gradient_left[h_ind,w_ind].copy()
        w_grads = grads*weights
        ## attention here ! two possible ways...
        ind = np.argsort(w_grads)
        med = 0
        grads = grads[ind]
        sz = grads.size
        if sz%2==0:
            med = (grads[sz/2]+grads[sz/2-1])/2.0
        else:
            med = grads[(sz-1)/2]
        gradmag.append(med)
        ## another way is median on w_grads directly.
        #gradmag.append(np.median(w_grads))
    print gradmag

    visu_grad = np.zeros(shape=(bandsz*2,bandsz*(len(intensity)+1)),dtype=np.uint8)
    visu_grad[:bandsz,:bandsz] = intensity[-1]
    for i in range(len(intensity)):
        visu_grad[:bandsz,(i+1)*bandsz:(i+2)*bandsz] = intensity[i]
        visu_grad[:bandsz,(i+1)*bandsz] = 0
        visu_grad[bandsz:,(i)*bandsz+bandsz/2:(i+1)*bandsz+bandsz/2] = int(gradmag[i]*255.0)
    imgdisp = Image.fromarray(visu_grad,'L')
    display(imgdisp)

    ########################
    ##   Circle Fitting   ##
    ########################

    nedge = len(edgelist)
    circle = np.ndarray(shape=(nedge,nedge,2),dtype=np.float32)
    radius = np.ndarray(shape=(nedge,nedge),dtype=np.float32)
    spatial_dev = np.zeros(shape=(nedge,nedge),dtype=np.float32)
    angular_con = np.zeros(shape=(nedge,nedge),dtype=np.float32)
    aperture = np.zeros(shape=(nedge,nedge),dtype=np.float32)
    for i in range(nedge):
        for j in range(nedge):
            x_pts = np.concatenate((edgelist[i][0],edgelist[j][0]),axis=0)
            y_pts = np.concatenate((edgelist[i][1],edgelist[j][1]),axis=0)
            pts = np.concatenate((x_pts.reshape(-1,1),y_pts.reshape(-1,1)),axis=1)
            xc_, yc_, R_, Ri_ = CircleFitting(x_pts, y_pts)
            circle[i,j,0] = xc_
            circle[i,j,1] = yc_
            radius[i,j] = R_
            spa_dev = np.median(np.abs(Ri_-R_))
            spatial_dev[i,j] = 2.16/(spa_dev+2.16)

            head_i = edgehead[i]
            tail_i = edgetail[i]
            head_j = edgehead[j]
            tail_j = edgetail[j]
            alpha_hi = math.atan2(head_i[1] - yc_, head_i[0] - xc_)
            alpha_ti = math.atan2(tail_i[1] - yc_, tail_i[0] - xc_)
            alpha_hj = math.atan2(head_j[1] - yc_, head_j[0] - xc_)
            alpha_tj = math.atan2(tail_j[1] - yc_, tail_j[0] - xc_)
            alpha_i = AngleDiff(alpha_ti,alpha_hi)
            alpha_j = AngleDiff(alpha_tj,alpha_hj)
            angular_con[i,j] = abs(alpha_i - alpha_j) / (abs(alpha_i) + abs(alpha_j) + 1e-6)

            alpha_h = AngleDiff(alpha_hi,alpha_hj)
            alpha_t = AngleDiff(alpha_ti,alpha_tj)
            if abs(alpha_h)<abs(alpha_t):
                aperture[i,j] = math.sqrt(1 - abs(alpha_h)/(math.pi*2.0))
            else:
                aperture[i,j] = math.sqrt(abs(alpha_h) / (math.pi*2.0))
    naturalness = 1 - spatial_dev * angular_con * aperture
    # print naturalness

    ##############################################
    ##   pairwise distance & couple potential   ##
    ##############################################

    dist_mat = np.ones(shape=(nedge,nedge),dtype=np.float32)
    for i in range(nedge):
        for j in range(nedge):
            if i==j:
                continue
            ci = intensity[i]/255.0
            cj = intensity[j]/255.0
            ti = intensity[(i+len(intensity)-1)%len(intensity)]/255.0
            tj = intensity[(j+len(intensity)-1)%len(intensity)]/255.0
            gi = gradmag[i]
            gj = gradmag[j]
            wij = naturalness[i,j]
            item_1 = np.mean(np.square(ti-cj))
            item_2 = np.mean(np.square(ti-cj))
            item_3 = math.pow(gi-gj,2)
            item_4 = math.pow(wij,2)
            assert(item_1<=1 and item_1>=0)
            assert(item_2<=1 and item_2>=0)
            assert(item_3<=1 and item_3>=0)
            assert(item_4<=1 and item_4>=0)
            dist_mat[i,j] = math.sqrt( (item_1 + item_2 + item_3 + item_4) / 4.0 )
    #for i in range(dist_mat.shape[0]):
    #    print dist_mat[i]

    potential = np.zeros(shape=(nedge,nedge),dtype=np.bool)
    for i in range(nedge-1):
        for j in range(i,nedge):
            min_j = np.min(dist_mat[:,j])*2.0
            min_i = np.min(dist_mat[i,:])*2.0
            if dist_mat[i,j] < min(min_j, min_i, math.sqrt(0.1)):
                potential[i,j] = True
                potential[j,i] = True
    #for i in range(potential.shape[0]):
    #    print potential[i]

    couples_all = []
    for i in range(nedge-1):
        for j in range(i+1,nedge):
            if potential[i,j]==True:
                couples_all.append((i,j))
    ncouples = len(couples_all)
    maxncouples = 13
    if ncouples>maxncouples:
        dist_arr = np.ndarray(shape=(ncouples,),dtype=np.float32)
        for i in range(ncouples):
            h = couples_all[i][0]
            w = couples_all[i][1]
            dist_arr[i] = dist_mat[h,w]
        sort_ind = np.argsort(dist_arr)
        couples_arr = []
        for i in range(maxncouples):
            couples_arr.append(couples_all[sort_ind[i]])
        couples_all = couples_arr
        ncouples = len(couples_all)

    print 'ncouples:',ncouples
    print 'couples: ',couples_all

    ################################
    ##    grouping edge couples   ##
    ################################

    nsubset = int(math.pow(2,ncouples))
    min_cost_cfg = 1e6
    best_config = []
    sparse_edge = []
    N_E = nedge
    print 'nsubset:',nsubset
    print 'nedge:',nedge

    print 'begin grouping edges...'
    start_time = time.time()
    nsubset_list = []
    for i in range(1,nsubset):
        nsubset_list.append(i)
    for i in pbar(nsubset_list):
        dec = i
        bmask = np.zeros(shape=(ncouples,),dtype=np.int32)
        for j in range(ncouples):
            if dec == 0:
                break
            bmask[j] = dec%2
            dec = dec//2

        bmat = np.zeros(shape=(nedge,nedge),dtype=np.int32)
        mat2mask = np.ndarray(shape=(nedge,nedge),dtype=np.int32)
        mat2mask[:,:] = -1
        for j in range(ncouples):
            if bmask[j]==1:
                u = couples_all[j][0]
                v = couples_all[j][1]
                bmat[u,v] = 1
                bmat[v,u] = 1
                mat2mask[u,v] = j
                mat2mask[v,u] = j

        sparseEdge = []
        for j in range(nedge):
            if np.sum(bmat[j,:])==0:
                sparseEdge.append(j)

        sum_arr = np.sum(bmat,axis=1)
        ind = np.where(sum_arr>0)
        ind = ind[0]
        #N_E = ind.size
        #print 'N_E: ',N_E
        for j in range(ncouples):
            if bmask[j]==0:
                continue
            bmask_ = bmask.copy()
            bmat_ = bmat.copy()
            config = []
            k = j
            tobreak = False
            while not tobreak:
                group = []
                q = couples_all[k]
                group.append(q)
                ek = q[0]
                el = q[1]
                bmat_[ek,el] = 0
                bmat_[el,ek] = 0
                bmask_[mat2mask[ek,el]] = 0
                if np.sum(bmask_) == 0:
                    tobreak = True

                ek_next = ek
                while not tobreak:
                    ek_next = (ek_next+1)%nedge
                    if ek_next==el:
                        break
                    if np.sum(bmat_[ek_next,:])==0:
                        continue
                    el_next = el
                    while not tobreak:
                        el_next = (el_next+nedge-1)%nedge
                        if el_next==ek_next:
                            break
                        if bmat_[ek_next,el_next]==0:
                            continue
                        else:
                            group.append((ek_next,el_next))
                            bmat_[ek_next,el_next] = 0
                            bmat_[el_next,ek_next] = 0
                            bmask_[mat2mask[ek_next,el_next]] = 0
                            if np.sum(bmask_) == 0:
                                tobreak = True
                            ek = ek_next
                            el = el_next
                            break

                ek = q[0]
                el = q[1]
                ek_next = ek
                while not tobreak:
                    ek_next = (ek_next+nedge-1)%nedge
                    if ek_next==el:
                        break
                    if np.sum(bmat_[ek_next,:])==0:
                        continue
                    el_next = el
                    while not tobreak:
                        el_next = (el_next+1)%nedge
                        if el_next==ek_next:
                            break
                        if bmat_[ek_next,el_next]==0:
                            continue
                        else:
                            group.append((ek_next,el_next))
                            bmat_[ek_next,el_next] = 0
                            bmat_[el_next,ek_next] = 0
                            bmask_[mat2mask[ek_next,el_next]] = 0
                            if np.sum(bmask_) == 0:
                                tobreak = True
                            ek = ek_next
                            el = el_next
                            break

                config.append(group)
                if not tobreak:
                    ind = np.where(bmask_==1)
                    ind = ind[0]
                    k = ind[0]

            tmp = 0
            for g in config:
                tmp = tmp + (len(g)-1)

            cost_glb = 1
            if (N_E//2)==1 and tmp==1:
                cost_glb = 0
            elif (N_E//2)>=2 and tmp>=1:
                cost_glb = 1 - tmp*1.0 / (N_E//2-1)

            cost_loc = 0
            for g in config:
                for cp in g:
                    u = cp[0]
                    v = cp[1]
                    up_term = 0
                    down_term = 1
                    beta_c = 0
                    beta_t = 0
                    u_c = (u+1)%nedge
                    u_t = (u+nedge-1)%nedge
                    for k in range(len(g)):
                        if g[k][0]==u_c or g[k][1]==u_c:
                            beta_c = 1
                        if g[k][0]==u_t or g[k][1]==u_t:
                            beta_t = 1
                        if beta_c+beta_t==2:
                            break
                    if beta_c==1:
                        lamda_u_c = intensity[u]
                        lamda_v_t = intensity[(v+nedge-1)%nedge]
                        up_term = up_term + math.pow((lamda_u_c.mean()-lamda_v_t.mean())/255.0,2)
                        down_term = down_term + 1
                    if beta_t==1:
                        lamda_u_t = intensity[(u+nedge-1)%nedge]
                        lamda_v_c = intensity[v]
                        up_term = up_term + math.pow((lamda_u_t.mean()-lamda_v_c.mean())/255.0,2)
                        down_term = down_term + 1
                    if beta_c==1 and beta_t==1:
                        grad_u = gradmag[u]
                        grad_v = gradmag[v]
                        up_term = up_term + math.pow(grad_u-grad_v,2)
                        down_term = down_term + 1
                    up_term = up_term + math.pow(naturalness[u,v],2)
                    cost_u_v = math.sqrt(up_term*1.0/down_term)
                    cost_loc = cost_loc + cost_u_v
            if tmp<1:
                cost_loc = 0
            else:
                cost_loc = cost_loc / tmp
            cost_cfg = cost_loc*0.8 + cost_glb*0.2

            if cost_cfg<min_cost_cfg:
                min_cost_cfg = cost_cfg
                best_config = config
                sparse_edge = sparseEdge

    print best_config
    print min_cost_cfg
    print sparse_edge
    print 'grouping time cost: %.4f secs'%(time.time()-start_time)

    #################################
    ##   reconstruct sparse edge   ##
    #################################

    sparse_edge = np.array(sparse_edge,dtype=np.uint32)
    lumin_diff = np.ndarray(shape=sparse_edge.shape,dtype=np.float32)
    for i in range(lumin_diff.shape[0]):
        e = sparse_edge[i]
        lamda_e_c = intensity[e].mean()
        lamda_e_t = intensity[(e+nedge-1)%nedge].mean()
        lumin_diff[i] = 0-abs(float(lamda_e_c-lamda_e_t))
    ind = np.argsort(lumin_diff,axis=0)
    sparse_edge = sparse_edge[ind]

    sparse_line = {}
    sparse_lineseg = {}
    for e in sparse_edge:
        h_ind = edgelist[e][0]
        w_ind = edgelist[e][1]
        ## h is x, w is y
        A,B,C = LineFitting(h_ind,w_ind)
        sparse_line[e] = np.array((A,B,C),dtype=np.float32)
        h_ind_ = np.ndarray(shape=(2,),dtype=np.float32)
        w_ind_ = np.ndarray(shape=(2,),dtype=np.float32)
        h0 = h_ind[0]
        w0 = w_ind[0]

        if h0<boxminh and w0>=boxminw-1 and w0<=boxmaxw+1:
            if B==0.:
                h_ind_[:] = boxminh
                w_ind_[:] = w0
                continue
            h_ind_[0] = boxminh
            w_ind_[0] = -(A*boxminh+C)/B
            w = -(A*boxmaxh+C)/B
            if w<boxminw:
                h_ind_[1] = -(B*boxminw+C)/A
                w_ind_[1] = boxminw
            elif w<=boxmaxw:
                h_ind_[1] = boxmaxh
                w_ind_[1] = w
            else:
                h_ind_[1] = -(B*boxmaxw+C)/A
                w_ind_[1] = boxmaxw
        elif h0>=boxminh-1 and h0<=boxmaxh+1 and w0>boxmaxw:
            if A==0:
                h_ind_[:] = h0
                w_ind_[:] = boxmaxw
                continue
            h_ind_[0] = -(B*boxmaxw+C)/A
            w_ind_[0] = boxmaxw
            h = -(B*boxminw+C)/A
            if h<boxminh:
                h_ind_[1] = boxminh
                w_ind_[1] = -(A*boxminh+C)/B
            elif h<=boxmaxh:
                h_ind_[1] = h
                w_ind_[1] = boxminw
            else:
                h_ind_[1] = boxmaxh
                w_ind_[1] = -(A*boxmaxh+C)/B
        elif h0>boxmaxh and w0>=boxminw-1 and w0<=boxmaxw+1:
            if B==0:
                h_ind_[:] = boxmaxh
                w_ind_[:] = w0
                continue
            h_ind_[0] = boxmaxh
            w_ind_[0] = -(A*boxmaxh+C)/B
            w = -(A*boxminh+C)/B
            if w<boxminw:
                h_ind_[1] = -(B*boxminw+C)/A
                w_ind_[1] = boxminw
            elif w<=boxmaxw:
                h_ind_[1] = boxminh
                w_ind_[1] = w
            else:
                h_ind_[1] = -(B*boxmaxw+C)/A
                w_ind_[1] = boxmaxw
        else:
            if A==0:
                h_ind_[:] = h0
                w_ind_[:] = boxminw
                continue
            h_ind_[0] = -(B*boxminw+C)/A
            w_ind_[0] = boxminw
            h = -(B*boxmaxw+C)/A
            if h<boxminh:
                h_ind_[1] = boxminh
                w_ind_[1] = -(A*boxminh+C)/B
            elif h<=boxmaxh:
                h_ind_[1] = -(B*boxmaxw+C)/A
                w_ind_[1] = boxmaxw
            else:
                h_ind_[1] = boxmaxh
                w_ind_[1] = -(A*boxmaxh+C)/B
        sparse_lineseg[e] = (h_ind_,w_ind_)

    for e in sparse_edge:
        print 'e: ',e
        if e in sparse_line:
            print 'sparse_line: ',sparse_line[e]
        if e in sparse_lineseg:
            print 'sparse_lineseg: ',sparse_lineseg[e]

    len_groups = np.ndarray(shape=(len(best_config),),dtype=np.int32)
    for i in range(len(best_config)):
        len_groups[i] = len(best_config[i])
    g_ind = np.flip(np.argsort(len_groups),axis=0)
    sorted_groups = []
    for i in range(g_ind.shape[0]):
        sorted_groups.append(best_config[g_ind[i]])

    num_curves = np.sum(len_groups) + len(sparse_edge)
    curve_map = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.int32)
    curve_map[:,:] = -1
    curve_map[boxminh:boxmaxh+1,boxminw:boxmaxw+1] = 0

    curve_id = 1
    curve2id = {}
    id2curve = {}

    rebuild = np.zeros(shape=(img.shape[0],img.shape[1],3),dtype=np.uint8)
    rebuild[boxminh-1,boxminw-1:boxmaxw+2,:] = 255
    rebuild[boxmaxh+1,boxminw-1:boxmaxw+2,:] = 255
    rebuild[boxminh-1:boxmaxh+2,boxminw-1,:] = 255
    rebuild[boxminh-1:boxmaxh+2,boxmaxw+1,:] = 255
    imgdisp = Image.fromarray(rebuild,'RGB')
    display(imgdisp)

    start_time = time.time()
    ofshw = np.concatenate((np.array(fxh).reshape(-1,1),np.array(fxw).reshape(-1,1)),axis=1)
    for i in range(len(sorted_groups)):
        for cp in sorted_groups[i]:
            u = cp[0]
            v = cp[1]
            Cir = (circle[u,v],radius[u,v])
            h1 = edgehead[u]
            t1 = edgetail[u]
            h2 = edgehead[v]
            t2 = edgetail[v]
            pixel_arr = GetCircle(curve_map, Cir, h1,t1,h2,t2)
            pixel_arr_ = np.tile(np.expand_dims(pixel_arr,axis=1),(1,ofshw.shape[0],1))
            ofshw_ = np.tile(np.expand_dims(ofshw,axis=0),(pixel_arr.shape[0],1,1))
            pixel_adj = pixel_arr_ + ofshw_
            inter = np.zeros(shape=(pixel_arr.shape[0],),dtype=np.bool)
            for j in range(ofshw.shape[0]):
                h_ind = pixel_adj[:,j,0]
                w_ind = pixel_adj[:,j,1]
                adj = curve_map[h_ind,w_ind]
                inter = np.where(adj>0,True,inter)
            ind = np.nonzero(inter)
            ind = ind[0]
            print ind
            start = 0
            end = pixel_arr.shape[0]-1
            if ind.size>0:
                start = max(start,ind[-1])
                end = min(end,ind[0])
            h_ind = pixel_arr[:end+1,0]
            w_ind = pixel_arr[:end+1,1]
            rebuild[h_ind,w_ind,:] = 255
            curve_map[h_ind,w_ind] = curve_id
            print 'start: ',start
            print 'end: ',end
            if start>end:
                h_ind = pixel_arr[start:,0]
                w_ind = pixel_arr[start:,1]
                rebuild[h_ind,w_ind,:] = 255
                curve_map[h_ind,w_ind] = curve_id

            curve2id[(u,v)] = curve_id
            id2curve[curve_id] = (u,v)
            curve_id = curve_id+1
            imgdisp = Image.fromarray(rebuild,'RGB')
            display(imgdisp)
    print 'drawing edges time cost: %.4f'%(time.time()-start_time)
    imgdisp = Image.fromarray(rebuild,'RGB')
    display(imgdisp)
    
    strip_map = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.int32)
    h_ind,w_ind = np.where(curve_map==0)
    strip_map[h_ind,w_ind] = 1
    connected_strip = measure.label(strip_map,background=0,connectivity=1)
    imgdisp = np.zeros(shape=(connected_strip.shape[0],connected_strip.shape[1],3),dtype=np.uint8)
    for i in range(np.min(connected_strip),np.max(connected_strip)+1):
        h_ind,w_ind = np.where(connected_strip==i)
        imgdisp[h_ind,w_ind,:] = np.random.choice(256,3)
    imgdisp = Image.fromarray(imgdisp,'RGB')
    display(imgdisp)

    width = boxmaxw+1-boxminw
    height = boxmaxh+1-boxminh
    inpaint_bound = np.ndarray(shape=((width+height+2)*2,2),dtype=np.int32)
    inpaint_bound[:width+1,0] = boxminh-1
    inpaint_bound[:width+1,1] = np.arange(width+1)+boxminw-1
    inpaint_bound[width+1:width+height+2,0] = np.arange(height+1)+boxminh-1
    inpaint_bound[width+1:width+height+2,1] = boxmaxw+1
    inpaint_bound[width+height+2:width*2+height+3,0] = boxmaxh+1
    inpaint_bound[width+height+2:width*2+height+3,1] = np.flip(np.arange(width+1),axis=0)+boxminw
    inpaint_bound[width*2+height+3:width*2+height*2+4,0] = np.flip(np.arange(height+1),axis=0)+boxminh
    inpaint_bound[width*2+height+3:width*2+height*2+4,1] = boxminw-1
    inpaint_bound_mask = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.bool)
    inpaint_bound_mask[inpaint_bound[:,0],inpaint_bound[:,1]] = True

    fill_flag = np.ndarray(shape=(img.shape[0],img.shape[1]),dtype=np.bool)
    fill_flag[:,:] = True
    fill_flag[boxminh:boxmaxh+1,boxminw:boxmaxw+1] = False
    result = img.copy()
    for i in range(1,np.max(connected_strip)+1):
        print 'i: ',i
        h_ind,w_ind = np.where(connected_strip==i)
        strip_hw = np.concatenate((h_ind.reshape(-1,1),w_ind.reshape(-1,1)),axis=1)
        strip_mask = np.zeros(shape=connected_strip.shape,dtype=np.bool)
        strip_mask[h_ind,w_ind] = True
        strip_bound = find_boundaries(strip_mask,mode='outer')
        bnd_h,bnd_w = np.where(strip_bound==True)
        ind = np.where(inpaint_bound_mask[bnd_h,bnd_w]==True)
        out_bnd_h = bnd_h[ind]
        out_bnd_w = bnd_w[ind]
        bnd_hw = np.concatenate((out_bnd_h.reshape(-1,1),out_bnd_w.reshape(-1,1)),axis=1)
        ind = np.where(inpaint_bound_mask[bnd_h,bnd_w]==False)
        in_bnd_h = bnd_h[ind]
        in_bnd_w = bnd_w[ind]
        ind = np.where(fill_flag[in_bnd_h,in_bnd_w]==False)
        in_bnd_h = in_bnd_h[ind]
        in_bnd_w = in_bnd_w[ind]
        in_bnd_hw = np.concatenate((in_bnd_h.reshape(-1,1),in_bnd_w.reshape(-1,1)),axis=1)
        strip_hw = np.concatenate((strip_hw,in_bnd_hw),axis=0)
        h_ind = strip_hw[:,0]
        w_ind = strip_hw[:,1]
        imgdisp = np.zeros(shape=(img.shape[0],img.shape[1],3),dtype=np.uint8)
        imgdisp[bnd_h,bnd_w,:] = 255
        imgdisp = Image.fromarray(imgdisp,'RGB')
        display(imgdisp)

        touch_bounds = []
        touch_bnd = []
        for j in range(inpaint_bound.shape[0]):
            j_hw = inpaint_bound[j]
            for k in range(bnd_hw.shape[0]):
                k_hw = bnd_hw[k]
                if j_hw[0]!=k_hw[0] or j_hw[1]!=k_hw[1]:
                    continue
                if len(touch_bnd)>0 and np.linalg.norm(k_hw.reshape(-1,1)-touch_bnd[-1],axis=0)>=2:
                    touch_bounds.append(touch_bnd)
                    touch_bnd = []
                touch_bnd.append(k_hw.reshape(-1,1))
        if len(touch_bnd)>0:
            touch_bounds.append(touch_bnd)
        if len(touch_bounds)>1:
            first_hw = touch_bounds[0][0]
            last_hw = touch_bounds[-1][-1]
            if np.linalg.norm(first_hw-last_hw,axis=0)<2:
                touch_bounds[-1].extend(touch_bounds[0])
                touch_bounds[0] = touch_bounds[-1]
                touch_bounds = touch_bounds[:-1]

        print 'num bounds: ',len(touch_bounds)
        for touch_bnd in touch_bounds:
            print 'bnd start: ',touch_bnd[0]
            print 'bnd end: ',touch_bnd[-1]
            imgdisp = np.zeros(shape=(img.shape[0],img.shape[1],3),dtype=np.uint8)
            touch_bnd = np.transpose(np.concatenate(touch_bnd,axis=1),(1,0))
            imgdisp[touch_bnd[:,0],touch_bnd[:,1],:] = 255
            imgdisp = Image.fromarray(imgdisp,'RGB')
            display(imgdisp)

        P_list = []
        Omega_list = []
        for j in range(len(touch_bounds)):
            touch_bnd = touch_bounds[j]
            start = touch_bnd[0]
            end = touch_bnd[-1]
            d2start_min = 1e9
            d2end_min = 1e9
            d2start_curve_id = -1
            d2end_curve_id = -1
            for curve_id in range(1,num_curves+1):
                curve_h, curve_w = np.where(curve_map==curve_id)
                curve_hw = np.concatenate((curve_h.reshape(-1,1),curve_w.reshape(-1,1)),axis=1)
                if curve_hw.shape[0]==0: continue
                start_ = np.tile(start.reshape(1,-1),(curve_hw.shape[0],1))
                d2start_ = np.linalg.norm(curve_hw-start_,axis=1)
                min1 = np.argsort(d2start_)
                end_ = np.tile(end.reshape(1,-1),(curve_hw.shape[0],1))
                d2end_ = np.linalg.norm(curve_hw-end_,axis=1)
                min2 = np.argsort(d2end_)
                if d2start_[min1[0]]<d2start_min:
                    d2start_min = d2start_[min1[0]]
                    d2start_curve_id = curve_id
                if d2end_[min2[0]]<d2end_min:
                    d2end_min = d2end_[min2[0]]
                    d2end_curve_id = curve_id
            start_curve = id2curve[d2start_curve_id]
            end_curve = id2curve[d2end_curve_id]

            curve_list = [start_curve,end_curve]
            P = []
            Omega = []
            for k in range(len(curve_list)):
                curve = curve_list[k]
                bnd_arr = np.concatenate(touch_bnd,axis=1)
                bnd_arr = np.transpose(bnd_arr,(1,0))
                diff = np.ndarray(shape=(bnd_arr.shape[0],strip_hw.shape[0]),dtype=np.float32)
                inter = np.ndarray(shape=(bnd_arr.shape[0]-1,strip_hw.shape[0]),dtype=np.float32)

                ## get nearset 'parallel' point on the boundary
                if isinstance(curve,tuple):
                    u = curve[0]
                    v = curve[1]
                    q = circle[u,v].reshape(1,2)
                    q_ = np.tile(q,(strip_hw.shape[0],1))
                    r_ = np.linalg.norm(strip_hw-q_,axis=1).reshape(-1,1)
                    q_ = np.tile(q,(bnd_arr.shape[0],1))
                    d_ = np.linalg.norm(bnd_arr-q_,axis=1).reshape(-1,1)
                    d__ = np.tile(d_,(1,strip_hw.shape[0]))
                    r__ = np.tile(r_.reshape(1,-1),(bnd_arr.shape[0],1))
                    diff[:,:] = d__-r__
                    inter[:,:] = diff[:-1,:]*diff[1:,:]
                    inter = np.where(inter<=0,0,1)

                else:
                    e = curve 
                    lineseg = sparse_lineseg_rnd[e]
                    e_vec = np.array(lineseg[1])-np.array(lineseg[0])
                    bnd_arr_ = np.tile(np.expand_dims(bnd_arr,axis=1),(1,strip_hw.shape[0],1))
                    strip_hw_ = np.tile(np.expand_dims(strip_hw,axis=0),(bnd_arr.shape[0],1,1))
                    bnd_vec = bnd_arr_-strip_hw_
                    diff[:,:] = np.cross(bnd_vec,e_vec.reshape(1,1,-1))
                    inter[:,:] = diff[:-1,:]*diff[1:,:]
                    inter = np.where(inter<=0,0,1)

                P_ = np.ndarray(shape=strip_hw.shape,dtype=np.float32)
                if inter.shape[0]==0:
                    P_[:,:] = bnd_arr[0,:]
                elif k==0:
                    infarr = np.ndarray(shape=(inter.shape[1],),dtype=np.int32)
                    infarr[:] = int(1e7)
                    idxarr = np.ndarray(shape=(inter.shape[1],),dtype=np.int32)
                    idxarr[:] = int(1e7)
                    tmparr = np.ndarray(shape=(inter.shape[1],),dtype=np.int32)
                    for k_ in range(inter.shape[0]):
                        tmparr[:] = k_
                        tmparr = tmparr + infarr[:] * inter[k_].reshape(-1)
                        idxarr = np.minimum(idxarr, tmparr)
                    idxarr = np.where(idxarr>=int(1e6),0,idxarr)
                    idxarr_ = idxarr + 1
                    diff = np.abs(diff)
                    arang = np.arange(inter.shape[1])
                    diff_1 = diff[idxarr, arang]
                    diff_2 = diff[idxarr_, arang]
                    diff_ = diff_1 - diff_2
                    idxarr__ = np.where(diff_<0, idxarr, idxarr_)
                    P_[arang,:] = bnd_arr[idxarr__,:]
                else:
                    infarr = np.ndarray(shape=(inter.shape[1],),dtype=np.int32)
                    infarr[:] = int(-1e7)
                    idxarr = np.ndarray(shape=(inter.shape[1],),dtype=np.int32)
                    idxarr[:] = int(-1e7)
                    tmparr = np.ndarray(shape=(inter.shape[1],),dtype=np.int32)
                    for k_ in range(inter.shape[0]-1,-1,-1):
                        tmparr[:] = k_
                        tmparr = tmparr + infarr[:] * inter[k_].reshape(-1)
                        idxarr = np.maximum(idxarr, tmparr)
                    idxarr = np.where(idxarr<=int(-1e6),0,idxarr)
                    idxarr_ = idxarr + 1
                    diff = np.abs(diff)
                    arang = np.arange(inter.shape[1])
                    diff_1 = diff[idxarr, arang]
                    diff_2 = diff[idxarr_, arang]
                    diff_ = diff_1 - diff_2
                    idxarr__ = np.where(diff_<0, idxarr, idxarr_)
                    P_[arang,:] = bnd_arr[idxarr__,:]
                P.append(P_)
                dist2P_ = np.linalg.norm(P_-strip_hw,axis=1)
                dist2P_ = np.where(dist2P_<0.01,0.01,dist2P_)

                cid = curve2id[curve]
                c_h,c_w = np.where(curve_map==cid)
                c_hw = np.concatenate((c_h.reshape(-1,1),c_w.reshape(-1,1)),axis=1)
                c_hw_ = np.tile(np.expand_dims(c_hw,axis=1),(1,strip_hw.shape[0],1))
                strip_hw_ = np.tile(np.expand_dims(strip_hw,axis=0),(c_hw_.shape[0],1,1))
                d2curve_ = np.linalg.norm(c_hw_-strip_hw_,axis=2)
                d2curve = np.amin(d2curve_,axis=0)
                d2curve = np.where(d2curve<0.01,0.01,d2curve)

                Omega_ = np.power(dist2P_*d2curve,-1)
                Omega.append(Omega_)

            P_list.append(P)
            Omega_list.append(Omega)

        if len(touch_bounds)==1:
            P = P_list[0]
            Omega = Omega_list[0]
            P1 = P[0]
            P2 = P[1]
            lamda_P1 = img[P1[:,0].astype(np.int32),P1[:,1].astype(np.int32)].reshape(-1,1)
            lamda_P2 = img[P2[:,0].astype(np.int32),P2[:,1].astype(np.int32)].reshape(-1,1)
            Omega1 = Omega[0].reshape(-1,1)
            Omega2 = Omega[1].reshape(-1,1)
            lamda_P = (Omega1*lamda_P1+Omega2*lamda_P2)/(Omega1+Omega2)
            result[h_ind,w_ind] = lamda_P[:,0]
            fill_flag[h_ind,w_ind] = True
        else:
            sum_weight = np.zeros(shape=(strip_hw.shape[0],1),dtype=np.float32)
            sum_lamda = np.zeros(shape=(strip_hw.shape[0],1),dtype=np.float32)
            for j in range(len(touch_bounds)):
                P = P_list[j]
                Omega = Omega_list[j]
                P1 = P[0]
                P2 = P[1]
                Omega1 = np.tile(Omega[0].reshape(-1,1),(1,2))
                Omega2 = np.tile(Omega[1].reshape(-1,1),(1,2))
                v_P = (Omega1*P1+Omega2*P2)/(Omega1+Omega2)
                d2v_P = np.linalg.norm(v_P-strip_hw,axis=1)
                d2v_P = np.where(d2v_P<0.01,0.01,d2v_P)
                cur_weight = np.power(d2v_P,-1).reshape(-1,1)
                sum_weight = sum_weight+cur_weight

                lamda_P1 = img[P1[:,0].astype(np.int32),P1[:,1].astype(np.int32)].reshape(-1,1)
                lamda_P2 = img[P2[:,0].astype(np.int32),P2[:,1].astype(np.int32)].reshape(-1,1)
                Omega1 = Omega[0].reshape(-1,1)
                Omega2 = Omega[1].reshape(-1,1)
                lamda_P = (Omega1*lamda_P1+Omega2*lamda_P2)/(Omega1+Omega2)
                lamda_Pw = lamda_P*cur_weight.reshape(-1,1)
                sum_lamda = sum_lamda+lamda_Pw
            for j in range(len(sum_weight)):
                if abs(sum_weight[j][0])<1e-4:
                    print sum_weight[j]
                    print 1.0/sum_weight[j]
                    print len(touch_bounds)
                    break

            sum_lamda = sum_lamda/sum_weight
            result[h_ind,w_ind] = sum_lamda[:,0]
            fill_flag[h_ind,w_ind] = True
    imgdisp = Image.fromarray(result,'L')
    display(imgdisp)
    return result

if __name__=='__main__':
    imname = 'input_0016.png'
    result_r = imrestore_gray(imname, 0)
    result_g = imrestore_gray(imname, 1)
    result_b = imrestore_gray(imname, 2)
    result = np.ndarray(shape=(result_r.shape[0],result_r.shape[1],3),dtype=np.uint8)
    result[:,:,0] = result_r
    result[:,:,1] = result_g
    result[:,:,2] = result_b
    imgdisp = Image.fromarray(result,'RGB')
    display(imgdisp)