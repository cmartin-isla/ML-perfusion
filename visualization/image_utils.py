import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import measure
from mpl_toolkits.axes_grid1 import ImageGrid


def show_imagegrid(img_lst,n_r,n_c):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_r, n_c),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    
    
    for ax, im in zip(grid, img_lst):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()



def bbox(msk, margin):
    a = np.where(msk != 0)
    bbox = np.min(a[0])-margin, np.max(a[0]) + margin, np.min(a[1])- margin, np.max(a[1]) + margin
    return msk[bbox[0]:bbox[1],bbox[2]:bbox[3]]  

def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped






def euclideanDistance(coordinate1, coordinate2):
    return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)


def create_chessboard(h,w,s):
    cb = np.zeros((h,w))
    
    for x in range(w):
        for y in range(h):
            if (x%(2*s) < s and y%(2*s) < s) or (x%(2*s) >= s and y%(2*s) >= s):
                cb[y,x] = 1
                
                
    return cb            
    
def contour_distance(contour):
    distance = 0
    for i,p in enumerate(contour[:-1]):
        
        distance += euclideanDistance(p, contour[i+1])
    return distance
    

def equidistant_landmarks(contour,n_points):
    
    contour = [p for p in list(contour)]
    distance = contour_distance(contour)
    step = distance/n_points

    cummulative = []
    
    distance = 0
    for i,p in enumerate(contour[:-1]):
        distance += euclideanDistance(p, contour[i+1])
        if distance > step:
            distance = 0
        
        cummulative.append(distance)
    cummulative.append(0)
        
    landmarks = [contour[n] for n in np.where(np.array(cummulative) == 0)[0].tolist()]

     
    return landmarks


def closest_landmarks(contour_1,contour_2):
    
    contour_1 = [p for p in list(contour_1)]
    contour_2 = [p for p in list(contour_2)]
    
    contour_out = []
    
    
    for p2 in contour_1:
        
        min = np.inf
        selected_p = None
        
        for p in contour_2:
            if euclideanDistance(p,p2) < min:
                min = euclideanDistance(p,p2)
                selected_p = p 
        
        contour_out.append(selected_p)
        
        
    return contour_out
        
        


def adapt_contour(img_in,mask_in,mask_out=None):
    
    
    contours_in = measure.find_contours(mask_in,0)
    landmarks_in = equidistant_landmarks(contours_in[0],24)
    
    contours_out = measure.find_contours(mask_out,0)
    landmarks_out = equidistant_landmarks(contours_out[0],24)
    

    if len(landmarks_in) > len(landmarks_out):
        landmarks_in = landmarks_in[:len(landmarks_out)]
    else:
        landmarks_out = landmarks_out[:len(landmarks_in)]
        
         
    tform = PiecewiseAffineTransform()
    tform.estimate(np.fliplr(np.array(landmarks_out)), np.fliplr(np.array(landmarks_in)))
    img_out = warp(img_in, tform, output_shape=img_in.shape)

    return img_out, landmarks_in,landmarks_out


def adapt_double_contour(img_in,mask_in,mask_out=None,n_points = 24):
    
    
    contours_in = measure.find_contours(mask_in,0)
    landmarks_in_epi = equidistant_landmarks(contours_in[0],n_points)
    #print('len contour in:',len(list(contours_in[0])))
    #print('len lm in:',len(list(landmarks_in_epi)))

    #landmarks_in_end = equidistant_landmarks(contours_in[1],n_points)
    landmarks_in_end = closest_landmarks(landmarks_in_epi,contours_in[1])
    contours_out = measure.find_contours(mask_out,0)
    landmarks_out_epi = equidistant_landmarks(contours_out[0],len(list(landmarks_in_epi)))
    landmarks_out_end = closest_landmarks(landmarks_out_epi,contours_out[1])

    
    """
    if len(landmarks_in_epi) > len(landmarks_out_epi):
        landmarks_in_epi = landmarks_in_epi[:len(landmarks_out_epi)]
    else:
        landmarks_out_epi = landmarks_out_epi[:len(landmarks_in_epi)]
        
    if len(landmarks_in_end) > len(landmarks_out_end):
        landmarks_in_end = landmarks_in_end[:len(landmarks_out_end)]
    else:
        landmarks_out_end = landmarks_out_end[:len(landmarks_in_end)]
    """
    
    
    landmarks_out = []    
    landmarks_out.extend(landmarks_out_end)
    landmarks_out.extend(landmarks_out_epi)    
    
    
    landmarks_in = []    
    landmarks_in.extend(landmarks_in_end)
    landmarks_in.extend(landmarks_in_epi)
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(np.fliplr(np.array(landmarks_out)), np.fliplr(np.array(landmarks_in)))
    img_out = warp(img_in, tform, output_shape=img_in.shape,order = 3)
    
    
    
    return img_out, landmarks_in_epi,landmarks_in_end,landmarks_out_epi,landmarks_out_end



def generate_circle(img, radius, cx, cy):
    
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    img[cy-radius:cy+radius, cx-radius:cx+radius][index] = 1
    
    return img


def generate_ring(img, radius_1,radius_2, cx, cy):
    
    im1 = generate_circle(img.copy(), radius_1, cx, cy)
    im2 = generate_circle(img.copy(), radius_2, cx, cy)
    

    return im1 - im2
    
    

def obtain_polars(img,msk,n_points = 24,topology = 'ring'):
    

    width = img.shape[0]
    
    img_size = width
    half_size= int(width/2)
    
    msk_in = crop_or_pad_slice_to_size(bbox(msk,5),img_size,img_size)
    img_in = crop_or_pad_slice_to_size(bbox(img*msk,5),width,width)
    
    if topology == 'circle':
        msk_out = generate_circle(np.zeros_like(msk_in),half_size-30,half_size,half_size)
        img_out,lm_in,lm_out = adapt_contour(img_in,msk_in,msk_out)
    elif topology == 'ring':
        
        msk_out = generate_ring(np.zeros_like(msk_in),half_size-10,half_size-30,half_size,half_size)
        img_out,lm_in,lm_in_2,lm_out,lm_out_2 = adapt_double_contour(img_in,msk_in,msk_out, n_points = n_points)
        
        img_out = img_out* msk_out
    """
    for idx,p in enumerate(lm_out_2):
        plt.text(img_size+p[1],p[0],str(idx)) 
    
    for idx,p in enumerate(lm_in_2):
        plt.text(p[1],p[0],str(idx))    
    
    for idx,p in enumerate(lm_out):
        plt.text(img_size+p[1],p[0],str(idx)) 
    
    for idx,p in enumerate(lm_in):
        plt.text(p[1],p[0],str(idx)) 
        
    lm_out.append(lm_out[0])
    lm_out_2.append(lm_out_2[0])


    plt.plot(img_size+np.array(lm_out)[:,1],np.array(lm_out)[:,0],'-')
    plt.plot(img_size+np.array(lm_out_2)[:,1],np.array(lm_out_2)[:,0],'-')


    lm_in.append(lm_in[0])
    lm_in_2.append(lm_in_2[0])
    

        
    plt.plot(np.array(lm_in)[:,1],np.array(lm_in)[:,0],'-')
    plt.plot(np.array(lm_in_2)[:,1],np.array(lm_in_2)[:,0],'-')
    
    
    #plt.scatter(np.array(lm_in)[:,1],np.array(lm_in)[:,0])
    


    plt.imshow(np.hstack((img_in,img_out*msk_out)))
    #plt.imshow(msk_out)

    plt.show()
    
    """
    return crop_or_pad_slice_to_size((bbox(img_out,2)/np.max(img_out)),width,width)






def get_3_slices(img_es,msk_es):
    
    img_es_2 = []
    msk_es_2 = []
    
    [img_es_2.append(img_es[...,slc]) for slc in range(msk_es.shape[2]) if np.sum(msk_es[...,slc]) != 0]
    [msk_es_2.append(msk_es[...,slc]) for slc in range(msk_es.shape[2]) if np.sum(msk_es[...,slc]) != 0]
        
        
    z_interval = int(len(img_es_2)/4)
    
    imgs = [img_es_2[z_interval],
            img_es_2[z_interval*2],
            img_es_2[z_interval*3]]
    msks = [msk_es_2[z_interval],
            msk_es_2[z_interval*2],
            msk_es_2[z_interval*3]]
        
    return imgs, msks    



    
def obtain_16_segments(img_es,msk_es, n_points = 24):
    
    
    imgs,msks = get_3_slices(img_es,msk_es)
    
    
    mid_img = imgs[1]  
    mid_msk = msks[1]
    bas_img = imgs[0]  
    bas_msk = msks[0]
    ape_img = imgs[2]  
    ape_msk = msks[2]
    
    """
    mid_img = create_chessboard(mid_img.shape[0],mid_img.shape[1],5)
    bas_img = mid_img
    ape_img = mid_img
    """
 
    mid_msk[mid_msk != 2] = 0
    bas_msk[bas_msk != 2] = 0
    ape_msk[ape_msk != 2] = 0
    
    mid_msk[mid_msk == 2] = 1
    bas_msk[bas_msk == 2] = 1
    ape_msk[ape_msk == 2] = 1
    
    """
    plt.imshow(np.hstack((ape_img*ape_msk,mid_img*mid_msk,bas_img*bas_msk)))
    plt.show()
    """
    
    img_size = 120
    half_size= int(120/2)
    
    
    msk_in_bas = crop_or_pad_slice_to_size(bbox(bas_msk,5),img_size,img_size)
    img_in_bas = crop_or_pad_slice_to_size(bbox(bas_img*bas_msk,5),120,120)    
    msk_out_bas = generate_ring(np.zeros_like(msk_in_bas),half_size-10,half_size-20,half_size,half_size)
    img_out_bas,lm_in,lm_out,lm_out_2 = adapt_double_contour(img_in_bas,msk_in_bas,msk_out_bas,n_points)
        #plt.scatter(np.array(lm_in)[:,1],np.array(lm_in)[:,0])
    lm_out.append(lm_out[0])
    #lm_out_2.append(lm_out_2[0])    
    plt.plot(np.array(lm_out)[:,1],np.array(lm_out)[:,0],'r-',linewidth=5)
    #plt.plot(np.array(lm_out_2)[:,1],np.array(lm_out_2)[:,0],'-')


    msk_in_mid = crop_or_pad_slice_to_size(bbox(mid_msk,5),img_size,img_size)
    img_in_mid = crop_or_pad_slice_to_size(bbox(mid_img*mid_msk,5),120,120)    
    msk_out_mid = generate_ring(np.zeros_like(msk_in_mid),half_size-20,half_size-30,half_size,half_size)
    img_out_mid,lm_in,lm_out,_ = adapt_double_contour(img_in_mid,msk_in_mid,msk_out_mid,n_points)
    lm_out.append(lm_out[0])
    plt.plot(np.array(lm_out)[:,1],np.array(lm_out)[:,0],'r-',linewidth=5)
    
    msk_in_ape = crop_or_pad_slice_to_size(bbox(ape_msk,5),img_size,img_size)
    img_in_ape = crop_or_pad_slice_to_size(bbox(ape_img*ape_msk,5),120,120)    
    msk_out_ape = generate_ring(np.zeros_like(msk_in_ape),half_size-30,half_size-40,half_size,half_size)
    img_out_ape,lm_in,lm_out,lm_out_endo = adapt_double_contour(img_in_ape,msk_in_ape,msk_out_ape,n_points)
    lm_out.append(lm_out[0])
    lm_out_endo.append(lm_out_endo[0])    

    plt.plot(np.array(lm_out)[:,1],np.array(lm_out)[:,0],'r-',linewidth=5)
    plt.plot(np.array(lm_out_endo)[:,1],np.array(lm_out_endo)[:,0],'r-',linewidth=5)
    
    img_out = img_out_ape+img_out_mid+img_out_bas
    #img_out = img_out_bas
    
    plt.imshow(img_out)
    #plt.imshow(np.vstack((np.hstack((img_out_ape,img_out_mid,img_out_bas)),np.hstack((img_in_ape,img_in_mid,img_in_bas)))))
    plt.show()
    
    return crop_or_pad_slice_to_size((bbox(img_out,2)/np.max(img_out)),65,65)


def gallery(array, ncols):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result
    