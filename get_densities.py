import numpy as np
import flika
from flika import global_vars as g
from flika.process.progress_bar import ProgressBar


def getMask(nx=5,ny=5):
    mask=np.zeros((nx,ny))
    center=np.array([(nx-1)/2, (ny-1)/2]).astype(np.int)
    x0,y0=center
    for x in np.arange(nx):
        for y in np.arange(ny):
            if ((x-x0)**2) / (x0**2) + ((y-y0)**2) / (y0**2) <= 1:
                mask[x,y] = 1
    return mask, center

def getDensities_single(Image, thresh, mask_radius):
    if mask_radius % 2 == 0:
        mask_radius += 1
    nCores = g.settings['nCores']
    pxls = np.array(np.where(Image > thresh)).T
    result = np.zeros(Image.shape)
    mx, my = Image.shape
    mask, center = getMask(mask_radius, mask_radius)
    mask_idx = np.where(mask)
    for i, pxl in enumerate(pxls):
        x, y = pxl
        square_cutout = Image[x - center[0]:x + center[0] + 1, y - center[1]:y + center[1] + 1]
        try:
            cutout = square_cutout[mask_idx]
        except IndexError:
            x0 = x - center[0]
            xf = x + center[0] + 1
            y0 = y - center[1]
            yf = y + center[1] + 1
            mask2 = mask
            if x0 < 0:
                mask2 = mask2[center[0] - x:, :]
                x0 = 0
            if y0 < 0:
                mask2 = mask2[:, center[1] - y:]
                y0 = 0
            if xf > mx - 1:
                mask2 = mask2[:-(xf - mx + 1), :]
                xf = mx - 1
            if yf > my - 1:
                mask2 = mask2[:, :-(yf - my + 1)]
                yf = my - 1
            mask2_idx = np.where(mask2)
            square_cutout = Image[x0:xf, y0:yf]
            cutout = square_cutout[mask2_idx]
        pxl_val = Image[x, y]
        result[x, y] = np.sum(cutout[np.logical_and(cutout < pxl_val, cutout > 0)])
    return result

def getDensities_multi(Image, thresh, mask_radius):
    if mask_radius % 2 == 0:
        mask_radius += 1
    nCores = g.settings['nCores']
    pxls = np.array(np.where(Image > thresh)).T
    block_ends = np.linspace(0, len(pxls), nCores + 1).astype(np.int)
    data_blocks = [pxls[block_ends[i]:block_ends[i + 1]] for i in np.arange(nCores)]
    args = (Image, mask_radius)
    progress = ProgressBar(calcDensity, data_blocks, args, nCores, msg='Calculating Density')
    if progress.results is None or any(r is None for r in progress.results):
        result = None
    else:
        result = progress.results
        result = np.sum(result, 0)
    return result


def calcDensity(q_results, q_progress, q_status, child_conn, args):
    pxls = child_conn.recv()  # unfortunately this step takes a long time
    percent = 0  # This is the variable we send back which displays our progress
    status = q_status.get(True)  # this blocks the process from running until all processes are launched
    if status == 'Stop':
        q_results.put(None)  # if the user presses stop, return None

    ''' For testing: 
    Image=A_norm
    mx,my=Image.shape
    mask,center=getMask(mask_radius,mask_radius)
    mask_idx=np.where(mask)
    pxls=np.array(np.where(Image>thresh)).T
    i=0
    pxl=pxls[100000]
    x,y=pxl
    square_cutout=Image[x-center[0]:x+center[0]+1,y-center[1]:y+center[1]+1]
    cutout=square_cutout[mask_idx]
    '''

    Image, mask_radius = args  # unpack all the variables inside the args tuple
    result = np.zeros(Image.shape)
    mx, my = Image.shape
    mask, center = getMask(mask_radius, mask_radius)
    mask_idx = np.where(mask)
    for i, pxl in enumerate(pxls):
        x, y = pxl
        square_cutout = Image[x - center[0]:x + center[0] + 1, y - center[1]:y + center[1] + 1]
        try:
            cutout = square_cutout[mask_idx]
        except IndexError:
            x0 = x - center[0]
            xf = x + center[0] + 1
            y0 = y - center[1]
            yf = y + center[1] + 1
            mask2 = mask
            if x0 < 0:
                mask2 = mask2[center[0] - x:, :]
                x0 = 0
            if y0 < 0:
                mask2 = mask2[:, center[1] - y:]
                y0 = 0
            if xf > mx - 1:
                mask2 = mask2[:-(xf - mx + 1), :]
                xf = mx - 1
            if yf > my - 1:
                mask2 = mask2[:, :-(yf - my + 1)]
                yf = my - 1
            mask2_idx = np.where(mask2)
            square_cutout = Image[x0:xf, y0:yf]
            cutout = square_cutout[mask2_idx]
        pxl_val = Image[x, y]
        result[x, y] = np.sum(cutout[np.logical_and(cutout < pxl_val, cutout > 0)])

        if percent < int(100 * i / len(pxls)):
            percent = int(100 * i / len(pxls))
            q_progress.put(percent)
        if not q_status.empty():  # check if the stop button has been pressed
            stop = q_status.get(False)
            q_results.put(None)
            return
            # finally, when we've finished with our calculation, we send back the result
    q_results.put(result)
