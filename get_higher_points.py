import numpy as np
from scipy import spatial
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

###############################################################################
#                    HIGHER POINTS
###############################################################################

def getHigherPoints_multi(Densities, density_thresh):
    """"
    STRUCTURE OF HIGHER_PTS:
    ['Distance to next highest point, index of higher point, value of current point']
    """

    nCores = g.settings['nCores']
    mx, my = Densities.shape
    idxs = np.where(Densities > density_thresh)
    densities = Densities[idxs]
    densities_jittered = densities + np.arange(len(densities)) / (
    2 * np.float(len(densities)))  # I do this so no two densities are the same, so each cluster has a peak.
    C = np.zeros((mx, my))
    C_idx = np.zeros((mx, my), dtype=np.int)
    idxs = np.vstack((idxs[0], idxs[1])).T
    C[idxs[:, 0], idxs[:, 1]] = densities_jittered
    C_idx[idxs[:, 0], idxs[:, 1]] = np.arange(len(idxs))
    print("Number of pixels to analyze: {}".format(len(idxs)))
    remander = np.arange(len(idxs))
    nTotal_pts = len(idxs)
    block_ends = np.linspace(0, len(remander), nCores + 1).astype(np.int)
    data_blocks = [remander[block_ends[i]:block_ends[i + 1]] for i in np.arange(nCores)]

    # create the ProgressBar object
    args = (nTotal_pts, C, idxs, densities_jittered, C_idx)
    progress = ProgressBar(getHigherPoint_multi, data_blocks, args, nCores, msg='Getting Higher Points')
    if progress.results is None or any(r is None for r in progress.results):
        higher_pts = None
    else:
        higher_pts = np.sum(progress.results, 0)
        progress.clear_memory()
    r = 99
    maxDistance = np.sqrt(2 * r ** 2)
    remander = np.argwhere(higher_pts[:, 0] == 0)
    remander = remander.T[0]
    if len(remander) == 1:
        ii = remander[0]
        higher_pts[ii] = [maxDistance, ii, densities_jittered[ii]]
    elif len(remander) > 1:
        dens2 = densities_jittered[remander]
        idxs2 = idxs[remander]
        D = spatial.distance_matrix(idxs2, idxs2)
        for pt in np.arange(len(D)):
            dd = D[pt, :]
            idx = np.argsort(dd)
            ii = 1
            while dens2[idx[0]] > dens2[idx[ii]]:  # We are searching for the closest point with a higher density
                ii += 1
                if ii == len(dd):  # if this is the most dense point, then no point will have a higher density
                    higher_pts[remander[pt]] = [maxDistance, remander[pt], dens2[pt]]
                    break
            if ii != len(dd):  # if this is the most dense point, then no point will have a higher density
                higher_pts[remander[pt]] = [dd[idx[ii]], remander[idx[ii]], dens2[idx[ii]]]
    return higher_pts, idxs


def getHigherPoint_multi(q_results, q_progress, q_status, child_conn, args):
    remander = child_conn.recv()  # unfortunately this step takes a long time
    percent = 0  # This is the variable we send back which displays our progress
    status = q_status.get(True)  # this blocks the process from running until all processes are launched
    if status == 'Stop':
        q_results.put(None)  # if the user presses stop, return None
    nTotal_pts, C, idxs, densities_jittered, C_idx = args
    mx, my = C.shape
    higher_pts = np.zeros((nTotal_pts, 3))
    for r in np.arange(5, 99, 2):
        mask, center = getMask(r, r)
        oldremander = remander
        remander = []
        percent = 0
        for loop_i, ii in enumerate(oldremander):
            if not q_status.empty():  # check if the stop button has been pressed
                stop = q_status.get(False)
                q_results.put(None)
                return
            if percent < int(100 * loop_i / len(oldremander)):
                percent = int(100 * loop_i / len(oldremander))
                q_progress.put(percent)
            idx = idxs[ii]
            density = densities_jittered[ii]
            x, y = idx
            center2 = np.copy(center)
            x0 = x - center[0]
            xf = x + center[0] + 1
            y0 = y - center[1]
            yf = y + center[1] + 1
            mask2 = np.copy(mask)
            if x0 < 0:
                mask2 = mask2[center[0] - x:, :]
                center2[0] = x
                x0 = 0
            elif xf > mx:
                crop = -(xf - mx)
                if crop < 0:
                    mask2 = mask2[:crop, :]
                xf = mx
            if y0 < 0:
                mask2 = mask2[:, center[1] - y:]
                center2[1] = y
                y0 = 0
            elif yf > my:
                crop = -(yf - my)
                if crop < 0:
                    mask2 = mask2[:, :crop]
                yf = my

            positions = np.array(np.where(mask2 * C[x0:xf, y0:yf] > density)).astype(float).T - center2
            if len(positions) == 0:
                remander.append(ii)
            else:
                distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
                higher_pt = positions[np.argmin(distances)].astype(np.int) + np.array([x0, y0]) + center2
                higher_pt = C_idx[higher_pt[0], higher_pt[1]]
                higher_pt = [np.min(distances), higher_pt, density]
                higher_pts[ii] = higher_pt
    q_results.put(higher_pts)
