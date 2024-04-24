import matplotlib.pyplot as plt
from daffodillib.Board import controller
import daffodillib.IVcurve as IVcurve
import daffodillib.read_array as read_array
import daffodillib.outerproduct as outerproduct
import numpy as np

def extract_Gnorm_Gmaps(board, layer_offsets, weight_shapes):

    Gnorms = []

    for idx, i in enumerate(layer_offsets):
        Gon_avgs = []
        Goff_avgs = []
        for neg, pos in i:
            k, x, y, _ = neg
            setG = np.array(board.sim_device.all_kernels[k].setG)
            Gon_avgs.append(np.average(setG[x:x+weight_shapes[idx][0], y:y+weight_shapes[idx][1] // 2]))
            resetG = np.array(board.sim_device.all_kernels[k].resetG)
            Goff_avgs.append(np.average(resetG[x:x+weight_shapes[idx][0], y:y+weight_shapes[idx][1] // 2]))

            k, x, y, _ = pos
            setG = np.array(board.sim_device.all_kernels[k].setG)
            Gon_avgs.append(np.average(setG[x:x+weight_shapes[idx][0], y:y+weight_shapes[idx][1] // 2]))
            resetG = np.array(board.sim_device.all_kernels[k].resetG)
            Goff_avgs.append(np.average(resetG[x:x+weight_shapes[idx][0], y:y+weight_shapes[idx][1] // 2]))

        Gnorms.append(np.average(np.array(Gon_avgs) - np.array(Goff_avgs)).tolist())

    print('Gnorms (layer wise)', Gnorms)
    return Gnorms

def disturb_states_random(layer_ensemble, perturb_percentage):
    for layer in layer_ensemble.layers:
        assert layer.board.sim_device and layer.board.sim_device.name == 'MTJ'
        devices_disturbed = int((perturb_percentage / 100) * layer.weight_shape[0] * layer.weight_shape[1] // 2)
        assert layer.mode == 'block_split'
        # Negative half
        neg_kernel = layer.board.sim_device.all_kernels[layer.neg_kernel]
        neg_idxs = []
        for i in range(devices_disturbed // 2):
            random_index1 = np.random.randint(0, layer.weight_shape[0])
            random_index2 = np.random.randint(0, layer.weight_shape[1] // 2)
            idx = [random_index1, random_index2]
            while (idx in neg_idxs):
                random_index1 = np.random.randint(0, layer.weight_shape[0])
                random_index2 = np.random.randint(0, layer.weight_shape[1] // 2)
                idx = [random_index1, random_index2]
            # new idx found
            neg_idxs.append(idx)
            kernel_idx = [idx[0]+layer.neg_xoffset, idx[1]+layer.neg_yoffset]
            neg_kernel.kern[kernel_idx[0]][kernel_idx[1]] = 500

        # Positive half
        pos_kernel = layer.board.sim_device.all_kernels[layer.pos_kernel]
        pos_idxs = []
        for i in range(devices_disturbed // 2):
            random_index1 = np.random.randint(0, layer.weight_shape[0])
            random_index2 = np.random.randint(0, layer.weight_shape[1] // 2)
            idx = [random_index1, random_index2]
            while (idx in pos_idxs):
                random_index1 = np.random.randint(0, layer.weight_shape[0])
                random_index2 = np.random.randint(0, layer.weight_shape[1] // 2)
                idx = [random_index1, random_index2]
            # new idx found
            pos_idxs.append(idx)
            kernel_idx = [idx[0]+layer.pos_xoffset, idx[1]+layer.pos_yoffset]
            pos_kernel.kern[kernel_idx[0]][kernel_idx[1]] = 500


        print('pos idxs', pos_idxs)
        print('neg idxs', neg_idxs)

    return

def disturb_chip_random(board, perturb_percentage):
   
    assert board.sim_device and board.sim_device.name == 'MTJ'
    devices_disturbed = int((perturb_percentage / 100) * board.xdim * board.ydim)
    print('devices to disturb / kernel', devices_disturbed)

    for kernel in range(32): # kernels to alter
        board_kernel = board.sim_device.all_kernels[kernel]
        idxs = []
        for i in range(devices_disturbed):
            random_index1 = np.random.randint(0, board.xdim)
            random_index2 = np.random.randint(0, board.ydim)
            idx = [random_index1, random_index2]
            while (idx in idxs):
                random_index1 = np.random.randint(0, board.xdim)
                random_index2 = np.random.randint(0, board.ydim)
                idx = [random_index1, random_index2]
            # new idx found
            idxs.append(idx)
            # simulate stuck ON
            board_kernel.kern[idx[0]][idx[1]] = 500
            board_kernel.setG[idx[0]][idx[1]] = 500
            board_kernel.resetG[idx[0]][idx[1]] = 500
    return

def overlap(a,b):
    return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]

def check_offset_validity(offsets, weight_shapes, debug=False):
    '''
    Written for block_split layer mode + forward configuration
    Probably will produce incorrect results for other configs
    offsets fmt = [ [(k1, x1, y1), (k2, x2, y2)] * layer_count ]
    '''
    offsets = np.copy(offsets).tolist()
    flattened_offsets = []
    # add weight shapes (num_cols, num_rows) to each offset
    for i in range(len(offsets)):
        for j in range(len(offsets[0])):
            for k in range(len(offsets[0][0])):
                temp = offsets[i][j][k]
                temp += [weight_shapes[i][0], weight_shapes[i][1] // 2]
                flattened_offsets.append(temp)

    # sort by kernel
    sorted_offsets = sorted(flattened_offsets, key=lambda x: x[0])

    # for each offset in a given layer, ensure that there are no collisions with any other offsets
    for curr_idx in range(len(sorted_offsets)):
        curr_k, curr_x, curr_y, curr_x_n, curr_y_n = sorted_offsets[curr_idx]
        curr_xs = [curr_x, curr_x+curr_x_n]
        curr_ys = [curr_y, curr_y+curr_y_n]
        for next_idx in range(curr_idx+1, len(sorted_offsets)):
            next_k, next_x, next_y, next_x_n, next_y_n = sorted_offsets[next_idx]
            if (curr_k != next_k): continue # kernels different, no collision possible here
            else:
                next_xs = [next_x, next_x+next_x_n]
                next_ys = [next_y, next_y+next_y_n]
                if (overlap(curr_xs, next_xs) and overlap(curr_ys, next_ys)):
                    if (debug): print('overlap, offsets invalid.')
                    return False
    return True

def check_offset_validity_simple(offsets, debug=False):

    # For flattened offsetss
    # sort by kernel
    sorted_offsets = sorted(offsets, key=lambda x: x[0])

    # for each offset in a given layer, ensure that there are no collisions with any other offsets
    for curr_idx in range(len(sorted_offsets)):
        curr_k, curr_x, curr_y, curr_x_n, curr_y_n, currStuck = sorted_offsets[curr_idx]
        curr_xs = [curr_x, curr_x+curr_x_n]
        curr_ys = [curr_y, curr_y+curr_y_n]
        for next_idx in range(curr_idx+1, len(sorted_offsets)):
            next_k, next_x, next_y, next_x_n, next_y_n, nextStuck = sorted_offsets[next_idx]
            if (curr_k != next_k): continue # kernels different, no collision possible here
            else:
                next_xs = [next_x, next_x+next_x_n]
                next_ys = [next_y, next_y+next_y_n]
                if (overlap(curr_xs, next_xs) and overlap(curr_ys, next_ys)):
                    if (debug): print('overlap, offsets invalid.')
                    return False
    return True

def find_ensemble_offsets_sim(board, shape, kernels, offsets_to_avoid, k=1, debug=False):
    '''
    Only use with Daffodil Sim
    k: # each n dimension (for m x n) layer should have at least one instance in ensemble with no stuck devices
    '''
    shape = [shape[0], shape[1] // 2] # block split mode

    # Todo: skip all offsets that would collide with offsets found so far
    offsets_so_far = []
    stuck_on_threshold = 400
    stuck_off_threshold = 10

    assert k >= 1

    stats = []

    success = False 

    # The first layer in the ensemble should be the one that has the minimum # of stuck devices altogether
    for kernel in kernels: # kernels to alter
        board_kernel = board.sim_device.all_kernels[kernel]

        setG = np.array(board_kernel.setG)
        resetG = np.array(board_kernel.resetG)

        for x in range(0, board.xdim - shape[0] + 1):
            for y in range(0, board.ydim - shape[1] + 1):

                setSubkernel = setG[x:x+shape[0], y:y+shape[1]]
                resetSubkernel = resetG[x:x+shape[0], y:y+shape[1]]

                # are there any shorts on the kernel? directly disregard such a col
                stuckOffCount = np.sum(np.maximum(np.sum(setSubkernel <= stuck_off_threshold, axis=0), np.sum(resetSubkernel <= stuck_off_threshold, axis=0)))
                if (stuckOffCount > 0): continue # shorts, avoid

                diff = setSubkernel - resetSubkernel
                stuckCount = np.sum(diff[(diff <= stuck_off_threshold) & (diff > 0)], axis=0) # second condition useful to avoid catching disturbed rows
                if (stuckCount > 0): continue # shorts, avoid

                setMapStuck = np.sum(setSubkernel > stuck_on_threshold, axis=0)
                resetMapStuck = np.sum(resetSubkernel > stuck_on_threshold, axis=0)
                
                maxMapStuck = np.maximum(setMapStuck, resetMapStuck)

                # if the current offset has any overlap with offsets to avoid, simply skip
                current_offset = [kernel, x, y, shape[0], shape[1], maxMapStuck.tolist()]

                temp_offset_list = np.copy(offsets_to_avoid).tolist()
                temp_offset_list.append(current_offset)

                if (not check_offset_validity_simple(temp_offset_list, debug)):
                    continue

                stuckCount = np.sum(maxMapStuck)
                stats.append((stuckCount, kernel, x, y, shape[0], shape[1], maxMapStuck.tolist()))

    stats = sorted(stats, key=lambda x: x[0], reverse=False) # sort by increasing # of stuck devices
    stuckCount, best_kernel, best_x, best_y, shape0, shape1, mapStuck = stats[0]
    stats = stats[1:]

    dims_remaining = np.ones(shape[1]) * k
    zeros = np.zeros(shape[1])

    # Add best found map in the given kernels to the offset list
    offsets_so_far.append([best_kernel, best_x, best_y, shape0, shape1, mapStuck])
    # Update count of dims so far - if the layer has a good row, decrement the amount required
    for i in range(shape[1]):
        if (mapStuck[i] == 0): dims_remaining[i] -= 1
    no_change = False

    dims_old = np.copy(dims_remaining)
    while not np.equal(dims_remaining, zeros).all() and not no_change:
    
        # Now in the remaining sorted offsets, we will continue to add until our list is exhausted

        for candidate_offset in stats:
            stuckCount, k, x, y, shape0, shape1, mapStuck = candidate_offset

            temp_offset_list = np.copy(offsets_so_far).tolist()
            temp_offset_list.append([k, x, y, shape0, shape1, mapStuck])

            if (not check_offset_validity_simple(temp_offset_list, debug)):
                if (debug): print('collision with existing offsets, skipping')
                continue
            else:


                # Valid candidate, but is it useful?
                if (debug): print('no collision', candidate_offset)
                if (debug): print(dims_remaining)
                # It would be useful if it adds to a defective dimensions
                useful = False
                for i in range(shape[1]):
                    if (dims_remaining[i] > 0): # if there's a dim remaining
                        if (mapStuck[i] == 0): # and if there's no defects for this
                            useful = True

                if not (useful):
                    if (debug): print('Dimension not useful, skipping')
                    continue

                # We have a dimension that is useful, add to our offset list

                offsets_so_far.append([k, x, y, shape0, shape1, mapStuck])
                # Update count of dims so far - if the layer has a good row, decrement the amount required
                for i in range(shape[1]):
                    if (mapStuck[i] == 0): 
                        dims_remaining[i] = max(0, dims_remaining[i] - 1)          

            if (np.equal(dims_remaining, zeros).all()):
                break # all conditions satisfied, break out

        if (np.equal(dims_old, dims_remaining).all()):
            if (debug): print('No Change!! Unable to satisfy conditions. Add more kernels for this layer.')
            no_change = True

        # for next iteration
        dims_old = np.copy(dims_remaining)

    if (np.equal(dims_remaining, zeros).all()):
        print("All conditions satisfied!!!")
        success = True

    if (debug):
        print(offsets_so_far)
    return offsets_so_far, success
