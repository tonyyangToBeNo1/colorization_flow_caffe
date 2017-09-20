from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label', loss_layer='loss'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        if(not os.path.exists(save_dir)):
            os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs[loss_layer].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label', loss_layer='loss', f=-1):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt, loss_layer=loss_layer, f=f)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label', loss_layer='loss', f=-1):
    if(np.array(f).flatten()[0]==-1):
        PRINT_TO_FILE = False # print to string
    else:
        PRINT_TO_FILE = True

    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt, loss_layer=loss_layer)

    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()

    if(PRINT_TO_FILE):
        f.write('Iteration: %i\n'%(iter))
        f.write('Loss: %.4f\n'%loss)

        acc = np.diag(hist).sum() / hist.sum()
        f.write('Overall Accuracy: %.4f\n'%(acc))
        # f.write('Iteration' + iter + 'overall accuracy' + acc)

        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        f.write('Mean Accuracy: %.4f\n'%(np.nanmean(acc)))

        # per-class IU
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        f.write('Mean IU: %.4f\n'%(np.nanmean(iu)))
        # f.write('Iteration' + iter + 'mean IU' + np.nanmean(iu))

        freq = hist.sum(1) / hist.sum()
        f.write('fwavacc: %.4f\n'%(freq[freq > 0] * iu[freq > 0]).sum())
        f.write('\n')
