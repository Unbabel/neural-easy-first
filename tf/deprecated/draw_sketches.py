import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    epochs = 12
    sent_id = 37 #793
    N = 9 #20

    cols = N
    rows = epochs
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    for epoch in xrange(epochs):

        sketch_file = "sketches/sketches_sent%d_epoch%d.pkl" % (sent_id, epoch)  # sketches for one sample sentence for one epoch
        #sketch_file = "sketches/hmm-nef/sketches_sent%d_epoch%d.pkl" % (sent_id, epoch)  # sketches for one sample sentence for one epoch
        print sketch_file
        # load sketches from pkl object
        all_sketches = pkl.load(open(sketch_file, "rb"))  # shape: (N, sequence len, state_size)

        # color sketches for samples over time
        for i, sketch in enumerate(all_sketches):
            print sketch
            ax = axes[epoch][i] #[epoch*cols+i+1]
            im = ax.imshow(sketch.transpose()[-2:-1], interpolation="none", vmin=0.0, vmax=0.5)
            #ax.set_title("Epoch %d, sketch %d" % (epoch+1, i+1))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


