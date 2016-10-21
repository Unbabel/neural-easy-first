import cPickle as pkl
import matplotlib.pyplot as plt
import argparse


def main(args):
    """
    Plot the attention (and sketches) for a single example across epochs and sketches
    :param args:
    :return:
    """
    cols = args.sent_len
    rows = args.epochs
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    for epoch in xrange(args.epochs):

        sketch_file = args.sketch_dir+'/sketches_sent%d_epoch%d.pkl' % (args.sent_id, epoch)  # sketches for one sample sentence for one epoch
        # load sketches from pkl object
        all_sketches = pkl.load(open(sketch_file, "rb"))  # shape: (N, sequence len, sketch_size+1)
        # color sketches for samples over time
        for i, sketch in enumerate(all_sketches):
            ax = axes[epoch][i] #[epoch*cols+i+1]
            print args.plot_sketches
            if args.plot_sketches:
                plot_values = sketch
            else:
                plot_values = sketch[:,-1:]  # just plot attention
            im = ax.imshow(plot_values.transpose(), interpolation="none", vmin=0.0, vmax=0.5)
            ax.get_yaxis().set_visible(False)
            if epoch < args.epochs-1:
                ax.get_xaxis().set_visible(False)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot sketches for example sentence')
    parser.add_argument('epochs', type=int, help='number of epochs to plot')
    parser.add_argument('sent_id', type=int, help='sentence to plot')
    parser.add_argument('sent_len', type=int, help='length of sentence')
    parser.add_argument('sketch_dir', type=str, help='directory where sketches are dumped')
    parser.add_argument('--plot_sketches', action='store_true', help='plot sketches in addition to the attention')
    args = parser.parse_args()
    main(args)




