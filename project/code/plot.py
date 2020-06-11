import numpy as np
import matplotlib.pyplot as plt


def plot_evaluation(save_path, xlabels, data_dict, title="", ylabel="", xlabel="", yticks=None, factor=1, vlines=[], show=False):
    """
    Plots values of a dictionary in a histogram.
    param save_path:        the filename to where the plot is saved
    param xlabels:          the labels used for the x-ticks
    param data_dict:        dictionary containing a key for each library and his metrics
    param title:            [opt] the title of the graph
    param ylabel:           [opt] label for the y-axis
    param xlabel:           [opt] label for the x-axis
    param yticks:           [opt] range representing the used y-ticks
    param factor:           [opt] factor used to multiply the values
    param vlines:           [opt] list to draw vertical red lines on the graph if neccessary
    param show:             [opt] let the graph show and the program wait
    returns:                saves the graph with the given filename
    """ 
    
    plt.rcParams.update({'font.size': 14})
    x = np.arange(len(xlabels))
    width = 0.8
    fig, ax = plt.subplots()
    rectss = []
    i = 0
    for l in sorted(data_dict.keys()):
        values = [round(v * factor, 1) for v in list(data_dict[l].values())]
        rects = ax.bar(x - width/2 + (i+0.5)*width/len(data_dict), values, width/len(data_dict), label=l)
        rectss.append(rects)
        i += 1

    #lines for pose evaluation
    for v in vlines:
        ax.axvline(x=v, color='r', linestyle='dashed', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.legend()


    for rects in rectss:
        for rect in rects:
            height = rect.get_height()
            ax.annotate("{}".format(int(round(height))),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize='small')

    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    

def plot_track(save_path, data_dict, title, fps=1, xlabel=None, ylabel=None, show=False):
    """
    Plot the track of certain values in different graphs
    param save_path:        the filename to where the plot is saved
    param data_dict:        dictionary containing a key for each list of points to track
    param title:            the title of the graph
    param fps:              [opt] the frames per second of the data track, this is used to calculate the corresponding time scale
    param xlabel:           [opt] label for the x-axis
    param ylabel:           [opt] label for the y-axis
    param show:             [opt] let the graph show and the program wait
    returns:                saves the graph with the given filename
    """ 
    plt.rcParams.update({'font.size': 14})
    #fig, ax = plt.subplots()
    #ax.set_title(title)
    fig, axs = plt.subplots(len(data_dict), sharex=True)
    fig.suptitle(title, fontsize=22)

    keys = [d for d in data_dict]
    for i in range(len(keys)):
        k = keys[i]
        d = data_dict[k]
        x = np.arange(len(d))/fps
        y_x = d[:,0]
        y_y = d[:,1]

        roundabout = 10.0
        dif = max( max(y_y) - min(y_y) , max(y_x) - min(y_x) )
        dif = round(dif/roundabout) * roundabout

        a1 = roundabout * round((min(y_x)-roundabout)/roundabout)
        b1 = a1 + dif + 2*roundabout
        
        a2 = roundabout * round((min(y_y)-roundabout)/roundabout)
        b2 = a2 + dif + 2*roundabout

        #print(dif)
        #print(a1, b1, a2, b2)

        lns1 = axs[i].plot(x, y_x, label=k + " X")
        axs[i].set_ylim([a1,b1])
        
        b = axs[i].twinx()
        lns2 = b.plot(x, y_y, label=k + " Y", color='tab:red')
        b.set_ylim([a2,b2])

        axs[i].set_title(k)
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        axs[i].legend(lns, labs, loc=0)

        axs[i].set_ylabel(ylabel + " X")
        b.set_ylabel(ylabel + " Y")
        
    
    for ax in axs.flat:
        ax.set(xlabel=xlabel)
        ax.label_outer()

    #for d in data_dict: 
    #    x = np.arange(len(data_dict[d]))/fps
    #    y1 = data_dict[d][:,0]
    #    y2 = data_dict[d][:,1]
    #    ax.plot(x, y1, label=d + " X")
    #    ax.plot(x, y2, label=d + " Y")
    #ax.legend()
    if show:
        plt.show()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.07)
    fig.savefig(save_path)
    plt.close(fig)


def plot_single_values(save_path, data_dict, title, fps=1, xlabel=None, ylabel=None, show=False): 
    """
    Plot the track of certain values in different graphs
    param save_path:        the filename to where the plot is saved
    param data_dict:        dictionary containing a key for each list of points to track
    param title:            the title of the graph
    param fps:              [opt] the frames per second of the data track, this is used to calculate the corresponding time scale
    param xlabel:           [opt] label for the x-axis
    param ylabel:           [opt] label for the y-axis
    param show:             [opt] let the graph show and the program wait
    returns:                saves the graph with the given filename
    """ 
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(len(data_dict), sharex=True)
    fig.suptitle(title, fontsize=22)
    keys = [d for d in data_dict]
    for i in range(len(keys)):
        k = keys[i]
        d = data_dict[k]
        x = np.arange(len(d))/fps
        axs[i].plot(x, d, label=d)
        axs[i].set_title(k)
    
    for ax in axs.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.label_outer()

    if show:
        plt.show()
    fig.set_size_inches(18.5, 10.5, forward=True)
    #fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.07)
    fig.savefig(save_path)
    plt.close(fig)