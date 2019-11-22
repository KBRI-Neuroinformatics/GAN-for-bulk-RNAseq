import matplotlib.pyplot as plt
import numpy as np

def plot_scatter_cond(dat, cond1, cond2=0):
    if type(cond2)==int:
        cond2=cond1
    
    plt.figure(figsize=(16,20))
    for y in range(6):
        for x in range(6):
            plt.subplot(6,6,y*6+x+1)
            plt.scatter(dat[cond1[y],:],dat[cond2[x],:])

def plot_tsne(initx, medx1, medx2, endx, n_tr=762, n_te=84):
    plt.figure(figsize=(16,3))
    plt.subplot(141)
    plt.title("500 epoch", fontsize=14)
    plt.scatter(initx[:n_tr,0], initx[:n_tr,1], alpha=0.5, c='red')
    plt.scatter(initx[n_tr:n_tr+n_te,0], initx[n_tr:n_tr+n_te,1], alpha=0.5, c='blue')
    plt.scatter(initx[n_tr+n_te:n_tr+2*n_te,0], initx[n_tr+n_te:n_tr+2*n_te,1], alpha=0.5, c='orange')
    plt.xlabel('tSNE dimension1', fontsize=13)
    plt.ylabel('tSNE dimension2', fontsize=13)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(142)
    plt.title("10K epoch", fontsize=14)
    plt.scatter(medx1[:n_tr,0], medx1[:n_tr,1], alpha=0.5, c='red')
    plt.scatter(medx1[n_tr:n_tr+n_te,0], medx1[n_tr:n_tr+n_te,1], alpha=0.5, c='blue')
    plt.scatter(medx1[n_tr+n_te:n_tr+2*n_te,0], medx1[n_tr+n_te:n_tr+2*n_te,1], alpha=0.5, c='orange')
    plt.xlabel('tSNE dimension1', fontsize=13)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(143)
    plt.title("25K epoch", fontsize=14)
    plt.scatter(medx2[:n_tr,0], medx2[:n_tr,1], alpha=0.5, c='red')
    plt.scatter(medx2[n_tr:n_tr+n_te,0], medx2[n_tr:n_tr+n_te,1], alpha=0.5, c='blue')
    plt.scatter(medx2[n_tr+n_te:n_tr+2*n_te,0], medx2[n_tr+n_te:n_tr+2*n_te,1], alpha=0.5, c='orange')
    plt.xlabel('tSNE dimension1', fontsize=13)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.subplot(144)
    plt.title("100K epoch", fontsize=14)
    plt.scatter(endx[:n_tr,0], endx[:n_tr,1], alpha=0.5, c='red')
    plt.scatter(endx[n_tr:n_tr+n_te,0], endx[n_tr:n_tr+n_te,1], alpha=0.5, c='blue')
    plt.scatter(endx[n_tr+n_te:n_tr+2*n_te,0], endx[n_tr+n_te:n_tr+2*n_te,1], alpha=0.5, c='orange')
    plt.xlabel('tSNE dimension1', fontsize=14)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.subplots_adjust(wspace=0.23)
    plt.legend(['Training','Test','Generated'], ncol=3, bbox_to_anchor=(-.5, -.3), fontsize=14)


def plot_hist_re_rld(real, fake, bins):
    plt.figure(figsize=(20,7))
    plt.suptitle('Histogram of rescaled RLD', fontsize=36, y=1.03)
    plt.subplot(121)
    plt.hist(real.flatten(), bins=bins)
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 10000)
    plt.xticks(fontsize=34)
    plt.yticks([0,2000,4000,6000,8000,10000], ['0', '2K', '4K', '6K', '8K', '10K'], fontsize=34)
    plt.xlabel('Rescaled RLD', fontsize=35)
    plt.ylabel('Counts', fontsize=35)
    plt.title('Real', fontsize=35)

    plt.subplot(122)
    plt.subplots_adjust(wspace=0.05)
    plt.hist(fake.flatten(), bins=bins, color='red')
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 10000)
    plt.xticks(fontsize=34)
    plt.yticks([0,2000,4000,6000,8000,10000],[])
    plt.xlabel('Rescaled RLD', fontsize=35)
    plt.title('Generated', fontsize=35)
    plt.plot()

def plot_hist_corr(real, fake, bins=100):
    plt.figure(figsize=(20,7))
    plt.suptitle('Histogram of correlation coefficient', fontsize=36, y=1.03)
    plt.subplot(121)
    plt.hist(real.flatten(), bins=bins)
    plt.xlim(-0.8,1.)
    plt.xticks([-0.75, -0.4, 0.0, 0.4, 0.75, 1.], fontsize=31)
    plt.yticks([0, 4000, 8000, 12000, 16000], ['0', '4K', '8K', '12K', '16K'], fontsize=34)
    plt.xlabel('Correlation', fontsize=35)
    plt.ylabel('Counts', fontsize=35)
    plt.title('Real vs. Real', fontsize=35)

    plt.subplot(122)
    plt.subplots_adjust(wspace=0.12)
    plt.hist(fake.flatten(), bins=bins, color='red')
    plt.xlim(-0.8,1.)
    plt.xticks([-0.75, -0.4, 0.0, 0.4, 0.75, 1.], fontsize=31)
    plt.yticks([])
    plt.xlabel('Correlation', fontsize=35)
    plt.title('Real vs. Generated', fontsize=35)
    plt.plot()

def plot_comp_real_and_genx(realx, genx, clist, idx):
    plt.figure(figsize=(16,3))
    plt.suptitle('Six 7M AD samples', fontsize=14)
    for i in range(len(clist[idx])): #AD_7M
        plt.subplot(1,6,i+1)
        plt.subplots_adjust(wspace=0.3)
        plt.scatter(realx[clist[idx][i]], genx[(6*idx)+i])
        plt.plot([0,1], color='red')
        plt.xlim(-1, 2.4)
        plt.ylim(-1, 2.4)
        #plt.yticks(np.arange(-1,3,1), [])
        plt.yticks(np.arange(-1,3,1), fontsize=12)
        plt.xticks(np.arange(-1,3,1), fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlabel('Real', fontsize=13)
        if i==0:
            plt.ylabel('Generated', fontsize=13)
            plt.yticks(np.arange(-1,3,1), fontsize=12)

def plot_generated_kwd(dat, genx, genc, org_clist, glists, kwd) :
    kwdidx = np.where(glists==kwd)[0][0]
    xrange = np.concatenate([np.arange(len(org_clist))+1]*len(org_clist))
    clist = np.array(['coral', 'red', 'maroon', 'turquoise', 'royalblue', 'navy'])

    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(111)
    #scatter
    ax1.set_ylim(0.8, 1.0)
    ax1.set_yticks(np.arange(0.8, 1.05, 0.05))
    ax1.set_yticklabels(np.arange(0.8, 1.05, 0.05).astype(np.float16), fontsize=15)
    ax1.set_xlim(-0.5, len(genc))
    ax1.set_xticks(np.arange(len(genc)))
    ax1.set_xticklabels(xrange, fontsize=15)
    ax1.grid(linestyle='--')
    for ns in range(genc.shape[0]):
        cls = ns//6
        tx = np.full(genc[ns].shape, ns)
        ax1.scatter(tx, genc[ns], c=clist[cls], edgecolors='black', linewidth='0.2')

    #bar
    ax2 = ax1.twinx()
    ay2 = ax1.twiny()
    ay2.set_xlim(-0.5, len(genc))
    ax2.set_ylim(0., 2.5)
    ax2.set_yticklabels(np.arange(0., 3., 0.5), fontsize=15)
    for ns in range(genx.shape[0]):
        cls = ns//6
        ax2.bar(ns, dat[org_clist[cls][ns%6],kwdidx], width=0.2, color=clist[cls])
        ax2.bar(ns+.25, genx[ns,kwdidx], width=0.2, color='tan')

    #secondary ticks
    ay2.set_xticks(np.array([-0.5, 2.7, 5.5, 8.7, 11.5, 14.7, 17.5, 20.7, 23.5, 26.7, 29.5, 32.7, 36]))
    ay2.set_xticklabels(['','AD 2M','', 'AD 4M', '', 'AD 7M', '', 'WT 2M', '', 'WT 4M', '', 'WT 7M', ''])
    ay2.xaxis.set_ticks_position('bottom')
    ay2.spines['bottom'].set_position(('outward', 40))

    #label
    ax1.set_ylabel('Correlation between \nreal and generated samples', fontsize=16)
    ax2.set_ylabel('Rescaled RLD ('+kwd+')', fontsize=16)
    ax1.set_xlabel('Samples', fontsize=16)
    ax1.yaxis.set_label_coords(-0.055, 0.5)
    ax2.yaxis.set_label_coords(1.05, 0.5)
    ax1.xaxis.set_label_coords(.5, -0.25)
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.show()

def plot_transition_curves(genx_A, genx_S, glists, flist):
    ncol = len(flist)
    plt.figure(figsize=(6*ncol,5))
    for fl in range(len(flist)):
        plt.subplot(1,ncol,fl+1)
        plt.subplots_adjust(wspace=0.2)

        for gs in range(len(flist[fl])):
            idx_ = np.where(glists==flist[fl][gs])[0][0]
            plt.plot(genx_A[2][:,idx_], label=flist[fl][gs], linewidth=3)
            y1, y2 = np.reshape(genx_A[2][:,idx_]-genx_S[2][:,idx_], (101)), np.reshape(genx_A[2][:,idx_]+genx_S[2][:,idx_], (101))
            plt.fill_between(range(101), y1, y2, alpha=0.2)
        plt.xticks([0,100], ['WT','AD'], fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('Rescaled RLD', fontsize=16)
        plt.legend(fontsize=14, loc='upper right')
    plt.plot()
