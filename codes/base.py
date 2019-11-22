import numpy as np

def load_dataset(dpath):
    #load from .npz
    dat = np.load(dpath)
    dat_rld, dat_ages, dat_types = dat['values'], dat['ages'], dat['types']
    dat_glist, dat_gender, dat_region = dat['genelist'], dat['genders'], dat['regions']
    dat_fromwhere = dat['fromwhere']

    #extract WT and AD
    rlds = np.concatenate((dat_rld[:423], dat_rld[1269:1692]), axis=0)
    ages = np.concatenate((dat_ages[:423], dat_ages[1269:1692]), axis=0)
    types = np.concatenate((dat_types[:423], dat_types[1269:1692]), axis=0)
    genders = np.concatenate((dat_gender[:423], dat_gender[1269:1692]), axis=0)
    regions = np.concatenate((dat_region[:423], dat_region[1269:1692]), axis=0)
    #print ('raw_rld:', rlds.shape)

    #extract DEG_7M genes
    idx_7m = np.where(dat_fromwhere!='4M')[0]
    #print ('len of DEG_7M:', len(idx_7m))
    rlds, glists = rlds[:,idx_7m], dat_glist[idx_7m]
    #print ('ext_rld:', rlds.shape, 'glist:', glists.shape)
    
    return rlds, ages, types, glists, genders, regions

def rescaling(rlds):
    #indexing
    AD_2M = np.array([0,46,83,111,130,140])
    AD_4M, AD_7M = np.add(AD_2M, 141), np.add(AD_2M,2*141)
    WT_2M, WT_4M, WT_7M = np.add(AD_2M, 3*141), np.add(AD_2M,4*141), np.add(AD_2M,5*141)

    Aug_AD_2M = np.arange(141)
    Aug_AD_4M, Aug_AD_7M = np.add(Aug_AD_2M, 141), np.add(Aug_AD_2M,2*141)
    Aug_WT_2M, Aug_WT_4M, Aug_WT_7M = np.add(Aug_AD_2M,3*141), np.add(Aug_AD_2M,4*141), np.add(Aug_AD_2M,5*141)

    #rescaling
    cond_list = [Aug_AD_2M,Aug_AD_4M,Aug_AD_7M,Aug_WT_2M,Aug_WT_4M,Aug_WT_7M]
    rld_mean = np.average(rlds, axis=0)

    rld_std = np.zeros((len(cond_list),rlds.shape[1]))
    for c in range(len(cond_list)):
        rld_std[c] = np.std(rlds[cond_list[c]], axis=0)
    max_rld_std = np.max(rld_std, axis=0)
    #print (rld_mean.shape, max_rld_std.shape)

    re_rld = (rlds-rld_mean)/max_rld_std
    std_re_rld = np.std(re_rld.flatten())
    re_rld = re_rld/(2*1.959*std_re_rld)+0.5
    #print (re_rld.shape)
    return re_rld, cond_list, [AD_2M, AD_4M, AD_7M, WT_2M, WT_4M, WT_7M]

def data_flatten(z_, avgz_, genx_, gencorr_):
    """
    #z:(eps,types(AD2/4/7M,WT2/4/7M),#augsamples,zdim)
    #avgz:(eps,ages(2/4/7M),zdim)
    #genx:(eps,ages(2/4/7M),#augsamples,timesteps,p)
    #gencorr:(eps,types(AD2/4/7M,WT2/4/7M),#augsamples)
    """
    #z
    WTADz=[]
    for ep in range(len(z_)):
        WTADz.append(np.concatenate((z_[ep][1], z_[ep][0]),axis=0))
    WTADz=np.array(WTADz)
    #print (WTADz.shape)
    
    #gen_corr
    WTADgcorr=[]
    for ep in range(len(gencorr_)):
        WTADgcorr.append(np.concatenate((gencorr_[ep][1], gencorr_[ep][0]),axis=0))
    WTADgcorr=np.array(WTADgcorr)
    #print (WTADgcorr.shape)
    return WTADz, avgz_, genx_, WTADgcorr

def ext_realv(z_, gencorr_, org_clist):
    #extract related to realx not augmentedx
    WTAD_realz = (np.concatenate((z_[:,0,org_clist[0]], z_[:,1,org_clist[0]], z_[:,2,org_clist[0]], z_[:,3,org_clist[0]], z_[:,4,org_clist[0]], z_[:,5,org_clist[0]]), axis=1))
    WTAD_realgcorr = (np.concatenate((gencorr_[:,0,org_clist[0]], gencorr_[:,1,org_clist[0]], gencorr_[:,2,org_clist[0]], gencorr_[:,3,org_clist[0]], gencorr_[:,4,org_clist[0]], gencorr_[:,5,org_clist[0]]), axis=1))
    return WTAD_realz, WTAD_realgcorr

