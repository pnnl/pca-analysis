"""The main script for performing PCA-SIMS analysis"""

from typing import List

import os
import sys
import traceback

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import patches
import warnings
import re

# Plotting settings
params={
    'font.family':'serif',
    'font.serif':'Times New Roman',
    # 'font.style':'italic',
    # 'font.weight':'bold', 
    'axes.labelsize': '28',
    'xtick.labelsize':'15',
    'ytick.labelsize':'15',
    'lines.linewidth':3,
    'lines.markersize':10,
    'legend.fontsize': '25',
    # 'figure.figsize': '12, 9',
    }
dpi = 300
pylab.rcParams.update(params)


def plot_pca_result(
    pca: PCA,
    pca_data: np.ndarray,
    samplelist: List,
    mass: List,
    sample_description_set: List,
    pcaDir: str,
    outDir: str,
    f_group_numbers: str='sims-data/OriginalData/Group Numbers.txt',
    max_pcacomp: int=5
):
    """
    The main plotting functions.
    """
    # SCREE PLOT
    per_var=np.round(pca.explained_variance_ratio_*100,decimals=2)[:max_pcacomp]
    labels=['PC'+str(x) for x in range(1, max_pcacomp+1)]

    per_varEx=np.round(pca.explained_variance_ratio_*100,decimals=2)[:10]
    labelsEx=['PC'+str(x) for x in range(1, 11)]
    fig_screeplot = os.path.join(pcaDir, outDir, 'Scree Plot.png')
    try:
        scree=pd.DataFrame(per_varEx)
        # scree.to_csv(pcaDir+'output/Scree_PC1-10.txt')
        scree.to_csv(os.path.join(pcaDir,outDir,'Scree_PC1-10.txt'))

        plt.bar(x=range(1,11),height=per_varEx,tick_label=labelsEx)
        plt.tick_params(labelsize=15)
        for a,b in zip(np.arange(11)+1,np.array(per_varEx)):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', va= 'bottom',fontsize=12)
        plt.ylabel('Percentage of Explained Variance', fontsize=15, fontname = 'Times New Roman', fontweight = 'bold')
        plt.xlabel('Principal Component',fontsize=15, fontname = 'Times New Roman', fontweight = 'bold')
        plt.title('Scree Plot')
        plt.tight_layout()
        plt.savefig(fig_screeplot,dpi=dpi)
        # plt.show()
        plt.close()

    except:
        print(traceback.print_exc())
        print('***Error! Cannot Drawing Correct Scree Plot to Exports File!***')
        sys.exit()

    # EXTRACT PC1-PCmax_pcacomp
    pca_data=pca_data[:,:max_pcacomp]
    pca_df=pd.DataFrame(pca_data,index=samplelist,columns=labels)

    # FETCH GROUP LABELS
    print('-------->Score Plot and Confidence Ellipse Drawing...')

    # TODO What if the user has added character(s) at the start (like the S here: S479-P2) that make x not an integer?
    '''
    This helper function takes the group names, which we expect to be of the following form:
             <A group of digits> - <P OR N> <A number from 1-6>
    for instance, 072-P6 is a typical group name. Here, we extract the digits for later use by splitting on the -.
    '''
    def ExtractString(x):
        x = str(x)
        x = re.sub("-", "_", x)
        x = x.split("_")[0]
        return int(x)

    try:
        pca_df['group']=pca_df.index
        pca_df['group']=pca_df['group'].apply(ExtractString)
        pca_df=pca_df.sort_values(by=['group'])
        # print(pca_df)
        pca_df.to_csv(os.path.join(pcaDir, outDir, 'PC1-{}_SCORE_TABLE.txt'.format(max_pcacomp)))
    except:
        print(traceback.print_exc())
        print('***Error! Missing Group Information!***')
        sys.exit()


    # DRAW SCORE PLOTS WITH CONFIDENCE ELLIPSE
    fig_scores_set, fig_scores_confid_set, fig_scores_single_set = [], [], []
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]


    # READ SUBSET OF GROUP NUMBERS ON WHICH WE WANT TO PERFORM PCA
    all_group_nums = pca_df['group'].unique()
    legend_labels = []
    N = 0
    try:
        sub_group_nums = pd.read_csv(f_group_numbers)
        sub_group_nums = sub_group_nums['Group'].unique().tolist()
        N = len(sub_group_nums)

        # Get the indices of the entries in df for which the user asks us to pull data (as specified in group numbers .txt file)
        sub_group_indices = []
        for i in range(len(sub_group_nums)):
            sub_group_indices.append(np.argwhere(all_group_nums == sub_group_nums[i])[0][0])

        # Pull legend labels from the sample descriptions (but only the subset specified the user selected)
        legend_labels = [sample_description_set[i][1] for i in sub_group_indices]
    except:
        print(traceback.print_exc())
        print('***Error! Group names missing or formatted incorrectly!***')
        sys.exit()


    try:
        nstd = 1.645
        figroup=list()
        markern = [ ',','o','H','^','v','x','1','D','X','2','3','4','8','s','p','.',
                    'h','+','d','|','_',0,1,2,3,4,5,6,7,8,9,10,11]
        colorn = ['purple','green','blue','brown','red',
                    'teal','orange','magenta','pink',
                    'gray','violet','turquoise','yellow',
                    'lavender','tan','cyan','aqua','yellowgreen','chocolate',
                    'coral','fuchsia','goldenrod','indigo',
                    'grey','darkorange','rosybrown','palegreen','deepskyblue']
        fign=1

        for j in range(1,max_pcacomp+1):
            fig_scores_set_j = []
            for k in range(j+1,max_pcacomp+1):
                if j != k:
                    plt.figure(figsize=(10,7))
                    ax = plt.subplot(111)
                    fign+=1

                    # i goes from 0,1,...n-1 and label goes from n,n+1,...m. This acounts for the cases where the metadata 
                    # doesn't start at 1.
                    for i,label in enumerate(sub_group_nums):
                        x=pca_df[pca_df['group']==label]['PC'+str(j)].values
                        y=pca_df[pca_df['group']==label]['PC'+str(k)].values
                        figroup.append(plt.scatter(x, y,color=colorn[i],marker=markern[i]))
                        

                    plt.xlabel('PC'+str(j)+' Scores'+' ('+str(per_varEx[j-1])+'%)'\
                        ,fontsize=28, fontname = 'Times New Roman', fontweight = 'bold')
                    plt.ylabel('PC'+str(k)+' Scores'+' ('+str(per_varEx[k-1])+'%)'\
                        ,fontsize=28, fontname = 'Times New Roman', fontweight = 'bold')
                    plt.legend(figroup,legend_labels,loc='center left', bbox_to_anchor=(1, 0.5))
                    # plt.tight_layout()
                    fig_scores = os.path.join(pcaDir, outDir, 'Origin_PC'+str(j)+'PC'+str(k)+'.png')
                    plt.savefig(fig_scores, bbox_inches='tight',dpi=dpi)
                    #plt.show()
                    plt.close()
                    fig_scores_set_j.append(fig_scores)
            fig_scores_set.append(fig_scores_set_j)

        for j in range(1,max_pcacomp+1):
            fig_scores_confid_set_j = []
            for k in range(j+1,max_pcacomp+1):
                if j != k:
                    plt.figure(figsize=(10,7))
                    ax = plt.subplot(111)
                    fign+=1
                    for i,label in enumerate(sub_group_nums):
                        x=pca_df[pca_df['group']==label]['PC'+str(j)].values
                        y=pca_df[pca_df['group']==label]['PC'+str(k)].values
                        cov = np.cov(x, y)
                        vals, vecs = eigsorted(cov)
                        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                        w, h = 2 * nstd * np.sqrt(np.abs(vals))
                        ell = patches.Ellipse(xy=(np.mean(x), np.mean(y)),
                                    width=w, height=h,
                                    angle=theta, edgecolor=colorn[i]\
                                        , linestyle='-', linewidth=2.0,
                                        facecolor=colorn[i], alpha=0.2)
                        # ell.set_facecolor('none')
                        ax.add_artist(ell)
                        figroup.append(plt.scatter(x, y,color=colorn[i],marker=markern[i]))

                    plt.xlabel('PC'+str(j)+' Scores'+' ('+str(per_varEx[j-1])+'%)'\
                        ,fontsize=28, fontname = 'Times New Roman', fontweight = 'bold')
                    plt.ylabel('PC'+str(k)+' Scores'+' ('+str(per_varEx[k-1])+'%)'\
                        ,fontsize=28, fontname = 'Times New Roman', fontweight = 'bold')
                    plt.legend(figroup,legend_labels,loc='center left', bbox_to_anchor=(1, 0.5))
                    
                    # plt.tight_layout()
                    # fig_scores_confid = pcaDir+'output/'+'Ellipse_PC'+str(j)+'PC'+str(k)+'.png'
                    fig_scores_confid = os.path.join(pcaDir, outDir, 'Ellipse_PC'+str(j)+'PC'+str(k)+'.png')
                    plt.savefig(fig_scores_confid, bbox_inches='tight',dpi=dpi)
                    #plt.show()
                    plt.close()
                    fig_scores_confid_set_j.append(fig_scores_confid)
            fig_scores_confid_set.append(fig_scores_confid_set_j)
                    
    
        ## Add new plot here!!! 2020-11-6
        for pc in range(1,max_pcacomp+1):
            plt.figure(figsize=(14,7))
            ax = plt.subplot(111)
            group_nums = len(sub_group_nums)
            plt.xlim(left=0, right=10*(group_nums+1))
            plt.xticks([])

            for i,label in enumerate(sub_group_nums):
                heights = pca_df[pca_df['group']==label]['PC'+str(pc)].values
                pt_nums = len(heights)
                
                if pt_nums == 1:
                    x_positions = [10*(i+1)]
                else:
                    x_positions = [10*(i+1)-7/2+x*7/(pt_nums-1) for x in range(pt_nums)]
                figroup.append(plt.scatter(x_positions,heights, color=colorn[i],marker=markern[i]))

            bottom, top = plt.ylim()
            
            plt.xlabel('Experiments',fontsize=28, fontname = 'Times New Roman', fontweight = 'bold')
            plt.ylabel('PC'+str(pc)+' Scores'+' ('+str(per_varEx[pc-1])+'%)'\
                ,fontsize=28, fontname = 'Times New Roman', fontweight = 'bold')
            plt.legend(figroup,legend_labels,loc='center left', bbox_to_anchor=(1, 0.5))
            
            fig_score_single = os.path.join(pcaDir, outDir, 'PC_scores_'+'PC'+str(pc)+'.png')
            plt.savefig(fig_score_single, bbox_inches='tight',dpi=dpi)
            plt.close()
            fig_scores_single_set.append(fig_score_single)

                    
    except:
        print(traceback.print_exc())
        print('***Error! Cannot Draw Correct Score Plot and Confidence Ellipse')
        # import trace
        # print(traceback.print_exc())
        sys.exit()


    # DRAW LOADING BAR PLOTS
    print('-------->Loading Plot Drawing and Peak Labeling...')

    try:

        fig_loading_set = []

        loading_scores=pca.components_[:max_pcacomp,:]
        loadingTable=pd.DataFrame(loading_scores.T,index=mass,columns=[1,2,3,4,5])
        fetchn=5
        fetchn_more=20


        for i in range(0,max_pcacomp):
            loadingTable=loadingTable.sort_values(by=[i+1])
            
            negative_tenx=loadingTable.iloc[0:fetchn_more][i+1].index.tolist()
            negative_teny=loadingTable.iloc[0:fetchn_more][i+1].tolist()
            positive_tenx=loadingTable.iloc[-fetchn_more:][i+1].index.tolist()[::-1]
            positive_teny=loadingTable.iloc[-fetchn_more:][i+1].tolist()[::-1]


            plt.figure(figsize=(14,7))
            plt.bar(mass,loading_scores[i,:],color='blue')
            for j in range(fetchn):
                plt.text(x=negative_tenx[j],y=negative_teny[j],s =negative_tenx[j],fontsize=15)
                plt.text(x=positive_tenx[j],y=positive_teny[j],s =positive_tenx[j],fontsize=15)
            plt.xlim(left=-20)
            plt.xlabel('m/z (amu)',fontsize=25, fontname = 'Times New Roman', fontweight = 'bold')
            plt.ylabel('PC'+str(i+1)+' Loadings '+'('+str(per_varEx[i])+'%)'\
                ,fontsize=25, fontname = 'Times New Roman', fontweight = 'bold')
            # plt.tight_layout()
        
            fig_loading = os.path.join(pcaDir, outDir, 'Loading'+'PC'+str(i+1)+'.png')
            plt.savefig(fig_loading,dpi=dpi)
            # plt.show()
            plt.close()
            fig_loading_set.append(fig_loading)


            #Exports LOADING BAR TABLE
            
            LoadingPositive_Excel=pd.DataFrame(data={"+ loading":[" "]*fetchn_more,"No. #":[x for x in range(1,fetchn_more+1)],
            "UnitMass":positive_tenx,"Accurate Mass":[" "]*fetchn_more,"Peak assignment":[" "]*fetchn_more})

            med=pd.DataFrame(data={"+ loading":["- loading"],"No. #":["No. #"],"UnitMass":["UnitMass"],"Accurate Mass":["Accurate Mass"],"Peak assignment":["Peak assignment"]})


            LoadingNegative_Excel=pd.DataFrame(data={"+ loading":[" "]*fetchn_more,"No. #":[x for x in range(1,fetchn_more+1)],
            "UnitMass":negative_tenx,"Accurate Mass":[" "]*fetchn_more,"Peak assignment":[" "]*fetchn_more})

            Loading_Excel=pd.concat([LoadingPositive_Excel,med,LoadingNegative_Excel])

            Loading_Excel.to_excel(os.path.join(pcaDir, outDir, 'Loading'+'PC'+str(i+1)+'.xlsx'),index=False)

    except:
        print(traceback.print_exc())
        print('Error! Cannot Draw Correct Loading Plot')
        sys.exit()
        
    # EXPORTS LOADING DATA
    try:
        loadingTable=pd.DataFrame(loading_scores.T,index=mass,columns=['PC'+str(i) for i in range(1,max_pcacomp+1)])
        loadingTable.to_csv(os.path.join(pcaDir, outDir, 'PC1-{}_loadingTable.txt'.format(max_pcacomp)))
    except:
        print(traceback.print_exc())
        print('***Error! Cannot Export Loading Data Correctly!***')
        sys.exit()
    
    return pca_df, fig_screeplot, fig_scores_set, fig_scores_confid_set, fig_scores_single_set, fig_loading_set 