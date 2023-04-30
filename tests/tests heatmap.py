import sys, os 
#sys.path.append("../omniplot")
from omniplot  import plot as op
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
#test="dotplot"
#test="triangle_heatmap"

test="triangle_heatmap"
test="radialtree"
test="regression"
test="correlation"
test="complex_clustermap"
test="heatmap"
test="heatmap"
if test=="correlation":
    df=sns.load_dataset("penguins")
    df=df.dropna(axis=0)
    
        
    op.correlation(df, category=["species", "island","sex"], method="pearson", ztransform=True)
    plt.show()

elif test=="dotplot":
    df=pd.DataFrame({"Experiments":["exp1","exp1","exp1","exp1","exp2","exp2","exp3"],
                        "GO":["nucleus","cytoplasm","chromosome","DNA binding","chromosome","RNA binding","RNA binding"],
                        "FDR":[10,1,5,3,1,2,0.5],
                        "odds":[3.3,1.1,2.5,2.1,0.8,2.3,0.9]})
    op.dotplot(df, row="GO",col="Experiments", size_val="FDR",color_val="odds", highlight="FDR",
    color_title="Odds", size_title="-log10 p",scaling=20)
    # df=pd.read_csv("/home/koh/ews/idr_revision/clustering_analysis/cellloc_longform.csv")
    # print(df.columns)
    # df=df.fillna(0)
    # #dotplot(df, size_val="pval",color_val="odds", highlight="FDR",color_title="Odds ratio", size_title="-log10 p value",scaling=20)
    #
    # dotplot(df, row="Condensate",col="Cluster", size_val="pval",color_val="odds", highlight="FDR",
    #         color_title="Odds", size_title="-log10 p value",scaling=20)
    plt.show()
elif test=="triangle_heatmap":
    s=20
    mat=np.arange(s*s).reshape([s,s])
    import string, random
    letters = string.ascii_letters+string.digits
    labels=[''.join(random.choice(letters) for i in range(10)) for _ in range(s)]
    df=pd.DataFrame(data=mat, index=labels, columns=labels)
    op.triangle_heatmap(df,grid_pos=[2*s//10,5*s//10,7*s//10],grid_labels=["A","B","C","D"])
elif test=="complex_clustermap":

    df=sns.load_dataset("penguins")
    
    df=df.dropna(axis=0)
    dfcol=pd.DataFrame({"features":["bill","bill","flipper"]})
    op.complex_clustermap(df,
                        dfcol=dfcol,
                        variables=["bill_length_mm","bill_depth_mm","flipper_length_mm"],
                        row_colors=["species","sex"],
                        row_scatter=["body_mass_g"],
                        row_plot=["body_mass_g"],
                        row_bar=["body_mass_g"],
                        col_colors=["features"],
                        approx_clusternum=3,
                        marginalsum=True, title="Penguins")
    plt.show()
elif test=="heatmap":
    from PIL import Image
    im=Image.open("/home/koh/Pictures/wave128.jpg").convert("LA")
    df=pd.DataFrame(np.array(im)[:,:,0][::-1])
    op.heatmap(df, shape="triangle",row_cluster=False, col_cluster=False,
    ztranform=False, row_plot={"mean": df.mean(axis=1)}, row_ticklabels=False, col_ticklabels=False)
    
    
    df=sns.load_dataset("penguins")
    
    df=df.dropna(axis=0)
    df=df.reset_index(drop=True)
    res=op.heatmap(df=df,boxlabels=True, 
    variables=["bill_length_mm","bill_depth_mm","flipper_length_mm"],
    row_split=True, clustering_method="kmeans",
               category=["species", "island"],
               col_colors={"colors": ["bill","bill","flipper"]}, 
               row_plot=["body_mass_g"], row_ticklabels=False, approx_clusternum=3)
    op.heatmap(df=df, dtype="categorical", variables=["species", "island", "sex"],
               row_plot=["body_mass_g"], row_ticklabels=False, approx_clusternum=3,column_wise_color=True) 
    # _df=df.sample(n=50, replace=False, random_state=1)
    # _df=_df.reset_index(drop=True)
    # op.heatmap(df=_df, 
    # dtype="categorical", 
    # variables=["species", "island", "sex"],
    #            row_plot=["body_mass_g"], 
    #            shape="by_category", shape_colors=np.arange(25*3).reshape([25,3]),
    #            row_ticklabels=False, 
    #            approx_clusternum=3,
    #            column_wise_color=True, sizes=np.arange(25*3).reshape([25,3])) 

    mat=np.concatenate([0.25*np.random.uniform(0,1,size=[10, 25]),0.5*np.random.uniform(0,1,size=[15, 25]),
                        np.random.uniform(0,1,size=[15, 25])])
    mat=np.concatenate([mat, np.random.uniform(0,1,size=[40, 15])], axis=1)
    print(mat.shape)
    df=pd.DataFrame(mat)
    res=op.heatmap(df, sizes=mat, size_title="random",cbar_title="random colors",
               col_split=True,
               row_plot={"mean": mat.mean(axis=1)}, 
               row_scatter={"uniform": np.random.uniform(size=mat.shape[0])}, 
               row_axis=2,
               col_bar={"range": np.arange(mat.shape[1])},
               
               edgecolor=None, approx_clusternum=3, ztranform=False)
    print(res.keys())
    ax=res["axes"]["row_0"]
    op.violinplot2(mat[res["rowsort"]], ax=ax,orientation="horizontal")
    ax.set_xlabel("violin")
    res["axes"]["row_0"].set_ylim(-0.5, mat.shape[0]-0.5)
    # # op.heatmap(df, shape="circle", sizes=mat, edgecolor=None, approx_clusternum=3, row_split=True, ztranform=False)
    # mat=np.arange(50).reshape([5,10]).astype(np.float)
    # df=pd.DataFrame(mat)
    # op.heatmap(df, shape="circle", sizes=mat, row_cluster=False,col_cluster=False, edgecolor=None, approx_clusternum=3, ztranform=False, row_split=True)
    print(res["axes"].keys())
    plt.show()
