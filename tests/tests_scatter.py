import sys, os 
import sys
#sys.path.append("../omniplot")
from omniplot  import plot as op
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
#test="dotplot"
#test="triangle_heatmap"
test="decomp"
test="manifold"
test="triangle_heatmap"
test="radialtree"
test="violinplot"
test="cluster"
test="regression"
test="dotplot"
test="regression"
test="nice_piechart_num"
test="pie_scatter"
test="correlation"
test="manifold"
test="stacked"
test="stackedlines"
test="correlation"
test="cluster"
test="radialtree"
test="scatterplot"

if test=="decomp":
    df=sns.load_dataset("penguins")
    df=df.dropna(axis=0)
    variables=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]

    op.decomplot(df, variables=variables,category=["species","sex"],method="pca",markers=True)
    plt.show()
elif test=="manifold":
    df=sns.load_dataset("penguins")
    df=df.dropna(axis=0)
    variables=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
    #df=df[features]
    op.manifoldplot(df, 
                    variables=variables,
                    category=["species", "island"],
                    method="tsne",markers=True)
    plt.show()
elif test=="cluster":
    df=sns.load_dataset("penguins")
    df=df.dropna(axis=0)
    features=["species","sex","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
    df=df[features]
    #op.clusterplot(df,category=["species","sex"],method="kmeans",n_clusters="auto", markers=True,topn_cluster_num=3,reduce_dimension="tsne")
    op.clusterplot(df,category=["species","sex"],method="fuzzy",n_clusters=3, piesize_scale=0.03,sp_kw={"marginal_dist":True})#,topn_cluster_num=3)
    #clusterplot(df,category=["species","sex"],method="hdbscan",eps=0.35)
    #clusterplot(df,category=["species","sex"],method="hdbscan",n_clusters="auto")
    plt.show()


elif test=="pie_scatter":
    f="/home/koh/ews/omniplot/data/energy_vs_gdp.csv"
    df=pd.read_csv(f, comment='#')
    df=df.set_index("country")
    op.pie_scatter(df, x="gdppc",y="pop", category=['biofuel_electricity',
                                                    'coal_electricity',
                                                    'gas_electricity',
                                                    'hydro_electricity',
                                                    'nuclear_electricity',
                                                    'oil_electricity',
                                                    'other_renewable_electricity',
                                                    'solar_electricity',
                                                    'wind_electricity'],logscalex=True,logscaley=True,
                                                piesizes="sum_of_each",
                                                min_piesize=0.3, label="topn_of_sum")
    plt.show()

elif test=="scatterplot":
    df=sns.load_dataset("penguins")
    df=df.dropna(axis=0)
    #features=["species","sex","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
    #df=df[features]
    print(df.shape)
    # op.scatterplot(df, x="bill_length_mm",y="bill_depth_mm", # xunit="mm", yunit="mm", 
    #                category=['species',"sex","island"],
    #                colors=["flipper_length_mm"],
    #                sizes="body_mass_g",#size_scale=1000,
    #                #show_labels={"val": "body_mass_g", "topn":5},
    #                marginal_dist=True,
    #                kmeans=True,
    #                                             )
    # fig, ax=plt.subplots()
    op.scatterplot(df,
                category=['species'],
               x="bill_length_mm",
               y="bill_depth_mm", save="/home/koh/data/omniplot/kde_kmeans")
    plt.show()