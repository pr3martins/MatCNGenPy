# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: MatCNGenpy
#     language: python
#     name: matcngenpy
# ---

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %matplotlib inline

WNET = pd.read_csv('plots/WNet.csv')
WNET = WNET.set_index("Threshold")
WNET

EmbA = pd.read_csv('plots/EmbA.csv')
EmbA = EmbA.set_index("Threshold")
EmbA

EmbB = pd.read_csv('plots/EmbB.csv')
EmbB = EmbB.set_index("Threshold")
EmbB

EmbC = pd.read_csv('plots/EmbC.csv')
EmbC = EmbC.set_index("Threshold")
EmbC

EmbSim = pd.read_csv('plots/EmbSim.csv')
EmbSim = EmbSim.set_index("Threshold")
EmbSim

WNET_EmbA = pd.read_csv('plots/WNET_EmbA.csv')
WNET_EmbA = WNET_EmbA.set_index("Threshold")
WNET_EmbA

WNET_EmbB = pd.read_csv('plots/WNET_EmbB.csv')
WNET_EmbB = WNET_EmbB.set_index("Threshold")
WNET_EmbB

WNET_EmbC = pd.read_csv('plots/WNET_EmbC.csv')
WNET_EmbC = WNET_EmbC.set_index("Threshold")
WNET_EmbC

NoSchema = pd.read_csv('plots/NoSchema.csv')
NoSchema = NoSchema.set_index("Threshold")
NoSchema

x = pd.concat([WNET['Precision'],EmbSim['Precision'],EmbA['Precision'],EmbB['Precision'],WNET_EmbA['Precision'],WNET_EmbB['Precision']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=True)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('Precision')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['Recall'],EmbSim['Recall'],EmbA['Recall'],EmbB['Recall'], WNET_EmbA['Recall'],WNET_EmbB['Recall']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=True)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('Recall')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Recall.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['F1'],EmbSim['F1'],EmbA['F1'],EmbB['F1'], WNET_EmbA['F1'],WNET_EmbB['F1']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=True)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('F1')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/F1.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['MaxSkippedCN'],EmbA['MaxSkippedCN'],EmbB['MaxSkippedCN'],EmbC['MaxSkippedCN'],EmbSim['MaxSkippedCN'],WNET_EmbA['MaxSkippedCN'],WNET_EmbB['MaxSkippedCN'],WNET_EmbC['MaxSkippedCN'],], axis=1)
x.columns=['WNET','EmbA', 'EmbB','EmbC','EmbSim','WNET_EmbA','WNET_EmbB','WNET_EmbC']
#plot=x.plot(legend=True)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('MaxSkippedCN')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/MaxSkippedCN.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['AvgSkippedCN'], EmbA['AvgSkippedCN'],EmbB['AvgSkippedCN'],EmbC['AvgSkippedCN'],EmbSim['AvgSkippedCN'],WNET_EmbA['AvgSkippedCN'],WNET_EmbB['AvgSkippedCN'],WNET_EmbC['AvgSkippedCN'],NoSchema['AvgSkippedCN']], axis=1)
x.columns=['WNET','EmbA', 'EmbB','EmbC','EmbSim','WNET_EmbA','WNET_EmbB','WNET_EmbC','NoSchema']
#plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('AvgSkippedCN')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/AvgSkippedCN.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['MRR'],EmbSim['MRR'], EmbA['MRR'],EmbB['MRR'],WNET_EmbA['MRR'],WNET_EmbB['MRR']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=True)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('MRR')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/MRR.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['MRR-nonempty'],EmbSim['MRR-nonempty'], EmbA['MRR-nonempty'],EmbB['MRR-nonempty'],WNET_EmbA['MRR-nonempty'],WNET_EmbB['MRR-nonempty']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('MRR Non-Empty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/MRR-nonempty.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['Precision1'], EmbSim['Precision1'],EmbA['Precision1'],EmbB['Precision1'],WNET_EmbA['Precision1'],WNET_EmbB['Precision1']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('P@1')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision1.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['Precision2'], EmbSim['Precision2'],EmbA['Precision2'],EmbB['Precision2'],WNET_EmbA['Precision2'],WNET_EmbB['Precision2']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('P@2')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision2.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['Precision3'],EmbSim['Precision3'],EmbA['Precision3'],EmbB['Precision3'], WNET_EmbA['Precision3'],WNET_EmbB['Precision3']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('P@3')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision3.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

# +
x = pd.concat([WNET['Precision1-nonempty'], EmbSim['Precision1-nonempty'],EmbA['Precision1-nonempty'],EmbB['Precision1-nonempty'],WNET_EmbA['Precision1-nonempty'],WNET_EmbB['Precision1-nonempty']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']

plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('P@1 Non-Empty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision1-nonempty.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')
# -

x = pd.concat([WNET['Precision2-nonempty'],EmbSim['Precision2-nonempty'],EmbA['Precision2-nonempty'],EmbB['Precision2-nonempty'], WNET_EmbA['Precision2-nonempty'],WNET_EmbB['Precision2-nonempty']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('P@2 Non-Empty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision2-nonempty.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

x = pd.concat([WNET['Precision3-nonempty'], EmbSim['Precision3-nonempty'],EmbA['Precision3-nonempty'],EmbB['Precision3-nonempty'],WNET_EmbA['Precision3-nonempty'],WNET_EmbB['Precision3-nonempty']], axis=1)
x.columns=['WNET','EmbSim','Emb10', 'Emb10N','WNET+Emb10','WNET+Emb10N']
plot=x.plot(legend=False)
lgd=plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_ylabel('P@3 Non-Empty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision3-nonempty.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

