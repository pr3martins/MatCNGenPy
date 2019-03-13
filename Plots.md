---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.2
  kernelspec:
    display_name: MatCNGenpy
    language: python
    name: matcngenpy
---

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline
```

```python
WNET = pd.read_csv('plots/WNET.csv')
WNET = WNET.set_index("Threshold")
WNET
```

```python
EmbA = pd.read_csv('plots/EmbA.csv')
EmbA = EmbA.set_index("Threshold")
EmbA
```

```python
EmbB = pd.read_csv('plots/EmbB.csv')
EmbB = EmbB.set_index("Threshold")
EmbB
```

```python
EmbC = pd.read_csv('plots/EmbC.csv')
EmbC = EmbC.set_index("Threshold")
EmbC
```

```python
EmbSim = pd.read_csv('plots/EmbSim.csv')
EmbSim = EmbSim.set_index("Threshold")
EmbSim
```

```python
WNET_EmbA = pd.read_csv('plots/WNET_EmbA.csv')
WNET_EmbA = WNET_EmbA.set_index("Threshold")
WNET_EmbA
```

```python
WNET_EmbB = pd.read_csv('plots/WNET_EmbB.csv')
WNET_EmbB = WNET_EmbB.set_index("Threshold")
WNET_EmbB
```

```python
WNET_EmbC = pd.read_csv('plots/WNET_EmbC.csv')
WNET_EmbC = WNET_EmbC.set_index("Threshold")
WNET_EmbC
```

```python
NoSchema = pd.read_csv('plots/NoSchema.csv')
NoSchema = NoSchema.set_index("Threshold")
NoSchema
```

```python
x = pd.concat([WNET['Precision'], WNET_EmbA['Precision'],WNET_EmbB['Precision'],WNET_EmbC['Precision'],EmbA['Precision'],EmbB['Precision'],EmbC['Precision'],EmbSim['Precision']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision.pdf")
```

```python
x = pd.concat([WNET['Recall'], WNET_EmbA['Recall'],WNET_EmbB['Recall'],WNET_EmbC['Recall'],EmbA['Recall'],EmbB['Recall'],EmbC['Recall'],EmbSim['Recall']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Recall')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Recall.pdf")
```

```python
x = pd.concat([WNET['F1'], WNET_EmbA['F1'],WNET_EmbB['F1'],WNET_EmbC['F1'],EmbA['F1'],EmbB['F1'],EmbC['F1'],EmbSim['F1']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('F1')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/F1.pdf")
```

```python
x = pd.concat([WNET['MaxSkippedCN'], WNET_EmbA['MaxSkippedCN'],WNET_EmbB['MaxSkippedCN'],WNET_EmbC['MaxSkippedCN'],EmbA['MaxSkippedCN'],EmbB['MaxSkippedCN'],EmbC['MaxSkippedCN'],EmbSim['MaxSkippedCN'],NoSchema['MaxSkippedCN']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('MaxSkippedCN')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/MaxSkippedCN.pdf")
```

```python
x = pd.concat([WNET['AvgSkippedCN'], WNET_EmbA['AvgSkippedCN'],WNET_EmbB['AvgSkippedCN'],WNET_EmbC['AvgSkippedCN'],EmbA['AvgSkippedCN'],EmbB['AvgSkippedCN'],EmbC['AvgSkippedCN'],EmbSim['AvgSkippedCN'],NoSchema['AvgSkippedCN']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('AvgSkippedCN')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/AvgSkippedCN.pdf")
```

```python
x = pd.concat([WNET['MRR'], WNET_EmbA['MRR'],WNET_EmbB['MRR'],WNET_EmbC['MRR'],EmbA['MRR'],EmbB['MRR'],EmbC['MRR'],EmbSim['MRR'],NoSchema['MRR']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('MRR')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/MRR.pdf")
```

```python
x = pd.concat([WNET['MRR-nonempty'], WNET_EmbA['MRR-nonempty'],WNET_EmbB['MRR-nonempty'],WNET_EmbC['MRR-nonempty'],EmbA['MRR-nonempty'],EmbB['MRR-nonempty'],EmbC['MRR-nonempty'],EmbSim['MRR-nonempty'],NoSchema['MRR-nonempty']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('MRR-nonempty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/MRR-nonempty.pdf")
```

```python
x = pd.concat([WNET['Precision1'], WNET_EmbA['Precision1'],WNET_EmbB['Precision1'],WNET_EmbC['Precision1'],EmbA['Precision1'],EmbB['Precision1'],EmbC['Precision1'],EmbSim['Precision1'],NoSchema['Precision1']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision1')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision1.pdf")
```

```python
x = pd.concat([WNET['Precision2'], WNET_EmbA['Precision2'],WNET_EmbB['Precision2'],WNET_EmbC['Precision2'],EmbA['Precision2'],EmbB['Precision2'],EmbC['Precision2'],EmbSim['Precision2'],NoSchema['Precision2']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision2')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision2.pdf")
```

```python
x = pd.concat([WNET['Precision3'], WNET_EmbA['Precision3'],WNET_EmbB['Precision3'],WNET_EmbC['Precision3'],EmbA['Precision3'],EmbB['Precision3'],EmbC['Precision3'],EmbSim['Precision3'],NoSchema['Precision3']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision3')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision3.pdf")
```

```python
x = pd.concat([WNET['Precision1-nonempty'], WNET_EmbA['Precision1-nonempty'],WNET_EmbB['Precision1-nonempty'],WNET_EmbC['Precision1-nonempty'],EmbA['Precision1-nonempty'],EmbB['Precision1-nonempty'],EmbC['Precision1-nonempty'],EmbSim['Precision1-nonempty'],NoSchema['Precision1-nonempty']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision1-nonempty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision1-nonempty.pdf")
```

```python
x = pd.concat([WNET['Precision2-nonempty'], WNET_EmbA['Precision2-nonempty'],WNET_EmbB['Precision2-nonempty'],WNET_EmbC['Precision2-nonempty'],EmbA['Precision2-nonempty'],EmbB['Precision2-nonempty'],EmbC['Precision2-nonempty'],EmbSim['Precision2-nonempty'],NoSchema['Precision2-nonempty']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision2-nonempty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision2-nonempty.pdf")
```

```python
x = pd.concat([WNET['Precision3-nonempty'], WNET_EmbA['Precision3-nonempty'],WNET_EmbB['Precision3-nonempty'],WNET_EmbC['Precision3-nonempty'],EmbA['Precision3-nonempty'],EmbB['Precision3-nonempty'],EmbC['Precision3-nonempty'],EmbSim['Precision3-nonempty'],NoSchema['Precision3-nonempty']], axis=1)
x.columns=['WNET','WNET_EmbA','WNET_EmbB','WNET_EmbC','EmbA', 'EmbB','EmbC','EmbSim','NoSchema']
plot=x.plot(legend=False)
#lgd=plot.legend(loc='center left', bbox_to_anchor=(-2, 0.5))
plot.set_ylabel('Precision3-nonempty')
fig = plot.get_figure()
fig.legends = []
fig.savefig("plots/Precision3-nonempty.pdf")
```
