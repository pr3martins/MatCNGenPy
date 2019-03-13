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
import gensim.models.keyedvectors as word2vec

model = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin"
    , binary=True, limit=300000)
```

```python
model.most_similar(positive=('person','role'))
```

```python
model.most_similar('synopsis')
```

```python
model.most_similar(positive=('movie','casting'))
```

```python
schemaWords = ['casting','note','name','character','movie','title','person','role']
```

```python
import itertools
from pprint import pprint as pp

for i in range(1,5):
    for group in itertools.combinations(schemaWords,i):
        print ('Grupo:',group,'\n')
        pp(model.most_similar(positive=group))
        print('\n\n--------------------------------------------------\n')
```

```python
model.most_similar(positive=('casting','note','name','character','movie','title','person','role'))
```

```python
schema = ['will','sound','music','movie','best','bond','actor','thriller','movie','don','king','return','cast','Friends']

for word in schema:
    print ('Palavra:',word,'\n')
    pp(model.most_similar(word))
    print('\n\n--------------------------------------------------\n')
    
```

```python
schema = ['will','sound','music','movie','best','bond','actor','thriller','movie','don','king','return','cast','Friends']

for word in schema:
    print ('Palavra:',word,'\n')
    pp(model.most_similar(word))
    print('\n\n--------------------------------------------------\n')
```

```python
x = ['denzel', 'washington', 'clint', 'eastwood', 'john', 'wayne', 'will', 'smith', 'harrison', 'ford', 'julia', 'roberts', 'tom', 'hanks', 'johnny', 'depp', 'angelina', 'jolie', 'morgan', 'freeman', 'gone', 'wind', 'star', 'wars', 'casablanca', 'lord', 'rings', 'sound', 'music', 'wizard', 'oz', 'notebook', 'forrest', 'gump', 'princess', 'bride', 'godfather']


for word in x:
    print ('Palavra:',word,'\n')
    if word in model:
        pp(model.most_similar(word))
    else:
        print('Not in model')
    print('\n\n--------------------------------------------------\n')
```

```python
'car' in model
```
