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

# Similaridades WordNet

* As métricas atuais testadas olham apenas a hierarquia do wordnet
    * Verificar se tem métricas que olham a definição (Information Content, definition)
        * Resnik, [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus)
        * [A Review of Semantic Similarity Measures in WordNet](http://www.cartagena99.com/recursos/alumnos/ejercicios/Article%201.pdf)
    * as vezes os termos similares estão em **derivationally related form**
        * Exemplo: [Actor](http://wordnetweb.princeton.edu/perl/webwn?o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&r=1&s=actor&i=3&h=10001000000#c), no caso o LCS de actor e casting é bem baixo, na verdade até mesmo actor e act
* Encontrar a clique de similaridade máxima. 
    * Apesar de casting ter vários sentidos, o melhor para ser usado é: the choice of **actors** to play particular **roles** in a play or **movie**
    * Escolher um elemento (significado) de cada partição (palavra)

```python
from pprint import pprint as pp
import itertools
import nltk 
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
```

```python
schemaWords = ['casting','note','name','char','movie','title','person','role']
```

```python
for x in wn.synsets('casting'):
    print(x,x.definition())
```

```python
wn.synset('cast.v.05')
```

```python
def showSimilarities(wordA,wordB):
    A = set(wn.synsets(wordA, pos=wn.NOUN))
    B = set(wn.synsets(wordB, pos=wn.NOUN))

    x = []
    for (sense1,sense2) in itertools.product(A,B):  
        #sim = wn.wup_similarity(sense1,sense2) or 0
        
        sim = sense1.wup_similarity(sense2)
        
        x.append( (sense1.definition(),sense2.definition(),sim) )
    x = sorted(x,key=lambda x: x[2],reverse=True)
    
    for e in x:
        print(e[0],'\n',e[1],'\n',e[2],'\n\n')
```

```python
showSimilarities('casting','actor')
```

```python
import nltk
nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
```

```python
nltk.download('genesis')
from nltk.corpus import genesis
genesis_ic = wn.ic(genesis, False, 0.0)
```
