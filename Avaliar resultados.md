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
x=[('s', 'character', '*', frozenset({'character'})), ('s', 'casting', 'note', frozenset({'script'})), ('s', 'person', '*', frozenset({'actress'})), ('s', 'character', '*', frozenset({'char'}))]

y=[(table+'.'+attr,tuple(keys)[0]) for (s,table,attr,keys) in x]
for line in y:
    print(line,sep='')
```

```python
tuple(frozenset({'character'}))[0]
```

```python
from pprint import pprint as pp

goldenStandards = [
    #William Smith nickname
    ('s', 'person', 'name', frozenset({'nickname'})),
    ('s', 'character', 'name', frozenset({'nickname'})),
    #protagonist sound music
    ('s', 'character', '*', frozenset({'protagonist'})),
    #character Forrest Gump
    ('s', 'character', '*', frozenset({'character'})),
    #script of Casablanca
    ('s', 'casting', 'note', frozenset({'script'})),
    #best movie award James Cameron
    ('s', 'movie', '*', frozenset({'movie'})),
    #actor James Bond
    ('s', 'person', '*', frozenset({'actor'})),
    #flick Ellen Page thriller
    ('s', 'movie', '*', frozenset({'flick'})),
    #movie Terry Gilliam Benicio del Toro Dr gonzo
    ('s', 'movie', '*', frozenset({'movie'})),
    #director artificial intelligent Haley Joel Osment
    #Trivia Don Quixote
    #Movie Steven Spielberg
    ('s', 'movie', '*', frozenset({'movie'})),
    #German fellow actor Mel Gibson
    ('s', 'person', '*', frozenset({'actor'})),
    #Fellowship Ring King Towers
    #Lord of the Rings films
    ('s', 'movie', '*', frozenset({'films'})),
    #Director John Hughes Matthew Broderick 1986
    #cast Friends
    ('s', 'casting', '*', frozenset({'cast'})),
    #Henry Fonda mine
    #name of actress in Lara Croft film
    ('s', 'character', 'name', frozenset({'name'})),
    ('s', 'person', 'name', frozenset({'name'})),
    ('s', 'person', '*', frozenset({'actress'})),
    #Russell Crowe gladiator char name
    ('s', 'character', '*', frozenset({'char'})),
    ('s', 'character', 'name', frozenset({'name'})),
    ('s', 'person', 'name', frozenset({'name'})),
    #Darth Vader
    #Norman Bates
    #Atticus surname
    ('s', 'character', 'name', frozenset({'surname'})),
    #social network
    #Space Odyssey Adventure year
    #Chihiro animation
    #actor Draco Harry Potter
    ('s', 'person', '*', frozenset({'actor'})),
]

TP=TN=FP=FN=0

with open('schemaMappings/threshold-0.8/wdnet_embeddingsA.txt') as f:
    for line in f.readlines():
        if eval(line) in goldenStandards:
            #print('nice', line)
            TP+=1
            goldenStandards.remove(eval(line))
        else:
            #print('bad ', line)
            FP+=1
    FN=len(goldenStandards)
        
    print('TP:',TP,'\nFP:',FP,'\nFN',FN)
    
    print(goldenStandards)
```

```python



```
