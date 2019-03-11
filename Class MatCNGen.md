---
jupyter:
  jupytext:
    formats: ipynb,md
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
from pprint import pprint as pp
import gc #garbage collector usado no createinvertedindex
```

```python
import gensim.models.keyedvectors as word2vec
from gensim.models import KeyedVectors

def loadWordEmbeddingsModel(filename = "word_embeddings/word2vec/GoogleNews-vectors-negative300.bin"):
    model = KeyedVectors.load_word2vec_format(filename,
                                                       binary=True, limit=500000)
    return model

model = loadWordEmbeddingsModel()
```

```python
class BabelItemsIter:
    def __init__(self,babelhash):
        __slots__ = ('__babelhash')
        self.__babelhash = babelhash    
        
    def __len__(self):
        return len(self.__babelhash)
    
    def __contains__(self,item):
        (key,value) = item
        return key in self.__babelhash and self.__babelhash[key]==value
        
    def __iter__(self):
        for key in self.__babelhash.keys():
            yield key, self.__babelhash[key]
            
    #Apesar de que segundo o PEP 3106 (https://www.python.org/dev/peps/pep-3106/) recomenda que façamos
    # outros métodos, como and,eq,ne para permitir que a saída seja um set,
    # não estamos preocupados com isso aqui.
```

```python
class BabelHash(dict):
    
    def __init__(self,babel={}):
        __slots__ = ('__babel')
        dict.__init__(self)
        self.__babel = babel
        
    def __getidfromkey__(self,key):
        return self.__babel[key]
    
    def __getkeyfromid__(self,key_id):
        key = self.__babel[key_id]
        return key
    
    def __getitem__(self,key):
        key_id = self.__getidfromkey__(key)
        return dict.__getitem__(self,key_id)
    
    def __setitem__(self,key,value):    
        try:
            key_id = self.__babel[key]
        except KeyError:
            key_id = len(self.__babel)+1
                     
            self.__babel[key] = key_id
            self.__babel[key_id] = key
        
        dict.__setitem__(self, key_id,value)
    
    def __delitem__(self, key):
        key_id = self.__getidfromkey__(key)
        dict.__delitem__(self, key_id)
        
    def __missing__(self,key):
        key_id = self.__getidfromkey__(key)
        return key_id
        
    def __delitem__(self, key):
        key_id = self.__getidfromkey__(key)
        dict.__delitem__(self,key_id)
    
    def __contains__(self, key):
        try:
            key_id = self.__getidfromkey__(key)
        except KeyError:
            return False
        
        return dict.__contains__(self,key_id)    
    
    def __iter__(self):
        for key_id in dict.keys(self):
            yield self.__getkeyfromid__(key_id)
    
    def keys(self):
        for key_id in dict.keys(self):
            yield self.__getkeyfromid__(key_id)
    
    def items(self):
        return BabelItemsIter(self)
    
    def get(self,key):
        value = None
        if key in self:
            value = self.__getitem__(key)
        return value
    
    def setdefault(self,key,default=None):
        if key not in self:
            self[key]=default
        return self[key]
    
    def printBabel(self):
        print(self.__babel)
```

```python
class WordHash(dict):      
        
    def __init__(self): 
        dict.__init__(self)
    
    def addMapping(self,word,table,attribute,ctid):
        self.setdefault( word, (0, BabelHash() ) )                    
        self[word].setdefault(table , BabelHash() )       
        self[word][table].setdefault( attribute , [] ).append(ctid)        
        
    def getMappings(self,word,table,attribute):
        return self[word][table][attribute]
    
    def getIAF(self,key):
        return dict.__getitem__(self,key)[0]
    
    def setIAF(self,key,IAF):

        oldIAF,oldValue = dict.__getitem__(self,key)
        
        dict.__setitem__(self, key,  (IAF,oldValue)  )
    
    def __getitem__(self,word):
        return dict.__getitem__(self,word)[1]
    
    def __setitem__(self,word,value): 
        oldIAF,oldValue = dict.__getitem__(self,word)
        dict.__setitem__(self, word,  (oldIAF,value)  )
```

```python
import psycopg2
from psycopg2 import sql
import string

import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')

stw_set = set(stopwords.words('english')) - {'will'}

class DatabaseIter:
    def __init__(self,embeddingModel,dbname='dblp',user='imdb',password='imdb'):
        self.dbname=dbname
        self.user=user
        self.password =password
        self.embeddingModel=embeddingModel

    def __iter__(self):
        with psycopg2.connect(dbname=self.dbname,user=self.user,password=self.password) as conn:
            with conn.cursor() as cur:

                # Get list of tablenames

                GET_TABLE_NAMES_SQL='''
                    SELECT DISTINCT table_name
                    FROM information_schema.columns 
                    WHERE table_schema='public';
                ''' 
                cur.execute(GET_TABLE_NAMES_SQL)

                tables = cur.fetchall()
                print(tables)
                for table in tables:
                    table_name = table[0]

                    if table_name not in self.embeddingModel:
                        print('TABLE ',table_name, 'SKIPPED')
                        continue

                    print('INDEXING TABLE ',table_name)

                    #Get all tuples for this tablename
                    cur.execute(
                        sql.SQL("SELECT ctid, * FROM {};").format(sql.Identifier(table_name))
                        #NOTE: sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
                    )

                    printSkippedColumns = True

                    for row in cur.fetchall(): 
                        for column in range(1,len(row)):
                            column_name = cur.description[column][0] 

                            if column_name not in self.embeddingModel or column_name=='id':
                                if printSkippedColumns:
                                    print('\tCOLUMN ',column_name,' SKIPPED')
                                continue

                            ctid = row[0]

                            for word in [word.strip(string.punctuation) for word in str(row[column]).lower().split()]:

                                #Ignoring STOPWORDS
                                if word in stw_set:
                                    continue

                                yield table_name,ctid,column_name, word

                        printSkippedColumns=False
```

```python
import psycopg2
from psycopg2 import sql
import string

import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')

stw_set = set(stopwords.words('english')) - {'will'}

def createInvertedIndex(embeddingModel,dbname='dblp',user='imdb',password='imdb',showLog=True):
    #Output: wordHash (Term Index) with this structure below
    #map['word'] = [ 'table': ( {column} , ['ctid'] ) ]

    '''
    The Term Index is built in a preprocessing step that scans only
    once all the relations over which the queries will be issued.
    '''

    
    wh = WordHash()
    ah = {}
    
    previousTable = None
    
    for table,ctid,column,word in DatabaseIter(model):        
        wh.addMapping(word,table,column,ctid)
        
        ah.setdefault(table,{}).setdefault(column,{}).setdefault(word,1)
        ah[table][column][word]+=1
        
    for table in ah:
        for column in ah[table]:
            
            maxFrequency = numDistinctWords = numWords = 0            
            for word, frequency in ah[table][column].items():
                
                numDistinctWords += 1
                
                numWords += frequency
                
                if frequency > maxFrequency:
                    maxFrequency = frequency
            
            norm = 0
            ah[table][column] = (norm,numDistinctWords,numWords,maxFrequency)

    print ('INVERTED INDEX CREATED')
    gc.collect()
    return wh,ah
```

```python
wh,ah = createInvertedIndex(model)
```

```python
pp(ah)
```

```python
from math import log1p 

def processIAF(wordHash,attributeHash):
    
    total_attributes = sum([len(attribute) for attribute in attributeHash.values()])
    
    for (term, values) in wordHash.items():
        attributes_with_this_term = sum([len(attribute) for attribute in wordHash[term].values()])
        IAF = log1p(total_attributes/attributes_with_this_term)
        wordHash.setIAF(term,IAF)        
        
    print('IAF PROCESSED')
```

```python
processIAF(wh,ah)
```

```python
def processNormsOfAttributes(wordHash,attributeHash):    
    for word in wh:
        for table in wh[word]:
            for column, ctids in wh[word][table].items():
                   
                (prevNorm,numDistinctWords,numWords,maxFrequency) = attributeHash[table][column]

                IAF = wordHash.getIAF(word)

                frequency = len(ctids)
                
                TF = frequency/maxFrequency
                
                Norm = prevNorm + (TF*IAF)

                attributeHash[table][column]=(Norm,numDistinctWords,numWords,maxFrequency)
                
    print ('NORMS OF ATTRIBUTES PROCESSED')
```

```python
processNormsOfAttributes(wh,ah)
```

## Class Tupleset

```python
import copy
class Tupleset:
   
    def __init__(self, table, predicates = None, tuples = None):            
        
        self.table = table
        self.predicates= predicates if predicates is not None else {}
        self.tuples= tuples if tuples is not None else set()
        
    def addTuple(self, tuple_id):
        self.tuples.add(tuple_id)
        
    def addTuples(self, tuple_ids):
        self.tuples.update(tuple_ids)
        
    def addAttribute(self,attribute):
        self.attributes[attribute].setdefault( (set(),set()) )
    
    def union(self, otherTupleset, changeSources = False, projectionOnly = False):
              
        if self.table != otherTupleset.table:
            return None
        
        if self.table == None:
            return None
        
        if len(self.getKeywords() & otherTupleset.getKeywords())>0:
            #tuple sets com palavras repetidas
            return None

        if projectionOnly:
            if self.isValueFreeTupleset()==False or otherTupleset.isValueFreeTupleset() == False:
                return None
                
        
        jointTuples = self.tuples & otherTupleset.tuples
        
        jointPredicates = {}
        
        jointPredicates.update(copy.deepcopy(self.predicates))
        
        for attribute, (schemaWords, valueWords) in otherTupleset.predicates.items():  
            jointPredicates.setdefault(attribute,   (set(),set())    ) 
            jointPredicates[attribute][0].update(schemaWords)
            jointPredicates[attribute][1].update(valueWords)
            
        jointTupleset = Tupleset(self.table, jointPredicates , jointTuples)
        
        if changeSources:
            self.tuples.difference_update(jointTuples)
            otherTupleset.tuples.difference_update(jointTuples)
        
        return jointTupleset    
        
    def addValueMapping(self,valueWord,attribute='*'):
        self.predicates.setdefault(attribute,   (set(),set())    ) 
        self.predicates[attribute][1].add(valueWord)
        
    
    def addSchemaMapping(self,schemaWord,attribute='*'):
        self.predicates.setdefault(attribute,   (set(),set())    ) 
        self.predicates[attribute][0].add(schemaWord)

    
    def getMappings(self):
        return [(self.table,attribute,schemaWords,valueWords) 
                for attribute, (schemaWords,valueWords) in self.predicates.items()]
    
    
    def getValueMappings(self):
        return [(self.table,attribute,valueWords) 
                for attribute, (schemaWords, valueWords ) in self.predicates.items() 
                if valueWords != set()]
                
    def getSchemaMappings(self): 
        return [(self.table,attribute,schemaWords) 
                for attribute, (schemaWords, valueWords ) in self.predicates.items() 
                if schemaWords != set()]
            
    def getAttributes(self):
        return [attr for attr in self.predicates.keys()]
                
    def getKeywords(self):
        keywords = set()
        for attribute in self.predicates.keys():
            
            schemaWords,valueWords = self.predicates[attribute]
            
            keywords.update(schemaWords)                      
            keywords.update(valueWords)
        return frozenset(keywords)
        
    def isFreeTupleset(self):
        return len(self.predicates)==0
    
    def isValueFreeTupleset(self):
        for schemaWords,valueWords in self.predicates.values():
            if len(valueWords)>0:
                return False
        return True
    
    def isSchemaFreeTupleset(self):
        for schemaWords,valueWords in self.predicates.values():
            if len(schemaWords)>0:
                return False
        return True
        
    def hasTuples(self):
        return len(self.tuples)>0
    
    def clearTuples(self):
        self.tuples.clear()
    
    def __repr__(self):
        return self.__str__()
    

    
    def __str__(self):
        result = self.table.upper()
        str_predicates = []
        
        for attribute in self.predicates.keys():
            schemaWords , valueWords = self.predicates[attribute]
            
            if schemaWords == set():
                schemaWords = {}
                
            if valueWords == set():
                valueWords = {}
            
            
            str_predicates.append (attribute + str(schemaWords) + str(valueWords))
            
        result += "(" + ','.join(str_predicates) + ")"
        return result        
    
    def __eq__(self, other):
        return isinstance(other, Tupleset) and self.table == other.table and self.predicates == other.predicates and self.tuples == other.tuples  
    
    def __hash__(self):
        return hash(self.__repr__())
    
x = Tupleset('paper')
x.addValueMapping('discover','title')
x.addTuple(1)
x.addTuple(2)

y = Tupleset('paper')
y.addSchemaMapping('2002','title')
y.addTuple(1)
y.addTuple(3)

w = x.union(y,changeSources = True)
pp([x,y,w])
```

```python
x = 'PERSON'
y = {'attrA':(set(),set()),'attrB':({1,2},set())}
p = [(x,attribute,schemaWords) for attribute, (schemaWords, _ ) in y.items() if schemaWords != set()]
```

```python
import itertools
def TSFindClass(Q,wordHash):
    #Input:  A keyword query Q=[k1, k2, . . . , km]
    #Output: Set of non-free and non-empty tuple-sets Rq

    '''
    The tuple-set Rki contains the tuples of Ri that contain all
    terms of K and no other keywords from Q
    '''
    
    #Part 1: Find sets of tuples containing each keyword
    P = set()
    for keyword in Q:
        
        if keyword not in wordHash:
            continue
        
        for table in wordHash[keyword]:
            for (attribute,ctids) in wordHash[keyword][table].items():
                
                ts = Tupleset(table)
                ts.addValueMapping(keyword,attribute)
                ts.addTuples(ctids)                
                P.add(ts)
    
    #Part 2: Find sets of tuples containing larger termsets
    TSInterMartins(P)
    
    
    #Part 3: Clean tuples
    for ts in P:
        ts.clearTuples()
    
    
    return P

# def TSInter(P):
#     #Input: A Set of non-empty tuple-sets for each keyword alone P 
#     #Output: The Set P, but now including larger termsets (process Intersections)

    
    
#     '''
#     Termset is any non-empty subset K of the terms of a query Q        
#     '''
    
#     Pprev = {}
#     Pprev=copy.deepcopy(P)
#     Pcurr = {}

#     combinations = [x for x in itertools.combinations(Pprev.keys(),2)]
#     for ( Ki , Kj ) in combinations:
#         Tki = Pprev[Ki]
#         Tkj = Pprev[Kj]
        
#         X = Ki | Kj
#         Tx = Tki & Tkj        
        
#         if len(Tx) > 0:            
#             Pcurr[X]  = Tx            
#             Pprev[Ki] = Tki - Tx         
#             Pprev[Kj] = Tkj - Tx
            
#     if Pcurr != {}:
#         Pcurr = copy.deepcopy(TSInter(Pcurr))
        
#     #Pprev = Pprev U Pcurr
#     Pprev.update(Pcurr)     
#     return Pprev   


def TSInterMartins(P):
    #Input: A Set of non-empty tuple-sets for each keyword alone P 
    #Output: The Set P, but now including larger termsets (process Intersections)

    '''
    Termset is any non-empty subset K of the terms of a query Q        
    '''
    
#     print('TSInter\n')
#     pp(P)
#     print('\n====================================\n')

    
    for ( Ti , Tj ) in itertools.combinations(P,2):
        
#         print('\nTESTANDO UNION {} \n {} \n'.format(Ti,Tj))
        
        
#         print('´´´´´´´TSInter\n')
#         pp(P)
        
        Tx = Ti.union(Tj, changeSources = True)        
        
#         print('\nUNION COMPILADO de {} \n {} \n {}\n\n\n'.format(Ti,Tj,Tx))
        
#         if Tx is not None:
#             print(len(Tx.tuples), 'tuples on union')
            
#         print('´´´´´´´TSInter\n')
#         pp(P)    
        
        
        if Tx is not None and Tx.hasTuples():            
            P.add(Tx)
            
            if Ti.hasTuples() == False:
#                 print('Ti {} has not tuples',Ti)
                P.remove(Ti)
#             else:
#                 print('{} has {} tuples'.format(Ti,len(Ti.tuples)))
                
            if Tj.hasTuples() == False:
#                 print('Tj {} has not tuples',Tj)
                P.remove(Tj)
#             else:
#                 print('{} has {} tuples'.format(Tj,len(Tj.tuples)))
            
            TSInterMartins(P)
            break
```

```python
def getQuerySets(filename='querysets/queryset_dblp_martins.txt'):
    QuerySet = []
    with open(filename,encoding='utf-8-sig') as f:
        for line in f.readlines():
            
            #The line bellow Remove words not in OLIVEIRA experiments
            #Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
            Q = tuple([word.strip(string.punctuation) for word in line.lower().split() if word not in stw_set])
            
            QuerySet.append(Q)
    return QuerySet
```

```python
Q= ['author','datacenter','2015']
```

```python
Rq = TSFindClass(Q,wh)
```

```python
Rq
```

## class SchemaGraph

```python
import pprint 

class SchemaGraph:
    
    def __init__(self):
        self.__graph = {}
    
    def addRelationship(self,tableA,columnA,tableB, columnB, direction = -1):        
        tsA = Tupleset(tableA)
        tsB = Tupleset(tableB)
        
        #A->B
        edge_info = (columnA,columnB,direction)
        self.__graph.setdefault(tsA,{}).setdefault(tsB,[]).append(edge_info)
        
        #B<-A
        edge_info = (columnB,columnA,direction*-1)
        self.__graph.setdefault(tsB,{}).setdefault(tsA,[]).append(edge_info)
        
    def getEdgeInfos(self,tsA,tsB):        
        return self.__graph[tsA][tsB]        
        
    def copyRelationships(self,sourceNode,targetNode):
        # target->neighbours    =    source->neighbours
        
        #print('cpRelations s: {} t: {}'.format(sourceNode,targetNode))
        
        self.__graph[targetNode] = copy.deepcopy(self.__graph[sourceNode])
            
        
        # neighbours->target    =    neighbours->source
        for neighbourNode in self.__graph[targetNode]:
            for node, edge_infos in self.__graph[neighbourNode].items():
                if node == sourceNode:
                    self.__graph[neighbourNode][targetNode] = edge_infos
                    break
                    
        
    def getMatchGraph(self,Match):
        
        Gts = copy.deepcopy(self)
        
        for ts in Match:
            Gts.copyRelationships(Tupleset(ts.table),ts)
            
        return Gts
    
    def getByTableName(self,tableName):
        return self.__graph[Tupleset(tableName)]
    
    def tables(self):
        return self.__graph.keys()
        
    def getAdjacentTables(self, table, sort = False):
        
        if not sort:
            return self.getByTableName(table).keys()
        else:
            # Sorting adjacents with non free tuple sets first
            return sorted(self.getByTableName(table).keys(),key=lambda ts : ts.isFreeTupleset() )
        
    def isJNTSound(self,Ji):
        if len(Ji)<3:
            return True
        
        #check if there is a case A->B<-C, when A.table=C.table
        
        for i in range(len(Ji)-2):
            tsA = Ji[i]
            tsB = Ji[i+1]
            tsC = Ji[i+2]
            
            if tsA.table == tsB.table:
                            
                for edge_info in self.__graph[tsA][tsB]:
                    (columnA,columnB,direction) = edge_info
                    
                    if direction == 1:
                        return False
        return True    
    
        
    def __repr__(self):
        return pprint.pformat(self.__graph)
    
    def __str__(self):
        return repr(self.__graph)
```

```python
def getSchemaGraph(dbname='dblp',user='imdb',password='imdb'):
    #Output: A Schema Graph G  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }
    
    G = SchemaGraph()
    with psycopg2.connect(dbname=dbname,user=user,password=password) as conn:
            with conn.cursor() as cur:
                sql = "SELECT DISTINCT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name FROM information_schema.table_constraints AS tc              JOIN information_schema.key_column_usage AS kcu                 ON tc.constraint_name = kcu.constraint_name             JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name WHERE constraint_type = 'FOREIGN KEY'"
                cur.execute(sql)
                relations = cur.fetchall()

                for (table,column,foreign_table,foreign_column) in relations:
                    #print('table,column,foreign_table,foreign_column\n{}, {}, {}, {}'.format(table,column,foreign_table,foreign_column))
                    G.addRelationship(table,column,foreign_table,foreign_column)  
                print ('SCHEMA CREATED')          
    return G
```

```python
G=getSchemaGraph()

G
```

## Class Similarities

```python
import copy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

class Similarities:
    
    def __init__(self, model, attributeHash,schemaGraph):

        self.model = model
        self.attributeHash = attributeHash
        self.schemaGraph = schemaGraph
        
        self.loadEmbeddingHashes()
    
    
    def wordnet_similarity(self,wordA,wordB):
        A = set(wn.synsets(wordA))
        B = set(wn.synsets(wordB))

        wupSimilarities = [0]
        pathSimilarities = [0]
        
        for (sense1,sense2) in itertools.product(A,B):        
            wupSimilarities.append(wn.wup_similarity(sense1,sense2) or 0)
            pathSimilarities.append(wn.path_similarity(sense1,sense2) or 0)
            
        return max(max(wupSimilarities),max(pathSimilarities))

    def jaccard_similarity(self,wordA,wordB):

        A = set(wordA)
        B = set(wordB)

        return len(A & B ) / len(A | B)
    
    
    def embedding10_similarity(self,word,table,column='*',Emb='B'):
        wnl = WordNetLemmatizer()
        
        # Os sinônimos do EmbA também são utilizados por todos
        sim_list = self.EmbA[table][column]
        
        if column != '*':
        
            if Emb == 'B':
                sim_list |= self.EmbB[table][column]

            elif Emb == 'C':
                sim_list |= self.EmbC[table][column]

        return wnl.lemmatize(word) in sim_list
    
    
    def embedding_similarity(self,wordA,wordB):
        if wordA not in self.model or wordB not in self.model:
            return 0
        return self.model.similarity(wordA,wordB)
    
    
    def word_similarity(self,word,table,column = '*',
                    wn_sim=True, 
                    jaccard_sim=True,
                    emb_sim=False,
                    emb10_sim='B'):
        sim_list=[0]
    
        if column == '*':
            schema_term = table
        else:
            schema_term = column

        if wn_sim:
            sim_list.append( self.wordnet_similarity(schema_term,word) )

        if jaccard_sim:
            sim_list.append( self.jaccard_similarity(schema_term,word) )

        if emb_sim:
            sim_list.append( self.embedding_similarity(schema_term,word) )

        sim = max(sim_list) 

        if emb10_sim:
            if self.embedding10_similarity(word,table,column,emb10_sim):
                if len(sim_list)==1:
                    sim=1
            else:
                sim=0
                
        print('sim({},{}.{}) = {}'.format(word,table,column,sim))        
        
        return sim    
    
    def __getSimilarSet(self,word, inputType = 'word'):
        if inputType == 'vector':
            sim_list = model.similar_by_vector(word)
        else:
            sim_list = model.most_similar(word)        
        return  {word.lower() for word,sim in sim_list}
    
    def loadEmbeddingHashes(self,weight=0.5):
        
        self.EmbA = {}
        self.EmbB = {}
        self.EmbC = {}
    
        for table in self.attributeHash:

            if table not in self.model:
                continue

            self.EmbA[table]={}
            self.EmbB[table]= {}
            self.EmbC[table]= {}
            
            self.EmbA[table]['*'] = self.__getSimilarSet(table) 

            for column in self.attributeHash[table]:
                if column not in model or column=='id':
                    continue
                
                self.EmbA[table][column]=self.__getSimilarSet(column)
                
                self.EmbB[table][column]=self.__getSimilarSet( (table,column) )
                  
                avg_vec = (model[table]*weight + model[column]*(1-weight))                   
                self.EmbC[table][column] = self.__getSimilarSet(avg_vec, inputType = 'vector')
                
                
                
        G = self.schemaGraph
        for tableA in G.tables():

            if tableA not in self.attributeHash or tableA not in model:
                continue

            for tableB in G.getAdjacentTables(tableA):

                if tableB not in self.attributeHash or tableB not in model:
                    continue

                self.EmbB[tableB][tableA] = self.EmbB[tableA][tableB] = self.__getSimilarSet( (tableA,tableB) )

        
```

```python
def SchSFind(Q,attributeHash,threshold=0.8, 
             sim_args={}):    
    S = set()
    
    sm = Similarities(model,ah,G)
    
    for keyword in Q:
        for table in attributeHash:            
            for attribute in ['*']+list(attributeHash[table].keys()):
                
                if(attribute=='id'):
                    continue
                
                sim = sm.word_similarity(keyword,table,attribute,**sim_args)
                
                if sim >= threshold:
                    ts = Tupleset(table)
                    ts.addSchemaMapping(keyword,attribute)
                    S.add(ts)
                    
    return S
```

```python
Sq = SchSFind(Q,ah)
```

```python
Sq
```

```python
def MinimalCover(MC, Q):
    #Input:  A subset MC (Match Candidate) to be checked as total and minimal cover
    #Output: If the match candidate is a TOTAL and MINIMAL cover

    Subset = [ts.getKeywords() for ts in MC]
    u = set().union(*Subset)    
    
    isTotal = (u == set(Q))
    for element in Subset:
        
        new_u = list(Subset)
        new_u.remove(element)
        
        new_u = set().union(*new_u)
        
        if new_u == set(Q):
            return False
    
    #print('MC({},{}) = {}'.format(MC,Q,isTotal))
    
    return isTotal
```

```python
def QMGen(Q,Rq):
    #Input:  A keyword query Q, The set of non-empty non-free tuple-sets Rq
    #Output: The set Mq of query matches for Q
    
    '''
    Query match is a set of tuple-sets that, if properly joined,
    can produce networks of tuples that fulfill the query. They
    can be thought as the leaves of a Candidate Network.
    
    '''
    
    Mq = []
    for i in range(1,len(Q)+1):
        for subset in itertools.combinations(Rq,i):            
            if(MinimalCover(subset,Q)):
                print('----------------------------------------------\nM')
                pp(set(subset))
                print('\n')
                
                M = MInter(set(subset))
                print('subset')
                pp(M)
                Mq.append(M)
                
                
    return Mq

def MInter(M):  
    somethingChanged = False    
    for tsA, tsB  in itertools.combinations(M,2):
        
        tsX = tsA.union(tsB, projectionOnly = True)

        if tsX is not None:
            
            M.add(tsX)      
            M.remove(tsA)
            M.remove(tsB)
            
            return MInter(M)
        
    return M   
```

```python

```

```python
Rq
```

```python
Sq
```

```python
Mq = QMGen(Q,Rq|Sq)
```

```python
for M in Mq:
    pp(M)
    print('\n')
```

```python
def QMRank(Mq, wordHash,attributeHash):
    Ranking = []  
    sm = Similarities(model,ah,G)
    for M in Mq:
        #print('=====================================\n')
        valueProd = 1 
        schemaProd = 1
        score = 1
        
        thereIsSchemaTerms = False
        thereIsValueTerms = False
        
        for ts in M:
            #print(ts)
            for table, attribute, valueWords in ts.getValueMappings():
                #print('t{} a{} v{}'.format(table,attribute,valueWords))             
                
                (Norm,numDistinctWords,numWords,maxFrequency) = attributeHash[table][attribute]                
                wsum = 0
                for term in valueWords:
                
                    #print('t{} a{} vt{}'.format(table,attribute,term))
                
                    IAF = wordHash.getIAF(term)
                    
                    frequency = len(wordHash.getMappings(term,table,attribute))
                    TF = frequency/maxFrequency
                    wsum = wsum + TF*IAF
    
                    thereIsValueTerms = True
                
                cos = wsum/Norm
                valueProd *= cos
                
                
            for table, attribute, schemaWords in ts.getSchemaMappings():
                schemasum = 0
                for term in schemaWords:
                    sim = sm.word_similarity(term,table,attribute)
                    schemasum += sim
                    
                    thereIsSchemaTerms = True
                    
                schemaProd *= schemasum           
        
        valueScore  = valueProd
        schemaScore = schemaProd
        
        if thereIsValueTerms:
            score *= valueScore
        else:
            valueScore = 0
            
            
        if thereIsSchemaTerms:
            score *= schemaScore
        else:
            schemaScore = 0
                
        Ranking.append( (M,score,valueScore,schemaScore) )
                
    return sorted(Ranking,key=lambda x: x[1],reverse=True)
                
```

```python
RankedMq = QMRank(Mq,wh,ah)

RankedMq
```

```python
G
```

```python
M = RankedMq[0][0]
M
```

```python
Gts = G.getMatchGraph(M)
```

```python
from queue import deque
def SingleCN(FM,Gts,TMax,showLog=False):  
  
    if showLog:
        print('================================================================================\nSINGLE CN')
        print('Tmax ',TMax)
        print('FM')
        pp(FM)
        
        #print('\n\nGts')
        #pp(Gts)
        #print('\n\n')
    
    F = deque()

    first_element = list(FM)[0]
    J = [first_element]
    
    if len(FM)==1:
        return J
    
    F.append(J)
    
    while F:
        J = F.popleft()           
        tsu = J[-1]
        
        sortedAdjacents = Gts.getAdjacentTables(tsu.table,sort = True)
        
        if showLog:
            print('--------------------------------------------\nParctial CN')
            print('J ',J,'\n')

            print('\nAdjacents:')
            pp(Gts.getAdjacentTables(tsu.table))
            
            print('\nSorted Adjacents:')
            pp(sortedAdjacents)
            
            print('F:')
            pp(F)
        
        for tsv in sortedAdjacents:
            
            if showLog:
                print('Checking adj:')
                pp(tsv)
                print()

            if (tsv.isFreeTupleset()) or (tsv not in J):
                
                Ji = J + [tsv]
                
                if (Ji not in F) and (len(Ji)<TMax) and (Gts.isJNTSound(Ji)):
                    
                    if showLog:
                        print('isSound=True')
                    
                    containsMatch = True
                    for ts in FM:
                        if ts not in Ji:
                            containsMatch = False    
                            
                    if containsMatch:
                        if showLog:
                            print('--------------------------------------------\nGenerated CN')
                            print('J ',Ji,'\n')
                        
                        return Ji
                    else:
                        F.append(Ji)
```

```python
def MatchCN(G,Sq,Rq,RankedMq,TMax=5):    
    Cns = []                        
    for  (M,score,schemascore,valuescore) in RankedMq:

        Gts = G.getMatchGraph(M)
        Cn = SingleCN(M,Gts,TMax=TMax)
        if(Cn is not None):
            
            
            #Dividindo score pelo tamanho da cn (SEGUNDA PARTE DO RANKING)
            
            CnScore = score/len(Cn)
            
            Cns.append( (Cn,Gts,CnScore,schemascore,valuescore) )
    
    #Ordena CNs pelo CnScore
    RankedCns=sorted(Cns,key=lambda x: x[3],reverse=True)
    
    return RankedCns
```

```python

```

```python
Cns = MatchCN(G,Sq,Rq,RankedMq)
for (Cn,Gts,CnScore,schemascore,valuescore) in Cns:
    pp(Cn)
    
    x = Cn
    y = Gts
    break
    print('\n')
```

## getSQLfromCN

```python
def getSQLfromCN(Gts,Cn,contract=True):
    selected_attributes = [] 
    hashTables = {}
    conditions=[]
    relationships = set()
    
    tables_id=[]
    tables=[]
    joincondiditions=[]
    
    for i in range(len(Cn)):
        
        tsA = Cn[i]
               
        A = 't' + str(i)
        
        if contract and tsA.isFreeTupleset():
            A = hashTables.setdefault(tsA.table,[A])[0]
        else:
            hashTables.setdefault(tsA.table, []).append(A)            
        
        for attr in tsA.getAttributes():
            selected_attributes.append(A +'.'+ attr)
        
        for table, attr, valueWords in tsA.getValueMappings():
            #tratamento de keywords
            for term in valueWords:
                condition = 'CAST('+A +'.'+ attr + ' AS VARCHAR) ILIKE \'%' + term + '%\''
                conditions.append(condition)
        
        
        #tratamento de join paths
        if (i>0):
            # B se refere ao tupleset anterior                
            tsB = Cn[i-1]
            
            # B vai receber o último valor de tx adicionado em hashTables[tableB]
            B = hashTables[tsB.table][-1]
            
            for joining_attrA,joining_attrB, direction in Gts.getEdgeInfos(tsA,tsB):            
                
                joincondiditions.append(A + '.' + joining_attrA + ' = ' + B + '.' + joining_attrB)
                relationships.add( frozenset([B,A]) ) 
    
    for tableX in hashTables.keys():
        for tx in hashTables[tableX]:
            tables_id.append(tx+'.__search_id')
            tables.append(tableX+' '+tx)
            
        
    relationshipsText = ['('+a+'.__search_id'+','+b+'.__search_id'+')' for (a,b) in relationships]
    
    sqlText = 'SELECT \n '
#     sqlText +=' ('+', '.join(tables_id)+') AS Tuples,\n '
#     if len(relationships)>0:
#         sqlText +='('+', '.join(relationshipsText)+') AS Relationships,\n '
        
    sqlText += ' ,\n '.join(selected_attributes)
    
    sqlText +='\nFROM\n ' + ',\n '.join(tables)
    
    sqlText +='\nWHERE\n '
    
    # Considerando que todas as pequisas tem ao menos um value term
    if  len(conditions)==0:
        sqlText+= ' 1=2'
        return sqlText
    
    sqlText +='\n AND '.join(joincondiditions)
    sqlText +='\n'
    if len(joincondiditions)>0:
        sqlText +='\n AND '
    sqlText +='\n AND '.join(conditions)
    
    
    #Considerando que nenhuma consulta tem mais de 1000 linhas no resultado
    sqlText += '\n LIMIT 1000'
    
    sqlText += ';'
    '''
    print('SELECT:\n',selected_attributes)
    print('TABLES:\n',hashTables)
    print('CONDITIONS:')
    pp(conditions)
    print('RELATIONSHIPS:')
    pp(relationships)
    '''    
    #print('SQL:\n',sql)
    return sqlText
```

```python
pp(x)
```

```python
sql = getSQLfromCN(y,x,contract=True)

print(sql)
```

```python
G
```
