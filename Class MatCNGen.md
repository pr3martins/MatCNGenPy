---
jupyter:
  jupytext:
    formats: md,ipynb,py
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
DBNAME = 'mondial_coffman'
DBUSER = 'imdb'
DBPASS = 'imdb'
EMBEDDINGFILE = "word_embeddings/word2vec/GoogleNews-vectors-negative300.bin"
QUERYSETFILE ='querysets/queryset_mondial_coffman_original.txt'
GOLDEN_CANDIDATE_NETWORKS_PATH ='golden_candidate_networks/mondial_coffman'

STEP_BY_STEP = True
PREPROCESSING = True
CUSTOM_QUERY = ('poland', 'cape', 'verde', 'organization')
```

```python
def validSchemaElement(text,embmodel=set()): 
    if 'id' in text or 'index' in text or 'code' in text or 'nr' in text:
        return False
    return True    
```

```python
from pprint import pprint as pp
import gc #garbage collector usado no createinvertedindex


import psycopg2
from psycopg2 import sql
import string

import nltk 
from nltk.corpus import stopwords

import gensim.models.keyedvectors as word2vec
from gensim.models import KeyedVectors

from math import log1p 
import copy
import itertools
import pprint 
import copy
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from queue import deque

from collections import Counter #used to check whether two CandidateNetworks are equal

import glob #to list filenames in directories
import re # used to parse string to keyword match class
import json #used to parse class to json

from collections import OrderedDict #To print dict results according to ordered queryset
```

```python
def tokenizeString(text):     
    return [word.strip(string.punctuation)
            for word in text.lower().split() 
            if word not in stopwords.words('english') or word == 'will']
    return [word
            for word in text.translate({ord(c):' ' for c in string.punctuation if c!='_'}).lower().split() 
            if word not in stopwords.words('english') or word == 'will']
```

```python
def loadWordEmbeddingsModel(filename = EMBEDDINGFILE):
    model = KeyedVectors.load_word2vec_format(filename,
                                                       binary=True, limit=500000)
    return model
```

```python
if STEP_BY_STEP and PREPROCESSING:
    wordEmbeddingsModel=loadWordEmbeddingsModel(EMBEDDINGFILE)
```

During the process of generating SQL queries, Lathe uses two data structures
which are created in a **Preprocessing stage**: the Value Index and the Schema Index.

The Value Index is an inverted index stores the occurrences of keyword in the database,
indicating the relations and tuples a keyword appear and which attributes are mapped
to that keyword. These occurrences are retrieved in the Query Matching phase. In
addition, the Value Index is also used to calculate term frequencies for the Ranking of
Query Matches. The Schema Index is an inverted index that stores information about
the database schema and statics about ranking of attributes, which is also used in the
Query Matches Ranking.

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
class DatabaseIter:
    def __init__(self,embeddingModel,dbname=None,user=None,password=None):
        if dbname is None:
            self.dbname=DBNAME
        else:
            self.dbname=dbname        
        if DBUSER is None:
            self.user=DBUSER
        else:
            self.user=user
        if DBPASS is None:
            self.password=DBPASS
        else:
            self.password=password   

        self.embeddingModel=embeddingModel

    def __iter__(self):
        with psycopg2.connect(dbname=self.dbname,user=self.user,password=self.password) as conn:
            with conn.cursor() as cur:

                GET_TABLE_AND_COLUMNS_WITHOUT_FOREIGN_KEYS_SQL='''
                    SELECT
                        c.table_name, 
                        c.column_name
                    FROM 
                        information_schema.table_constraints AS tc 
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                          AND tc.constraint_type = 'FOREIGN KEY' 
                        RIGHT JOIN information_schema.columns AS c 
                          ON c.table_name=tc.table_name
                          AND c.column_name = kcu.column_name
                          AND c.table_schema = kcu.table_schema
                    WHERE
                        c.table_schema='public'
                        AND tc.constraint_name IS NULL;
                ''' 
                cur.execute(GET_TABLE_AND_COLUMNS_WITHOUT_FOREIGN_KEYS_SQL)
                table_hash = {}
                for table,column in cur.fetchall():
                    table_hash.setdefault(table,[]).append(column)
                    
                for table,columns in table_hash.items():

                    indexable_columns = [col for col in columns if validSchemaElement(col)]

                    if len(indexable_columns)==0:
                        continue
                    
                    print('\nINDEXING {}({})'.format(table,', '.join(indexable_columns)))
                    
                    '''
                    NOTE: Table and columns can't be directly passed as parameters.
                    Thus, the sql.SQL command with sql.Identifiers is used
                    '''
                    cur.execute(
                        sql.SQL("SELECT ctid, {} FROM {};")
                        .format(sql.SQL(', ').join(sql.Identifier(col) for col in indexable_columns),
                                sql.Identifier(table))
                            )  
                                       
                    for i,row in enumerate(cur.fetchall()): 
                        ctid = row[0]
                        for col in range(1,len(row)):
                            column = cur.description[col][0]
                            for word in tokenizeString( str(row[col]) ):
                                yield table,ctid,column, word
                        
                        if i%100000==1:
                            print('*',end='')
```

```python
def createInvertedIndex(embeddingModel,showLog=True):
    #Output: wordHash (Term Index) with this structure below
    #map['word'] = [ 'table': ( {column} , ['ctid'] ) ]

    '''
    The Term Index is built in a preprocessing step that scans only
    once all the relations over which the queries will be issued.
    '''

    
    wh = WordHash()
    ah = {}
    
    previousTable = None
    
    for table,ctid,column,word in DatabaseIter(embeddingModel):        
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

    print ('\nINVERTED INDEX CREATED')
    gc.collect()
    return wh,ah
```

```python
if STEP_BY_STEP and PREPROCESSING:
    (wordHash,attributeHash) = createInvertedIndex(wordEmbeddingsModel)
```

```python
def processIAF(wordHash,attributeHash):
    
    total_attributes = sum([len(attribute) for attribute in attributeHash.values()])
    
    for (term, values) in wordHash.items():
        attributes_with_this_term = sum([len(attribute) for attribute in wordHash[term].values()])
        IAF = log1p(total_attributes/attributes_with_this_term)
        wordHash.setIAF(term,IAF)        
        
    print('IAF PROCESSED')
```

```python
if STEP_BY_STEP and PREPROCESSING:
    processIAF(wordHash,attributeHash)
```

```python
def processNormsOfAttributes(wordHash,attributeHash):    
    for word in wordHash:
        for table in wordHash[word]:
            for column, ctids in wordHash[word][table].items():
                   
                (prevNorm,numDistinctWords,numWords,maxFrequency) = attributeHash[table][column]

                IAF = wordHash.getIAF(word)

                frequency = len(ctids)
                
                TF = frequency/maxFrequency
                
                Norm = prevNorm + (TF*IAF)

                attributeHash[table][column]=(Norm,numDistinctWords,numWords,maxFrequency)
                
    print ('NORMS OF ATTRIBUTES PROCESSED')
```

```python
if STEP_BY_STEP and PREPROCESSING:
    processNormsOfAttributes(wordHash,attributeHash)
```

```python
def preProcessing(emb_file=EMBEDDINGFILE):
    wordEmbeddingsModel=loadWordEmbeddingsModel(emb_file)
    (wordHash,attributeHash) = createInvertedIndex(wordEmbeddingsModel)
    processIAF(wordHash,attributeHash)
    processNormsOfAttributes(wordHash,attributeHash)
    
    print('PRE-PROCESSING STAGE FINISHED')
    return (wordHash,attributeHash,wordEmbeddingsModel)
```

```python
if not STEP_BY_STEP and PREPROCESSING:
    (wordHash,attributeHash,wordEmbeddingsModel) = preProcessing()
```

# Processing Stage

```python
def getQuerySets(filename=QUERYSETFILE):
    QuerySet = []
    with open(filename,encoding='utf-8-sig') as f:
        for line in f.readlines():
            
            #The line bellow Remove words not in OLIVEIRA experiments
            #Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
            Q = tuple(tokenizeString(line))
            
            QuerySet.append(Q)
    return QuerySet
```

```python
if STEP_BY_STEP:
    QuerySets = getQuerySets(QUERYSETFILE)
    if CUSTOM_QUERY is None:
        Q = QuerySets[0]
    else:
        Q = CUSTOM_QUERY
    print(Q)
```

## Class Graph

```python
class Graph:
    __slots__ = ['__graph_dict','__enable_edge_info','__edges_info']
    def __init__(self, graph_dict=None, has_edge_info=False):
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict
        
        self.__enable_edge_info = has_edge_info
        if has_edge_info:
            self.__edges_info = {}
        else:
            self.__edges_info = None

    def add_vertex(self, vertex):
        self.__graph_dict.setdefault(vertex, (set(),set()) )
        return vertex
    
    def add_outgoing_neighbour(self,vertex,neighbour):
        self.__graph_dict[vertex][0].add(neighbour)
    
    def add_incoming_neighbour(self,vertex,neighbour):
        self.__graph_dict[vertex][1].add(neighbour)
    
    def outgoing_neighbours(self,vertex):
        yield from self.__graph_dict[vertex][0]
           
    def incoming_neighbours(self,vertex):
        yield from self.__graph_dict[vertex][1]
    
    def neighbours(self,vertex):
        #This method does not use directed_neighbours to avoid generating tuples with directions
        yield from self.outgoing_neighbours(vertex)
        yield from self.incoming_neighbours(vertex)
        
    def directed_neighbours(self,vertex):
        #This method does not use directed_neighbours to avoid generating tuples with directions
        for outgoing_neighbour in self.outgoing_neighbours(vertex):
            yield ('>',outgoing_neighbour)
        for incoming_neighbour in self.incoming_neighbours(vertex):
            yield ('<',incoming_neighbour)

    def add_edge(self, vertex1, vertex2,edge_info = None, edge_direction='>'):
        if edge_direction=='>':        
            self.add_outgoing_neighbour(vertex1,vertex2)
            self.add_incoming_neighbour(vertex2,vertex1)

            if self.__enable_edge_info:
                self.__edges_info[(vertex1, vertex2)] = edge_info
        elif edge_direction=='<':        
            self.add_edge(vertex2, vertex1,edge_info = edge_info)
        else:
            raise SyntaxError('edge_direction must be > or <')
    
    def get_edge_info(self,vertex1,vertex2):
        if self.__enable_edge_info == False:
            return None
        return self.__edges_info[(vertex1, vertex2)]
    
    def vertices(self):
        return self.__graph_dict.keys()
            
    def edges(self):
            for vertex in self.vertices():
                for neighbour in self.outgoing_neighbours(vertex):
                    yield (vertex,)+ (neighbour,)
    
    def dfs_pair_iter(self, source_iter = None, root_predecessor = False):
        last_vertex_by_level=[]
        
        if source_iter is None:
            source_iter = self.leveled_dfs_iter()
        
        for direction,level,vertex in source_iter:
            if level < len(last_vertex_by_level):
                last_vertex_by_level[level] = vertex
            else:
                last_vertex_by_level.append(vertex)
            
            if level>0:
                prev_vertex = last_vertex_by_level[level-1]
                yield (prev_vertex,direction,vertex)
            elif root_predecessor:
                yield (None,'',vertex)
        
                
    def leveled_dfs_iter(self,start_vertex=None,visited = None, level=0, direction='',two_way_transversal=True):
        if len(self)>0:
            if start_vertex is None:
                start_vertex = self.get_starting_vertex()             
            if visited is None:
                visited = set()
            visited.add(start_vertex)

            yield( (direction,level,start_vertex) )
            
            for neighbour in self.outgoing_neighbours(start_vertex):
                if neighbour not in visited:
                    yield from self.leveled_dfs_iter(neighbour,visited,
                                                     level=level+1,
                                                     direction='>',
                                                     two_way_transversal=two_way_transversal) 
            
            # two_way_transversal indicates whether the DFS will expand through incoming neighbours
            if two_way_transversal:
                for neighbour in self.incoming_neighbours(start_vertex):
                    if neighbour not in visited:
                        yield from self.leveled_dfs_iter(neighbour,visited,
                                                         level=level+1,
                                                         direction='<',
                                                         two_way_transversal=two_way_transversal)   
    
    
    def leaves(self):
        for vertice in self.vertices():            
            if sum(1 for neighbour in self.neighbours(vertice)) == 1:
                yield vertice
    
    def get_starting_vertex(self):
        return next(iter(self.vertices()))
    
    def pp(self):
        pp(self.__graph_dict)
    
    def __repr__(self):
        return repr(self.__graph_dict)
    
    def __len__(self):
         return len(self.__graph_dict)
        
    def str_graph_dict(self):
        return str(self.__graph_dict)
    
    def str_edges_info(self):
        return str(self.__edges_info)
```

```python
def getSchemaGraph(dbname=None,user=None,password=None):
    
    if dbname is None:
        dbname=DBNAME
    if DBUSER is None:
        user=DBUSER
    if DBPASS is None:
        password=DBPASS
    
    #Output: A Schema Graph G  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }
    
    G = Graph(has_edge_info=True)
    with psycopg2.connect(dbname=dbname,user=user,password=password) as conn:
            with conn.cursor() as cur:
                sql = "SELECT DISTINCT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name FROM information_schema.table_constraints AS tc              JOIN information_schema.key_column_usage AS kcu                 ON tc.constraint_name = kcu.constraint_name             JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name WHERE constraint_type = 'FOREIGN KEY'"
                cur.execute(sql)
                relations = cur.fetchall()

                for (table,column,foreign_table,foreign_column) in relations:
                    #print('table,column,foreign_table,foreign_column\n{}, {}, {}, {}'.format(table,column,foreign_table,foreign_column))
                    G.add_vertex(table)
                    G.add_vertex(foreign_table)
                    G.add_edge(table,foreign_table, (column,foreign_column) )
                print ('SCHEMA CREATED')          
    return G
```

```python
if STEP_BY_STEP:
    G = getSchemaGraph()  
    print(G)
    for direction,level,vertex in G.leveled_dfs_iter():
        print(level*'\t',direction,vertex)
    print([x for x in G.dfs_pair_iter(root_predecessor=True)])
```

## Class KeywordMatch

```python
class KeywordMatch:
   
    def __init__(self, table, value_filter={},schema_filter={}):  
        self.__slots__ =['table','schema_filter','value_filter']
        self.table = table        
        self.schema_filter= frozenset({ (key,frozenset(keywords)) for key,keywords in schema_filter.items()})            
        self.value_filter= frozenset({ (key,frozenset(keywords)) for key,keywords in value_filter.items()})            

    def is_free(self):
        return len(self.schema_filter)==0 and len(self.value_filter)==0
        
    def schema_mappings(self): 
        for attribute, keywords in self.schema_filter:
            yield (self.table,attribute,keywords)
    
    def value_mappings(self): 
        for attribute, keywords in self.value_filter:
            yield (self.table,attribute,keywords)
            
    def mappings(self):
        for attribute, keywords in self.schema_filter:
            yield ('s',self.table,attribute,keywords)
        for attribute, keywords in self.value_filter:
            yield ('v',self.table,attribute,keywords)
                
    def keywords(self,schema_only=False):
        for attribute, keywords in self.schema_filter:
            yield from keywords
        if not schema_only:
            for attribute, keywords in self.value_filter:
                yield from keywords
            
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        str_km = ""
        
        def str_filter(filter_type,fltr):
            if len(fltr) == 0:
                return ""  
            
            return ".{}({})".format(
                filter_type,
                ','.join(
                    [ "{}{{{}}}".format(
                        attribute,
                        ','.join(keywords)
                        ) for attribute,keywords in fltr
                    ]
                )
            )
        
        return "{}{}{}".format(self.table.upper(),str_filter('s',self.schema_filter),str_filter('v',self.value_filter)) 
    
    def __eq__(self, other):
        return (isinstance(other, KeywordMatch)
                and self.table == other.table
                and set(self.keywords(schema_only=True)) == set(other.keywords(schema_only=True))
                and self.value_filter == other.value_filter)
    
    def __hash__(self):
        return hash( (self.table,frozenset(self.keywords(schema_only=True)),self.value_filter) )
    
    def to_json_serializable(self):
        
        def filter_object(fltr):
            return [{'attribute':attribute,
                            'keywords':list(keywords)} for attribute,keywords in fltr]
        
        return {'table':self.table,
                'schema_filter':filter_object(self.schema_filter),
                'value_filter':filter_object(self.value_filter),}
    
    def to_json(self):
        return json.dumps(self.to_json_serializable())
    
    @staticmethod
    def from_str(str_km):
        re_km = re.compile('([A-Z,_,1-9]+)(.*)')
        re_filters = re.compile('\.([vs])\(([^\)]*)\)')
        re_predicates = re.compile('([\w\*]*)\{([^\}]*)\}\,?')
        re_keywords = re.compile('(\w+)\,?')

        m_km=re_km.match(str_km)
        table = m_km.group(1).lower()
        schema_filter={}
        value_filter={}
        for filter_type,str_predicates in re_filters.findall(m_km.group(2)):

            if filter_type == 'v':
                predicates = value_filter
            else:
                predicates = schema_filter
            for attribute,str_keywords in re_predicates.findall(str_predicates):
                predicates[attribute]={key for key in re_keywords.findall(str_keywords)}
        return KeywordMatch(table,value_filter=value_filter,schema_filter=schema_filter,)
    
    
    def from_json_serializable(json_serializable):
        
        def filter_dict(filter_obj):
            return {predicate['attribute']:predicate['keywords'] for predicate in filter_obj}

        return KeywordMatch(json_serializable['table'],
                            value_filter  = filter_dict(json_serializable['value_filter']),
                            schema_filter  = filter_dict(json_serializable['schema_filter']),)
    
    def from_json(str_json):
        return KeywordMatch.from_json_serializable(json.loads(str_json))
```

```python
kmx= KeywordMatch.from_str('CHARACTER.s(name{name}).v(name{scissorhands,edward},birthdate{1997})')
KeywordMatch.from_json(kmx.to_json())
```

```python
def VKMGen(Q,wordHash):
    #Input:  A keyword query Q=[k1, k2, . . . , km]
    #Output: Set of non-free and non-empty tuple-sets Rq

    '''
    The tuple-set Rki contains the tuples of Ri that contain all
    terms of K and no other keywords from Q
    '''
    
    #Part 1: Find sets of tuples containing each keyword
    P = {}
    for keyword in Q:
        
        if keyword not in wordHash:
            continue
        
        for table in wordHash[keyword]:
            for (attribute,ctids) in wordHash[keyword][table].items():
                vkm = KeywordMatch(table, value_filter={attribute:{keyword}})
                P[vkm] = set(ctids)
    
    #Part 2: Find sets of tuples containing larger termsets
    TSInterMartins(P)
    
    #Part 3: Ignore tuples
    return set(P)

def TSInterMartins(P):
    #Input: A Set of non-empty tuple-sets for each keyword alone P 
    #Output: The Set P, but now including larger termsets (process Intersections)

    '''
    Termset is any non-empty subset K of the terms of a query Q        
    '''
    
    for ( vkm_i , vkm_j ) in itertools.combinations(P,2):
        
        
        if (vkm_i.table == vkm_j.table and
            set(vkm_i.keywords()).isdisjoint(vkm_j.keywords())
           ):
            
            jointTuples = P[vkm_i] & P[vkm_j]
            
            if len(jointTuples)>0:
                                
                joint_predicates = {}
                
                for attribute, keywords in vkm_i.value_filter:
                    joint_predicates.setdefault(attribute,set()).update(keywords)
                
                for attribute, keywords in vkm_j.value_filter:
                    joint_predicates.setdefault(attribute,set()).update(keywords)
                
                vkm_ij = KeywordMatch(vkm_i.table,value_filter=joint_predicates)
                P[vkm_ij] = jointTuples
                                
                P[vkm_i].difference_update(jointTuples)
                if len(P[vkm_i])==0:
                    del P[vkm_i]
                
                P[vkm_j].difference_update(jointTuples)
                if len(P[vkm_j])==0:
                    del P[vkm_j]                

                return TSInterMartins(P)
    return
```

```python
if STEP_BY_STEP:
    print('FINDING TUPLE-SETS')
    Rq = VKMGen(Q, wordHash)
    print(len(Rq),'TUPLE-SETS CREATED\n')
    pp(Rq)
```

## Class Similarities

```python
class Similarities:    
    def __init__(self, model, attributeHash,schemaGraph,
                usePathSim=True,useWupSim=True,
                useJaccardSim=True,useEmbSim=False,
                useEmb10Sim=True,Emb10SimType='B',
                ):
        
        self.model = model
        self.attributeHash = attributeHash
        self.schemaGraph = schemaGraph
        
        self.usePathSim=usePathSim, 
        self.useWupSim=useWupSim,
        self.useJaccardSim=useJaccardSim,
        self.useEmbSim=useEmbSim,
        self.useEmb10Sim=useEmb10Sim  
        self.Emb10SimType=Emb10SimType
        
        #self.porter = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        if useEmbSim or useEmb10Sim:
            self.loadEmbeddingHashes()     
    
    def path_similarity(self,wordA,wordB):
        A = set(wn.synsets(wordA))
        B = set(wn.synsets(wordB))

        pathSimilarities = [0]
        
        for (sense1,sense2) in itertools.product(A,B):        
            pathSimilarities.append(wn.path_similarity(sense1,sense2) or 0)
            
        return max(pathSimilarities)
    
    def wup_similarity(self,wordA,wordB):
        A = set(wn.synsets(wordA))
        B = set(wn.synsets(wordB))

        wupSimilarities = [0]
        
        for (sense1,sense2) in itertools.product(A,B):        
            wupSimilarities.append(wn.wup_similarity(sense1,sense2) or 0)
            
        return max(wupSimilarities)

    def jaccard_similarity(self,wordA,wordB):

        A = set(wordA)
        B = set(wordB)

        return len(A & B ) / len(A | B)
    
    
    def embedding10_similarity(self,word,table,column='*',Emb='B'):      
        #print('word: {} ,table: {}, column: {}'.format(word,table,column))
        if table not in self.EmbA or column not in self.EmbA[table]:
            return False
        
        if Emb == 'A':
            sim_list = self.EmbA[table][column]
        
        elif Emb == 'B':
            sim_list = self.EmbA[table][column]   
            
            if column != '*':
                sim_list |= self.EmbB[table][column]
            else:                
                for neighbourTable in self.schemaGraph.neighbours(table):

                    if neighbourTable not in self.model:
                        continue

                    sim_list |= self.EmbB[table][neighbourTable]   
        
        elif Emb == 'C':
            
            sim_list = self.EmbA[table][column]   
            
            if column != '*':
                sim_list |= self.EmbC[table][column]
        
        else:
            sim_list=[]
        
        
        #print('sim({},{}.{}) = {}'.format(word,table,column,sim_list))        
        #return self.porter.stem(word) in sim_list
        return self.lemmatizer.lemmatize(word) in sim_list
                
    
    
    def embedding_similarity(self,wordA,wordB):
        if wordA not in self.model or wordB not in self.model:
            return 0
        return self.model.similarity(wordA,wordB)
    
    
    def word_similarity(self,word,table,column = '*'):
        sim_list=[0]
    
        if column == '*':
            schema_term = table
        else:
            schema_term = column
            
        if self.usePathSim:
            sim_list.append( self.path_similarity(schema_term,word) )
            
        if self.useWupSim:
            sim_list.append( self.wup_similarity(schema_term,word) )

        if self.useJaccardSim:
            sim_list.append( self.jaccard_similarity(schema_term,word) )

        if self.useEmbSim:
            sim_list.append( self.embedding_similarity(schema_term,word) )

        sim = max(sim_list) 
        
        if self.useEmb10Sim:
            if self.embedding10_similarity(word,table,column,self.Emb10SimType):
                if len(sim_list)==1:
                    sim=1
            else:
                sim=0     
        return sim    
    
    def __getSimilarSet(self,word, inputType = 'word'):
        if inputType == 'vector':
            sim_list = self.model.similar_by_vector(word)
        else:
            sim_list = self.model.most_similar(word) 
        
        #return  {self.porter.stem(word.lower()) for word,sim in sim_list}
        return  {self.lemmatizer.lemmatize(word.lower()) for word,sim in sim_list}
    
    def loadEmbeddingHashes(self,weight=0.5):
        
        self.EmbA = {}
        self.EmbB = {}
        self.EmbC = {}
    
        for table in self.schemaGraph.vertices():

            if table not in self.model:
                continue

            self.EmbA[table]={}
            self.EmbB[table]= {}
            self.EmbC[table]= {}
            
            self.EmbA[table]['*'] = self.__getSimilarSet(table) 

            if table in self.attributeHash:
                for column in self.attributeHash[table]:
                    if column not in self.model or column=='id':
                        continue

                    self.EmbA[table][column]=self.__getSimilarSet(column)

                    self.EmbB[table][column]=self.__getSimilarSet( (table,column) )

                    avg_vec = (self.model[table]*weight + self.model[column]*(1-weight))                   
                    self.EmbC[table][column] = self.__getSimilarSet(avg_vec, inputType = 'vector')
                
            for neighbor_table in self.schemaGraph.neighbours(table):

                if neighbor_table not in self.model:
                    continue
                
                self.EmbB[table][neighbor_table] = self.__getSimilarSet( (table,neighbor_table) )
        
```

```python
if STEP_BY_STEP:
    similarities=Similarities(wordEmbeddingsModel,
                              attributeHash,
                              getSchemaGraph(),
                              useEmb10Sim=False
                              )
```

```python
def SKMGen(Q,attributeHash,similarities,threshold=1):    
    S = set()
    
    for keyword in Q:
        for table in attributeHash:            
            for attribute in ['*']+list(attributeHash[table].keys()):
                
                if(attribute=='id'):
                    continue
                
                sim = similarities.word_similarity(keyword,table,attribute)
                
                if sim >= threshold:
                    skm = KeywordMatch(table,schema_filter={attribute:{keyword}})
                    S.add(skm)
                    
    return S
```

```python
if STEP_BY_STEP:    
    print('FINDING SCHEMA-SETS')        
    Sq = SKMGen(Q,attributeHash,similarities)
    print(len(Sq),' SCHEMA-SETS CREATED\n')
    pp(Sq)
```

## Query Matching

```python
def MinimalCover(MC, Q):
    #Input:  A subset MC (Match Candidate) to be checked as total and minimal cover
    #Output: If the match candidate is a TOTAL and MINIMAL cover
    if {keyword 
        for keyword_match in MC 
        for keyword in keyword_match.keywords()
       } != set(Q):
        return False
    
    for element in MC: 
        if {keyword 
            for keyword_match in MC 
            for keyword in keyword_match.keywords() 
            if keyword_match!=element
           } == set(Q):
            return False
    
    return True
```

```python
def MergeSchemaFilters(QM):
    table_hash={}
    for keyword_match in QM:
        joint_schema_filter,value_keyword_matches = table_hash.setdefault(keyword_match.table,({},[]))

        for attribute, keywords in keyword_match.schema_filter:
            joint_schema_filter.setdefault(attribute,set()).update(keywords)

        if len(keyword_match.value_filter) > 0:
            value_keyword_matches.append(keyword_match)
    merged_qm = []
    for table,(joint_schema_filter,value_keyword_matches) in table_hash.items():    
        if len(value_keyword_matches) > 0:
            joint_value_filter = {attribute:keywords 
                                  for attribute,keywords in value_keyword_matches.pop().value_filter}
        else:
            joint_value_filter={}

        joint_keyword_match = KeywordMatch(table,
                                           value_filter=joint_value_filter,
                                           schema_filter=joint_schema_filter)

        merged_qm.append(joint_keyword_match)
        merged_qm+=value_keyword_matches  

    return merged_qm
```

```python
def QMGen(Q,Rq, TMaxQM = 5):
    #Input:  A keyword query Q, The set of non-empty non-free tuple-sets Rq
    #Output: The set Mq of query matches for Q
    
    '''
    Query match is a set of tuple-sets that, if properly joined,
    can produce networks of tuples that fulfill the query. They
    can be thought as the leaves of a Candidate Network.
    
    '''  
    
    Mq = []
    for i in range(1,min(len(Q),TMaxQM)+1):
        for M in itertools.combinations(Rq,i):            
            if(MinimalCover(M,Q)):
                merged_qm = MergeSchemaFilters(M)
                Mq.append(merged_qm)
                   
    return Mq 
```

```python
if STEP_BY_STEP:
    print('GENERATING QUERY MATCHES')
    TMaxQM = 3
    Mq = QMGen(Q,Rq|Sq,TMaxQM=TMaxQM)
    print (len(Mq),'QUERY MATCHES CREATED\n')  
```

```python
def QMRank(Q, Mq,wordHash,attributeHash,similarities,showLog=False):
    Ranking = []  

    for M in Mq:
        #print('=====================================\n')
        valueProd = 1 
        schemaProd = 1
        score = 1
        
        thereIsSchemaTerms = False
        thereIsValueTerms = False
        
        for keyword_match in M:
            
            for table, attribute, valueWords in keyword_match.value_mappings():

                (Norm,numDistinctWords,numWords,maxFrequency) = attributeHash[table][attribute]                
                wsum = 0


                if showLog:
                    print('Norm: {}\nMaxFrequency {}\n'.format(Norm,maxFrequency))


                for term in valueWords:    

                    IAF = wordHash.getIAF(term)

                    frequency = len(wordHash.getMappings(term,table,attribute))
                    TF = (frequency/maxFrequency)
                    wsum = wsum + TF*IAF
                    if showLog:
                        print('- Term: {}\n  Frequency:{}\n  TF:{}\n  IAF:{}\n'.format(term,frequency,TF,IAF))

                    thereIsValueTerms = True

                '''
                for i in range(len(Q)-1):
                    if Q[i] in valueWords and Q[i+1] in valueWords:
                        wsum = wsum * 3
                '''        
                cos = wsum/Norm
                valueProd *= cos     
        
        
            for table, attribute, schemaWords in keyword_match.schema_mappings():
                schemasum = 0
                for term in schemaWords:
                    sim = similarities.word_similarity(term,table,attribute)
                    schemasum += sim

                    if showLog:
                        print('- Term: {}\n  Sim:{}\n'.format(term,sim))

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
                            
    return sorted(Ranking,key=lambda x: x[1]/len(x[0]),reverse=True)
                
```

```python
if STEP_BY_STEP:
    print('RANKING QUERY MATCHES')
    RankedMq = QMRank(Q,Mq,wordHash,attributeHash,similarities)   
    
    topK = 20
    numPrunedQMs = len(RankedMq)-topK
    if numPrunedQMs>0:
        print(numPrunedQMs,' QUERY MATCHES SKIPPED (due to low score)')
    else:
        numPrunedQMs=0        
        
    for (j, (M,score,valuescore,schemascore) ) in enumerate(RankedMq[:topK]):
        print(j+1,'ª QM')           

        print('Schema Score:',"%.8f" % schemascore,
            '\nValue Score: ',"%.20f" % valuescore,
            '\n|M|: ',"%02d (Não considerado para calcular o total score)" % len(M),
            '\nTotal Score: ',"%.8f" % score)
        pp(M)
        #print('\n----Details----\n')
        #QMRank(Q, [M],wordHash,attributeHash,similarities,showLog=True)

        print('----------------------------------------------------------------------\n')
```

## Class CN

```python
class CandidateNetwork(Graph):
    def add_vertex(self, vertex, default_alias=True):
        if default_alias:
            vertex = (vertex, 't{}'.format(self.__len__()+1))
        return super().add_vertex(vertex)
        
    def keyword_matches(self):
        return {keyword_match for keyword_match,alias in self.vertices()}
    
    def non_free_keyword_matches(self):
        return {keyword_match for keyword_match,alias in self.vertices() if not keyword_match.is_free()}
            
    def isSound(self):
        if len(self) < 3:
            return True
        
        #check if there is a case A->B<-C, when A.table=C.table
        for vertex,(outgoing_neighbours,incoming_neighbours) in self._Graph__graph_dict.items():
            if len(outgoing_neighbours)>=2:
                outgoing_tables = set()
                for neighbour,alias in outgoing_neighbours:
#                     print('neighbour,alias  ',neighbour,alias)
                    if neighbour.table not in outgoing_tables:
                        outgoing_tables.add(neighbour.table)
                    else:
                        return False
        
        return True
                
    def get_starting_vertex(self):
        vertex = None
        for vertex in self.vertices():
            keyword_match,alias = vertex
            if not keyword_match.is_free():
                break
        return vertex
    
    def remove_vertex(self,vertex):
        print('vertex:\n{}\n_Graph__graph_dict\n{}'.format(vertex,self._Graph__graph_dict))
        outgoing_neighbours,incoming_neighbours = self._Graph__graph_dict[vertex]
        for neighbour in incoming_neighbours:
            self._Graph__graph_dict[neighbour][0].remove(vertex)
        self._Graph__graph_dict.pop(vertex)
         
    def minimal_cover(self,QM):
        if self.non_free_keyword_matches()!=set(QM):
            return False
        
        for vertex in self.vertices():
            keyword_match,alias = vertex
            if keyword_match.is_free():
                visited = {vertex}
                start_node = next(iter( self.vertices() - visited ))
                
                for x in self.leveled_dfs_iter(start_node,visited=visited):
                    #making sure that the dfs algorithm runs until the end of iteration
                    continue
                
                if visited == self.vertices():
                    return False
        return True
    
    def unaliased_edges(self):
        for (keyword_match,alias),(neighbour_keyword_match,neighbour_alias) in self.edges():
            yield (keyword_match,neighbour_keyword_match)
    
    def __eq__(self, other):
        return hash(self)==hash(other) and isinstance(other,CandidateNetwork)
    
    #Although this is a multable object, we made the hash function since it is not supposed to change after inserted in the lsit of generated cns
    def __hash__(self):
        return hash((frozenset(Counter(self.unaliased_edges()).items()),frozenset(self.keyword_matches())))
    
    def __repr__(self):
        if len(self)==0:
            return 'EmptyCN'            
        print_string = ['\t'*level+direction+str(vertex[0])  for direction,level,vertex in self.leveled_dfs_iter()]            
        return '\n'.join(print_string)
    
    def to_json_serializable(self):
        return [{'keyword_match':keyword_match.to_json_serializable(),
            'alias':alias,
            'outgoing_neighbours':[alias for (km,alias) in outgoing_neighbours],
            'incoming_neighbours':[alias for (km,alias) in incoming_neighbours]}
            for (keyword_match,alias),(outgoing_neighbours,incoming_neighbours) in self._Graph__graph_dict.items()]
            
    def to_json(self):
        return json.dumps(self.to_json_serializable())
    
    @staticmethod
    def from_str(str_cn):

        def parser_iter(str_cn):
            re_cn = re.compile('^(\t*)([><]?)(.*)',re.MULTILINE)
            for i,(space,direction,str_km) in enumerate(re_cn.findall(str_cn)):
                yield (direction,len(space), (KeywordMatch.from_str(str_km),'t{}'.format(i+1)))

        imported_cn = CandidateNetwork()

        prev_vertex = None
        for prev_vertex,direction,vertex in imported_cn.dfs_pair_iter(root_predecessor=True,source_iter=parser_iter(str_cn)):
            imported_cn.add_vertex(vertex,default_alias=False)
            if prev_vertex is not None:
                imported_cn.add_edge(prev_vertex,vertex,edge_direction=direction)  

        return imported_cn
    
    def from_json_serializable(json_serializable_cn):
        alias_hash ={}
        edges=[]
        for vertex in json_serializable_cn:
            keyword_match = KeywordMatch.from_json_serializable(vertex['keyword_match'])
            alias_hash[vertex['alias']]=keyword_match

            for outgoing_neighbour in vertex['outgoing_neighbours']:
                edges.append( (vertex['alias'],outgoing_neighbour) )
                
        candidate_network = CandidateNetwork()
        for alias,keyword_match in alias_hash.items():
            candidate_network.add_vertex( (keyword_match,alias) , default_alias=False)
        for alias1, alias2 in edges:
            vertex1 = (alias_hash[alias1],alias1)
            vertex2 = (alias_hash[alias2],alias2)
            candidate_network.add_edge(vertex1,vertex2)
        return candidate_network
    
    def from_json(json_cn):
        return CandidateNetwork.from_json_serializable(json.loads(json_cn))
   
```

```python
cnx=CandidateNetwork.from_str('''TITLE.s(*{title})
	<CAST_INFO
		>NAME.v(name{bond,james})''')
CandidateNetwork.from_json(cnx.to_json())
```

```python
def sum_norm_attributes(directed_neighbor):
    direction,adj_table = directed_neighbor
    if adj_table not in attributeHash:
        return 0
    return sum(Norm for (Norm,numDistinctWords,numWords,maxFrequency) in attributeHash[adj_table].values())
```

```python
sorted([(sum(Norm for (Norm,numDistinctWords,numWords,maxFrequency) in attributeHash[table].values()),
  table) for table in attributeHash],reverse=True)
```

```python
def CNGraphGen(QM,G,TMax=10,showLog=False,directed_neighbor_sorting = None,topKCNsPerQM=2):  
    if showLog:
        print('================================================================================\nSINGLE CN')
        print('Tmax ',TMax)
        print('FM')
        pp(QM)
        #print('\n\n')
        #print('\n\nGts')
        #pp(Gts)
        #print('\n\n')
    
    if directed_neighbor_sorting is None:
        directed_neighbor_sorting=sum_norm_attributes
    
    CN = CandidateNetwork()
    CN.add_vertex(QM[0])
    
    if len(QM)==1:
        return {CN}
    
    returnedCNs = set()
    
    table_hash={}
    for keyword_match in QM:
        table_hash.setdefault(keyword_match.table,set()).add(keyword_match)    
    
    F = deque()
    F.append(CN)
        
    while F:        
        CN = F.popleft()
#         print('///////////////')
#         pp([x for x in CN.vertices()])
        
        
        for vertex_u in CN.vertices():
            keyword_match,alias = vertex_u
            
            
            sorted_directed_neighbors = sorted(G.directed_neighbours(keyword_match.table),
                                               reverse=True,
                                               key=directed_neighbor_sorting) 
            
            for direction,adj_table in sorted_directed_neighbors:
#                 print('CHECKING TABLE ',adj_table)
#                 print('NON-FREE KEYWORD MATCHES')
                
                if adj_table in table_hash:                    
                    for adj_keyword_match in table_hash[adj_table]:
                        if adj_keyword_match not in CN.keyword_matches():
                            new_CN = copy.deepcopy(CN)
                            vertex_v = new_CN.add_vertex(adj_keyword_match)
                            new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction)         

                            if (new_CN not in F and
                                new_CN not in returnedCNs and
                                len(new_CN)<=TMax and
                                new_CN.isSound() and
                                len(list(new_CN.leaves())) <= len(QM)):
#                                 print('Adding ',adj_keyword_match,' to current CN')
                                if new_CN.minimal_cover(QM):
                                    print('Found CN')
                                    print(new_CN)
#                                     print('GENERATED THE FIRST ONE')
                                    if len(returnedCNs)<topKCNsPerQM:
                                        returnedCNs.add(new_CN)
                                    
                                    if len(returnedCNs)==topKCNsPerQM:
                                        return returnedCNs
                                elif len(new_CN)<TMax:
#                                     print('Adding\n{}\n'.format(new_CN))
                                    F.append(new_CN)
                                    
                
                
                new_CN = copy.deepcopy(CN)
#                 print('FREE KEYWORD MATCHES')
                adj_keyword_match = KeywordMatch(adj_table)
                vertex_v = new_CN.add_vertex(adj_keyword_match)
                new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction)
                if (new_CN not in F and
                    new_CN not in returnedCNs and
                    len(new_CN)<TMax and
                    new_CN.isSound() and 
                    len(list(new_CN.leaves())) <= len(QM)):
#                     print('Adding ',adj_keyword_match,' to current CN')
#                     print('Adding\n{}\n'.format(new_CN))
                    F.append(new_CN)
                        
    return returnedCNs
```

```python
if STEP_BY_STEP:
    TMax=5
   
    (QM,score,valuescore,schemascore) = RankedMq[0]
    print('GENERATING CNs FOR QM:',QM)
    
    Cns = CNGraphGen(QM,G,TMax=TMax,topKCNsPerQM=20)
    
    for j, Cn in enumerate(Cns):
        print(j+1,'ª CN',
              '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
              '\nTotal Score: ',"%.8f" % (score/len(Cn)))
        print(Cn)
```

```python
for candidate_network in Cns:
    if execSQL(getSQLfromCN(G,candidate_network,rowslimit=1), showResults=False,):
        print(candidate_network)
```

```python
def MatchCN(attributeHash,G,RankedMq,TMax=10,maxNumCns=20,topKCNsPerQM=2,directed_neighbor_sorting=None,showLog=False):    
    UnrankedCns = []    
    generated_cns=[]
    
    for i,(QM,score,valuescore,schemascore) in enumerate(RankedMq):
        if showLog:
            print('{}ª QM:\n{}\n'.format(i+1,QM))
        Cns = CNGraphGen(QM,G,TMax=TMax,topKCNsPerQM=topKCNsPerQM,directed_neighbor_sorting=directed_neighbor_sorting)
        if showLog:
            print('Cns:')
            pp(Cns)
        if len(UnrankedCns)>=maxNumCns:
            break
    
        for Cn in Cns:
            if(Cn not in generated_cns):          
                generated_cns.append(Cn)

                #Dividindo score pelo tamanho da cn (SEGUNDA PARTE DO RANKING)                
                CnScore = score/len(Cn)

                UnrankedCns.append( (Cn,CnScore,valuescore,schemascore) )
    #Ordena CNs pelo CnScore
    RankedCns=sorted(UnrankedCns,key=lambda x: x[1],reverse=True)
    
    return RankedCns
```

```python
if STEP_BY_STEP:   
    print('GENERATING CANDIDATE NETWORKS')  
    RankedCns = MatchCN(attributeHash,G,RankedMq,
                        TMax=TMax,
                        maxNumCns=20,
                       showLog=True)
    print (len(RankedCns),'CANDIDATE NETWORKS CREATED AND RANKED\n')
    
    for (j, (Cn,score,valuescore,schemascore) ) in enumerate(RankedCns):
        print(j+1,'ª CN')
        print('Schema Score:',"%.8f" % schemascore,
              '\nValue Score: ',"%.8f" % valuescore,
              '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
              '\nTotal Score: ',"%.8f" % score)
        pp(Cn)
```

## getSQLfromCN

```python
def getSQLfromCN(G,Cn,showEvaluationFields=False,rowslimit=1000):
    
    hashtables = {} # used for disambiguation

    selected_attributes = set()
    filter_conditions = []
    disambiguation_conditions = []
    selected_tables = []

    tables__search_id = []
    relationships__search_id = []
    
    for prev_vertex,direction,vertex in Cn.dfs_pair_iter(root_predecessor=True):    
        keyword_match, alias = vertex
        for type_km,_ ,attr,keywords in keyword_match.mappings():
            selected_attributes.add('{}.{}'.format(alias,attr))
            if type_km == 'v':
                for keyword in keywords:
                    condition = 'CAST({}.{} AS VARCHAR) ILIKE \'%{}%\''.format(alias,attr,keyword.replace('\'','\'\'') )
                    filter_conditions.append(condition)

        hashtables.setdefault(keyword_match.table,[]).append(alias)
        
        if showEvaluationFields:
            tables__search_id.append('{}.__search_id'.format(alias))

        if prev_vertex is None:
            selected_tables.append('{} {}'.format(keyword_match.table,alias))
        else:
            # After the second table, it starts to use the JOIN syntax
            prev_keyword_match,prev_alias = prev_vertex
            if direction == '>':
                (prev_column,column) = G.get_edge_info(prev_keyword_match.table,keyword_match.table)
            else:
                (column,prev_column) = G.get_edge_info(keyword_match.table,prev_keyword_match.table)

            selected_tables.append('JOIN {} {} ON {}.{} = {}.{}'.format(keyword_match.table,
                                                                        alias,
                                                                        prev_alias,
                                                                        prev_column,
                                                                        alias,
                                                                        column ))
            if showEvaluationFields:
                relationships__search_id.append('({}.__search_id, {}.__search_id)'.format(alias,prev_alias))


    for table,aliases in hashtables.items():        
        for i in range(len(aliases)):
            for j in range(i+1,len(aliases)):
                disambiguation_conditions.append('{}.ctid <> {}.ctid'.format(aliases[i],aliases[j]))
        
    if len(tables__search_id)>0:
        tables__search_id = ['({}) AS Tuples'.format(', '.join(tables__search_id))]
    if len(relationships__search_id)>0:
        relationships__search_id = ['({}) AS Relationships'.format(', '.join(relationships__search_id))]

    sqlText = '\nSELECT\n\t{}\nFROM\n\t{}\nWHERE\n\t{}\nLIMIT {};'.format(
        ',\n\t'.join( tables__search_id+relationships__search_id+list(selected_attributes) ),
        '\n\t'.join(selected_tables),
        '\n\tAND '.join( disambiguation_conditions+filter_conditions),
        rowslimit)
    return sqlText
```

```python
if STEP_BY_STEP:
    (Cn,score,valuescore,schemascore)= RankedCns[0]
    print(Cn)
    print(getSQLfromCN(G,Cn,showEvaluationFields=True))
```

```python
def execSQL (SQL,dbname=None,user=None,password=None,showResults=True):
    #print('RELAVANCE OF SQL:\n')
    #print(SQL)
    
    if dbname is None:
        dbname=DBNAME
    if DBUSER is None:
        user=DBUSER
    if DBPASS is None:
        password=DBPASS
    
    from prettytable import PrettyTable

    
    with psycopg2.connect(dbname=dbname,user=user,password=password) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(SQL)

                #Results = cur.fetchall()
                #Description = cur.description

                #x.field_names = Description


                if cur.description:  
                    table = PrettyTable()
                    table.field_names = [ '{}{}'.format(i,col[0]) for i,col in enumerate(cur.description)]
                    for row in cur.fetchall():
                        table.add_row(row)
                if showResults:
                    print(table)
            except:
                print('ERRO SQL:\n',SQL)
                raise
                        
            return cur.rowcount>0
```

```python
execSQL(getSQLfromCN(G,Cn,rowslimit=1))
```

```python
def keywordSearch(wordHash,attributeHash,wordEmbeddingsModel,
                  showLog=False,
                  SimilarityThreshold=0.9,
                  querySetFileName=QUERYSETFILE,
                  topK=15,
                  topKCNsPerQM = 5,
                  TMax=10,
                  TMaxQM=3,
                  directed_neighbor_sorting = None,
                  similarities = None,
                  queryset = None
         ):
    
    if queryset is None:
        QuerySets = getQuerySets(querySetFileName)
    else:
        QuerySets=queryset
    G = getSchemaGraph()    
    
    
    returnedCn = {}
    
    if similarities is None:
        similarities=Similarities(wordEmbeddingsModel,attributeHash,G)
    
    
    for (i,Q) in enumerate(QuerySets):
       
        print(i+1,'ª QUERY ',Q,'\n')
        
        print('FINDING TUPLE-SETS')
        Rq = VKMGen(Q, wordHash)
        print(len(Rq),'TUPLE-SETS CREATED\n')
        
        if showLog:
            pp(Rq)
        
        print('FINDING SCHEMA-SETS')        
        Sq = SKMGen(Q,attributeHash,similarities)
        print(len(Sq),' SCHEMA-SETS CREATED\n')
        
        if showLog:
            pp(Sq)

        
        print('GENERATING QUERY MATCHES')
        Mq = QMGen(Q,Rq|Sq,TMaxQM=TMaxQM)
        print (len(Mq),'QUERY MATCHES CREATED\n')
        
        print('RANKING QUERY MATCHES')
        RankedMq = QMRank(Q,Mq,wordHash,attributeHash,similarities)   
        
        numPrunedQMs = len(RankedMq)-topK
        
        if numPrunedQMs>0:
            print(numPrunedQMs,' QUERY MATCHES SKIPPED (due to low score)')
        else:
            numPrunedQMs=0        
        
        if showLog:
            for (j, (QM,score,valuescore,schemascore) ) in enumerate(RankedMq[:topK]):
                print(i+1,'ª Q ',j+1,'ª QM')           
                
                print('Schema Score:',"%.8f" % schemascore,
                      '\nValue Score: ',"%.8f" % valuescore,
                      '\n|QM|: ',"%02d (Não considerado para calcular o total score)" % len(QM),
                      '\nTotal Score: ',"%.8f" % score)
                pp(QM)
                
                print('----------------------------------------------------------------------\n')
        
        
        print('GENERATING CANDIDATE NETWORKS')     
        RankedCns = MatchCN(attributeHash,G,RankedMq,
                            TMax=TMax,
                            maxNumCns=topK,
                            topKCNsPerQM=topKCNsPerQM,
                            directed_neighbor_sorting=directed_neighbor_sorting)
        
        print (len(RankedCns),'CANDIDATE NETWORKS CREATED RANKED\n')
        
        if showLog:
            for (j, (Cn,score,valuescore,schemascore) ) in enumerate(RankedCns):
                print(j+1,'ª CN')
                print('Schema Score:',"%.8f" % schemascore,
                      '\nValue Score: ',"%.8f" % valuescore,
                      '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
                      '\nTotal Score: ',"%.8f" % score)
                pp(Cn)
        
        print('\n==========================================================================\
==========================================================================\
==========================================================================\
==========================================================================\
==========================================================================\
==========================================================================\n')
        returnedCn[Q]=RankedCns
    return returnedCn
```

```python
results = keywordSearch(wordHash,attributeHash,wordEmbeddingsModel,
                       showLog=True,
                       querySetFileName=QUERYSETFILE,
                       topK=10,
                       topKCNsPerQM=2,
                       TMax=5,
                       TMaxQM=3,
                       similarities=Similarities(wordEmbeddingsModel,attributeHash,G,useEmb10Sim=False)
                      )
```

```python
results2 = keywordSearch(wordHash,attributeHash,wordEmbeddingsModel,
                       showLog=True,
                       topK=10,
                       topKCNsPerQM=7,
                       TMax=5,
                       TMaxQM=3,
                        queryset=getQuerySets()[35:45],
                       similarities=Similarities(wordEmbeddingsModel,attributeHash,G,useEmb10Sim=False)
                      )
```

```python
def getGoldenCandidateNetworks(path=None):
    if path is None:
        path = GOLDEN_CANDIDATE_NETWORKS_PATH
    golden_cns = OrderedDict()
    for filename in sorted(glob.glob('{}/*.txt'.format(GOLDEN_CANDIDATE_NETWORKS_PATH.rstrip('/')))):
        with open(filename) as f:
            json_serializable = json.load(f)
            golden_cns[tuple(json_serializable['query'])] = \
                CandidateNetwork.from_json_serializable(json_serializable['candidate_network'])
    return golden_cns

```

```python
if STEP_BY_STEP:
    golden_cns=getGoldenCandidateNetworks()
    
    for i,Q in enumerate(getQuerySets(QUERYSETFILE)):
        if Q not in golden_cns:
            continue
    
        print('{} Q: {}\nCN: {}\n'.format(i+1,Q,golden_cns[Q]))
```

```python
golden_cns
```

```python
[(i+1,Q) for i,Q in enumerate(getQuerySets(QUERYSETFILE)) if Q not in golden_cns]
```

```python
def setGoldenCandidateNetworks(result,golden_cns=None):
    from IPython.display import clear_output
    if golden_cns is None:
        golden_cns = OrderedDict()
    for i,Q in enumerate(getQuerySets(QUERYSETFILE)):
        if golden_cns.setdefault(Q,None) is not None:
            continue
        
        answer = None
            
        
        if Q not in result or len(result[Q]) == 0:
            answer = input('The system did not generate any Candidate Network for the Query Q given.\nQ:\n{}\nWould you like to write the correct relevant CN for it?(type y if yes)\n'.format(Q))
            clear_output()
            if answer == 'y':
                candidate_network = CandidateNetwork.from_str(input("Write the golden Candidate Network:"))
                golden_cns[Q]=candidate_network
            continue
        
        for candidate_network,_,_,_ in result[Q]:
            answer = input('Is the following Candidate Network\nrelevant for the Query Q given?(type y if yes)\n{}a Q:\n{}\nCN:\n{}\n'.format(i+1,Q,candidate_network))
            clear_output()
            if answer == 'y':
                golden_cns[Q]=candidate_network
                break
            elif answer == 'custom':
                candidate_network = CandidateNetwork.from_str(input("Write the golden Candidate Network:"))
                golden_cns[Q]=candidate_network
            elif answer == 'skip':
                break
            elif answer == 'stop':
                return golden_cns
    return golden_cns
```

```python
gs = setGoldenCandidateNetworks(results,golden_cns)
```

```python
def generateGoldenCNFiles(golden_standards,path=None):  
    if path is None:
        path = GOLDEN_CANDIDATE_NETWORKS_PATH
        
    for i,Q in enumerate(getQuerySets(QUERYSETFILE)):
        
        filename = "{}/{:0>3d}.txt".format(GOLDEN_CANDIDATE_NETWORKS_PATH.rstrip('/'),i+1) 
        
        if Q not in golden_standards:
                print("File {} not created because there is\n no golden standard set for the query\n {}".format(filename,Q))
                continue
        
        with open(filename,mode='w') as f:
            json_serializable = {'candidate_network':golden_standards[Q].to_json_serializable(),
                                 'query':Q,} 
            f.write(json.dumps(json_serializable,indent=4))
```

```python
#generateGoldenCNFiles(golden_cns)
```

```python
def get_relevant_positions(results,golden_stantards):
    relevant_positions = OrderedDict()
    for Q,golden_standard in golden_stantards.items():
        idx = 0
        non_empty_idx = 0
        found = False
        for (candidate_network,_,_,_) in results[Q]:
            if execSQL(getSQLfromCN(G,candidate_network,rowslimit=1), showResults=False,):
                non_empty_idx+=1
            idx+=1
            
            if candidate_network==golden_standard:
                found=True
                break
        
        if not found:
            idx = -1
            non_empty_idx = -1
            
        relevant_positions[Q]=(idx,non_empty_idx)
    return relevant_positions
```

```python
e = get_relevant_positions(results,golden_cns)
```

```python
golden_cns[('slovakia', 'hungary')]
```

```python
[(i,Q) for i,Q in enumerate(e) if e[Q] == (-1,-1)]
```

```python

```

```python
for Q in e:
    if e[Q] == (-1,-1):
        print('\nQ:\n{}\nGS:\n{}\n'.format(Q,golden_cns[Q]))
        for i,x in enumerate(results[Q]):
            cn= x[0]
            print('{}ª CN:\n{}'.format(i+1,cn))
```

```python
def mrr(position_list):
    return sum(1/p for p in position_list if p != -1)/len(position_list)

def precision_at(position_list,threshold = 3):
    return len([p for p in position_list if p != -1 and p<=threshold])/len(position_list)
           
```

```python
e
```

```python

```

```python
print(getSQLfromCN(G,golden_cns[('slovakia', 'german')],showEvaluationFields=True))
```

```python
position_list = [idx for idx,non_empty_idx in e.values()]
non_empty_position_list = [non_empty_idx for idx,non_empty_idx in e.values()]
```

```python
def metrics(position_list):
    result = OrderedDict()
    result['mrr']=mrr(position_list),
    result['p@1']=precision_at(position_list,threshold=1),
    result['p@2']=precision_at(position_list,threshold=2),
    result['p@3']=precision_at(position_list,threshold=3),
    result['max']=max(position_list),
    return result
```

```python
metrics(position_list)
```

```python
metrics(non_empty_position_list)
```

```python
def evaluation(results,golden_cns):
    relevant_positions = get_relevant_positions(results,golden_cns)
    
    position_list = [idx for idx,non_empty_idx in relevant_positions.values()]
    non_empty_position_list = [non_empty_idx for idx,non_empty_idx in relevant_positions.values()]
    
    return metrics(position_list), metrics(non_empty_position_list)
```

```python
evaluation(results,golden_cns)
```
