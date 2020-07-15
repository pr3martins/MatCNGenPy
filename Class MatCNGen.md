# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: md,ipynb,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: LATHE
    language: python
    name: lathe
---

```python
from pprint import pprint as pp
import gc  #garbage collector usado no create_inverted_index

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

from collections import Counter  #used to check whether two CandidateNetworks are equal

import glob  #to list filenames in directories
import re  # used to parse string to keyword match class
import json  #used to parse class to json

from collections import OrderedDict  #To print dict results according to ordered queryset

from prettytable import PrettyTable  # To print sql results

import numpy
from timeit import default_timer as timer
import datetime

import shelve

import matplotlib.pyplot as plt
import numpy as np
```

## Class DatasetConfiguration

```python
class DatasetConfiguration:
    __slots__ = 'name','value_index','schema_index','conn_string'   
    def __init__(self, name,
                 conn_string):
        self.name = name
        self.conn_string = conn_string
        
    def __repr__(self):
        return 'DatasetConfiguration({})'.format(self.name)
    
    def __str__(self):
        return repr(self)
    
    def to_tuple(self):
        return (self.name,
                self.conn_string,)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self,other):
        return (isinstance(other,DatasetConfiguration) and
                self.to_tuple() == other.to_tuple())     
    def indexes(self):
        return self.value_index,self.schema_index
```

```python
try:
    mondial_coffman_dsconfig
except NameError:
    print('Generating condiguration')
    mondial_coffman_dsconfig =  DatasetConfiguration('mondial_coffman','dbname=mondial_coffman user=paulo password=')
```

```python
try:
    imdb_coffman_dsconfig
except NameError:
    print('Generating condiguration')
    imdb_coffman_dsconfig =  DatasetConfiguration('imdb_coffman','dbname=imdb_coffman_subset user=paulo password=')
```

```python
# try:
#     wikipedia_coffman_dsconfig
# except NameError:
#     print('Generating condiguration')
#     wikipedia_coffman_dsconfig =  DatasetConfiguration('wikipedia_coffman','dbname=wikipedia_coffman_subset user=paulo password=')
```

```python
# try:
#     imdb_ijs_dsconfig
# except NameError:
#     print('Generating condiguration')
#     imdb_ijs_dsconfig =  DatasetConfiguration('imdb_ijs','dbname=imdb_ijs user=paulo password=')
```

```python
try:
    mas_dsconfig
except NameError:
    print('Generating condiguration')
    mas_dsconfig =  DatasetConfiguration('mas','dbname=mas user=paulo password=')
```

```python
def valid_schema_element(text): 
    skip_list = ['id','index','code','nr', '_num','doi', 'homepage', 'photo', 'rank', 'relation', 'exist', 'citing', 'cited']
    for word in skip_list:
        if word in text:
            return False
    return True    
```

```python
def tokenize_string(text):     
    return [word.strip(string.punctuation)
            for word in text.lower().split() 
            if word not in stopwords.words('english') or word == 'will']
    return [word
            for word in text.translate({ord(c):' ' for c in string.punctuation if c!='_'}).lower().split() 
            if word not in stopwords.words('english') or word == 'will']
```

## Class QuerysetConfiguration

```python
class QuerysetConfiguration:
    def __init__(self,
                 name,
                 queryset_filename,
                 golden_query_matches_directory,
                 golden_candidate_networks_directory,
                 dsconfig
                 ):
        
        self.name = name
        self.queryset_filename = queryset_filename
        self.golden_query_matches_directory = golden_query_matches_directory
        self.golden_candidate_networks_directory = golden_candidate_networks_directory
        self.dsconfig = dsconfig
        self.params = {}
        self.value_index = None
        self.schema_index = None
        
    def __repr__(self):
        return 'QuerysetConfiguration({})'.format(self.name)
    
    def __str__(self):
        return repr(self)
    
    def to_tuple(self):
        return (self.name,
                    self.queryset_filename,
                    self.golden_query_matches_directory,
                    self.golden_candidate_networks_directory,
                    self.dsconfig)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self,other):
        return (isinstance(other,QuerysetConfiguration) and
                self.to_tuple() == other.to_tuple()
               )        
    
    def copy(self,copy_name):
        return QuerysetConfiguration(copy_name,
                                     
                                     self.queryset_filename,
                                     self.golden_query_matches_directory,
                                     self.golden_candidate_networks_directory,
                                     self.dsconfig)
```

```python
mondial_coffman_original_qsconfig = QuerysetConfiguration('MONDIAL',
                                                          'querysets/queryset_mondial_coffman_simple_graph.txt',
                                                          'golden_query_matches/mondial_coffman',
                                                          'golden_candidate_networks/mondial_coffman', 
                                                          mondial_coffman_dsconfig
                                                         )
```

```python
mondial_coffman_ci_qsconfig = QuerysetConfiguration('MONDIAL-DI',
                                                          'querysets/queryset_mondial_coffman_simple_graph_clear_intents.txt',
                                                          'golden_query_matches/mondial_coffman_simple_graph_clear_intents',
                                                          'golden_candidate_networks/mondial_coffman_simple_graph_clear_intents', 
                                                          mondial_coffman_dsconfig
                                                         )
```

```python
imdb_coffman_original_qsconfig = QuerysetConfiguration('IMDB',
                                                       'querysets/queryset_imdb_coffman_original.txt',
                                                       'golden_query_matches/imdb_coffman_original',
                                                       'golden_candidate_networks/imdb_coffman_original',
                                                       imdb_coffman_dsconfig
                                                      )
```

```python
imdb_coffman_ci_qsconfig = QuerysetConfiguration('IMDB-DI',
                                                       'querysets/queryset_imdb_coffman_clear_intents.txt',
                                                       'golden_query_matches/imdb_coffman_clear_intents',
                                                       'golden_candidate_networks/imdb_coffman_clear_intents',
                                                       imdb_coffman_dsconfig
                                                      )
```

```python
# imdb_ijs_martins_qsconfig = QuerysetConfiguration('imdb_ijs_martins',
#                                                   'querysets/queryset_imdb_martins_qualis.txt',
#                                                   'golden_query_matches/imdb_ijs_martins',    
#                                                   'golden_candidate_networks/imdb_ijs_martins',
#                                                   imdb_ijs_dsconfig
#                                                  )
```

```python
mas_jagadish_qsconfig = QuerysetConfiguration('mas',
                                                  'querysets/queryset_mas_aspas_jagadish.txt',
                                                  '',    
                                                  '',
                                                  mas_dsconfig
                                                 )
```

```python
CREATE_PERSIST_INDEX =  False
PREPROCESSING = False
STEP_BY_STEP = True
CUSTOM_QUERY = None
LOAD_EMBEDDINGS = True

# DSCONFIGS_TO_PREPROCESS = [imdb_coffman_dsconfig,
#                            mondial_coffman_dsconfig]
# QSCONFIGS_TO_PROCESS = [imdb_coffman_original_qsconfig,
#                         imdb_coffman_ci_qsconfig,
#                         mondial_coffman_original_qsconfig,
#                         mondial_coffman_ci_qsconfig]

# DSCONFIGS_TO_PREPROCESS = [mas_dsconfig]
# QSCONFIGS_TO_PROCESS = [mas_jagadish_qsconfig]

DSCONFIGS_TO_PREPROCESS = [mas_dsconfig,
                           imdb_coffman_dsconfig,
                           mondial_coffman_dsconfig,]

QSCONFIGS_TO_PROCESS = [mas_jagadish_qsconfig,
                        imdb_coffman_original_qsconfig,
                        imdb_coffman_ci_qsconfig,
                        mondial_coffman_original_qsconfig,
                        mondial_coffman_ci_qsconfig]


DEFAULT_QSCONFIG = QSCONFIGS_TO_PROCESS[0]
DEFAULT_DSCONFIG =  DEFAULT_QSCONFIG.dsconfig
```

## get_query_sets

```python
# def get_query_sets(queryset_filename):
        
#     QuerySet = []
#     with open(queryset_filename,
#               encoding='utf-8-sig') as f:
#         for line in f.readlines():
            
#             #The line bellow Remove words not in OLIVEIRA experiments
#             #Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
#             Q = tuple(tokenize_string(line))
            
#             QuerySet.append(Q)
#     return QuerySet
```

```python
def get_query_sets(queryset_filename):
    with open(queryset_filename,
              encoding='utf-8-sig') as f:
        keyword_queries =  f.read().split('\n')
    if '' in keyword_queries:
        keyword_queries.remove('')
    return keyword_queries
```

```python
if STEP_BY_STEP:
    QuerySets = get_query_sets(DEFAULT_QSCONFIG.queryset_filename)
    if CUSTOM_QUERY is None:
        keyword_query = QuerySets[0]
    else:
        keyword_query = CUSTOM_QUERY
    print(keyword_query)
```

```python
# get_query_sets(DEFAULT_QSCONFIG.queryset_filename)
```

```python
Q = tuple(tokenize_string(keyword_query))
Q
```

# Preprocessing stage

```python
def load_embeddings(embedding_filename = 'word_embeddings/word2vec/GoogleNews-vectors-negative300.bin'):
    
    return KeyedVectors.load_word2vec_format(embedding_filename,
                                             binary=True, limit=500000)
```

```python
try:
    word_embeddings_model
except NameError:
    word_embeddings_model = None
    
if LOAD_EMBEDDINGS:
    if word_embeddings_model is None:
        word_embeddings_model=load_embeddings()
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


## Class BabelItemsIter

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
    
    babel = {}
        
    def __getidfromkey__(self,key):
        return BabelHash.babel[key]
    
    def __getkeyfromid__(self,key_id):
        key = BabelHash.babel[key_id]
        return key
    
    def __getitem__(self,key):
        key_id = self.__getidfromkey__(key)
        return dict.__getitem__(self,key_id)
    
    def __setitem__(self,key,value):    
        try:
            key_id = BabelHash.babel[key]
        except KeyError:
            key_id = len(BabelHash.babel)+1
                     
            BabelHash.babel[key] = key_id
            BabelHash.babel[key_id] = key
        
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
    
    def print_babel(self):
        print(BabelHash.babel)
```

## Class ValueIndex

```python
class ValueIndex(dict):      
        
    def __init__(self): 
        dict.__init__(self)
    
    def add_mapping(self,word,table,attribute,ctid):
        
        default_IAF = 0
        
        self.setdefault( word, (default_IAF, BabelHash() ) )                    
        self[word].setdefault(table , BabelHash() )       
        self[word][table].setdefault( attribute , [] ).append(ctid)        
        
    def get_mappings(self,word,table,attribute):
        return self[word][table][attribute]
    
    def get_IAF(self,key):
        return dict.__getitem__(self,key)[0]
    
    def set_IAF(self,key,IAF):

        old_IAF,old_value = dict.__getitem__(self,key)
        
        dict.__setitem__(self, key,  (IAF,old_value)  )
    
    def return_full_item(self,word):
        return dict.__getitem__(self,word)
    
    def __getitem__(self,word):
        return dict.__getitem__(self,word)[1]
    
    def __setitem__(self,word,value): 
        if dict.__contains__(self,word):       
            old_IAF,old_value = dict.__getitem__(self,word) 
            dict.__setitem__(self, word,  (old_IAF,value)  )
        else:
            dict.__setitem__(self, word,value)
```

```python
# babelx = BabelHash.babel
```

```python
# BabelHash.babel = {}
```

```python
# BabelHash.babel = babelx
```

```python
# with shelve.open('mas.shelve') as db:
#     for key,value in y.items():
#         db[key]=value
```

```python
# with shelve.open('mas.shelve') as db:
#     print(repr(db['thomas'][1]['name']))
#     a = db['thomas'][1]
```

## Class DatabaseIter

```python
from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
```

```python
# i=[0]

# dh = display(i,display_id=True)

# def print_row(i,display):
#     display.update(i[0])
        
# rt = RepeatedTimer(1, print_row,i,dh) 

# for j in range(50):
#     sleep(1)
#     i[0]+=1
# rt.stop()
```

```python
class DatabaseIter:
    def __init__(self,conn_string,embedding_model):
        self.conn_string = conn_string
        self.embedding_model=embedding_model

    def __iter__(self):
        with psycopg2.connect(self.conn_string) as conn:
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

                    indexable_columns = [col for col in columns if valid_schema_element(col)]

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
                    
                    i=[0]
                    
                    dh = display('',display_id=True)
                    
                    def print_row(i,display):
                        display.update('{}/{}'.format(i[0],cur.rowcount))
                        
                    rt = RepeatedTimer(10, print_row,i,dh) 
                                       
                    for row in cur.fetchall(): 
                        ctid = row[0]
                        for col in range(1,len(row)):
                            column = cur.description[col][0]
                            for word in tokenize_string( str(row[col]) ):
                                yield table,ctid,column, word                        
                        
                        i[0]+=1
                        #if i%100000==1:
                            #print('*',end='')
                    
                    rt.stop()
                    dh.update('{}/{}'.format(i[0],cur.rowcount))
```

## Create Inverted Index

```python
def create_inverted_indexes(dsconfig,embedding_model,show_log=True):
    
    #Output: value_index (Term Index) with this structure below
    #map['word'] = [ 'table': ( {column} , ['ctid'] ) ]

    '''
    The Term Index is built in a preprocessing step that scans only
    once all the relations over which the queries will be issued.
    '''
    
    dsconfig.value_index = ValueIndex()
    dsconfig.schema_index = {}
    
    previous_table = None
    
    for table,ctid,column,word in DatabaseIter(dsconfig.conn_string,embedding_model):        
        dsconfig.value_index.add_mapping(word,table,column,ctid)
                
        dsconfig.schema_index.setdefault(table,{}).setdefault(column,{}).setdefault(word,1)
        dsconfig.schema_index[table][column][word]+=1
        
    for table in dsconfig.schema_index:
        for column in dsconfig.schema_index[table]:
            
            max_frequency = num_distinct_words = num_words = 0            
            for word, frequency in dsconfig.schema_index[table][column].items():
                
                num_distinct_words += 1
                
                num_words += frequency
                
                if frequency > max_frequency:
                    max_frequency = frequency
            
            norm = 0
            dsconfig.schema_index[table][column] = (norm,num_distinct_words,num_words,max_frequency)

    print ('\nINVERTED INDEX CREATED')
    gc.collect()
```

```python
if STEP_BY_STEP and PREPROCESSING:
    create_inverted_indexes(DEFAULT_DSCONFIG,word_embeddings_model)
```

## Processing IAF

```python
def process_iaf(dsconfig):
    
    total_attributes = sum([len(attribute) for attribute in dsconfig.schema_index.values()])
    
    for (term, values) in dsconfig.value_index.items():
        attributes_with_this_term = sum([len(attribute) for attribute in dsconfig.value_index[term].values()])
        IAF = log1p(total_attributes/attributes_with_this_term)
        dsconfig.value_index.set_IAF(term,IAF)        
        
    print('IAF PROCESSED')
```

```python
if STEP_BY_STEP and PREPROCESSING:
    process_iaf(DEFAULT_DSCONFIG)
```

## Processing Attribute Norms

```python
def process_norms_of_attributes(dsconfig):    
    for word in dsconfig.value_index:
        for table in dsconfig.value_index[word]:
            for column, ctids in dsconfig.value_index[word][table].items():
                   
                (prev_norm,num_distinct_words,num_words,max_frequency) = dsconfig.schema_index[table][column]

                IAF = dsconfig.value_index.get_IAF(word)

                frequency = len(ctids)
                
                TF = frequency/max_frequency
                
                Norm = prev_norm + (TF*IAF)

                dsconfig.schema_index[table][column]=(Norm,num_distinct_words,num_words,max_frequency)
                
    print ('NORMS OF ATTRIBUTES PROCESSED')
```

```python
if STEP_BY_STEP and PREPROCESSING:
    process_norms_of_attributes(DEFAULT_DSCONFIG)
```

## pre_processing

```python
def pre_processing(configs_to_preprocess,word_embeddings_model): 
    if word_embeddings_model is None:
        word_embeddings_model=load_embeddings()
    
    for dsconfig in configs_to_preprocess:
        print('-'*80)
        print('ITERATING THROUGH DATABASE {}'.format(dsconfig))
        
        if STEP_BY_STEP and dsconfig == DEFAULT_DSCONFIG:
            print('\tDEFAULT DSCONFIG ALREADY PREPROCESSED')
            continue
        
        create_inverted_indexes(dsconfig,word_embeddings_model)
        process_iaf(dsconfig)
        process_norms_of_attributes(dsconfig)
    
    print('PRE-PROCESSING STAGE FINISHED')
    return configs_to_preprocess,word_embeddings_model
```

```python
if PREPROCESSING:
    pre_processed_configs,word_embeddings_model=pre_processing(DSCONFIGS_TO_PREPROCESS,
                                                              word_embeddings_model)   
```

```python
def persist_indexes(dsconfig):
    with shelve.open('persistent_indexes/{}_value_index.shelve'.format(dsconfig.name)) as storage:
        for key,value in dsconfig.value_index.items():
            storage[key]=value
            
    with shelve.open('persistent_indexes/{}_schema_index.shelve'.format(dsconfig.name)) as storage:
        for key,value in dsconfig.schema_index.items():
            storage[key]=value
```

```python
if CREATE_PERSIST_INDEX:
    for dsconfig in DSCONFIGS_TO_PREPROCESS:
        persist_indexes(dsconfig)
```

```python
def load_value_index(dsconfig,**kwargs):   
   
    value_index = ValueIndex()    
    with shelve.open('persistent_indexes/{}_value_index.shelve'.format(dsconfig.name),flag='r') as storage:    

        for keyword in kwargs.get('keywords',storage.keys()):
            try:            
                value_index[keyword]=storage[keyword]
            except KeyError:
                continue
                
    schema_index = {}           
    with shelve.open('persistent_indexes/{}_schema_index.shelve'.format(dsconfig.name),flag='r') as storage:
        for key,value in storage.items():
            schema_index[key]=value  
            
    dsconfig.value_index = value_index
    dsconfig.schema_index = schema_index
```

```python
for dsconfig in DSCONFIGS_TO_PREPROCESS:
    
    keywords_to_load = set([keyword
                        for qsconfig in QSCONFIGS_TO_PROCESS
                        for keyword_query in get_query_sets(qsconfig.queryset_filename)
                        for keyword in tokenize_string(keyword_query)
                        if qsconfig.dsconfig ==  dsconfig
                       ])
    
    load_value_index(dsconfig,keywords=keywords_to_load)
```

```python
# queryset_value_index = ValueIndex()
# with shelve.open('persistent_indexes/mas_value_index.shelve',flag='r') as complete_value_index:
#     keywords = set([keyword for keyword_query in get_query_sets(DEFAULT_QSCONFIG.queryset_filename) for keyword in tokenize_string(keyword_query)])
#     for keyword in keywords:
#         queryset_value_index[keyword]=complete_value_index[keyword]
# DEFAULT_DSCONFIG.value_index = queryset_value_index
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

## get_schema_graph

```python
def get_schema_graph(conn_string = None, fk_constraints = None):
    
    #Output: A Schema Graph G  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }
    
    G = Graph(has_edge_info=True)
    
    if fk_constraints is None:    
        with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cur:
                    sql = '''
                        SELECT conname AS constraint_name,
                           conrelid::regclass AS table_name,
                           ta.attname AS column_name,
                           confrelid::regclass AS foreign_table_name,
                           fa.attname AS foreign_column_name
                        FROM (
                            SELECT conname, conrelid, confrelid,
                                  unnest(conkey) AS conkey, unnest(confkey) AS confkey
                             FROM pg_constraint
                             WHERE contype = 'f'
                        ) sub
                        JOIN pg_attribute AS ta ON ta.attrelid = conrelid AND ta.attnum = conkey
                        JOIN pg_attribute AS fa ON fa.attrelid = confrelid AND fa.attnum = confkey
                        ORDER BY 1,2,4;
                    '''
                    cur.execute(sql)
                    fk_constraints = cur.fetchall()
                
    edges_hash = {}

    for (constraint,table,column,foreign_table,foreign_column) in fk_constraints:
        edges_hash.setdefault((table,foreign_table),{})
        edges_hash[(table,foreign_table)].setdefault(constraint,[]).append((column,foreign_column))

    for (table,foreign_table) in edges_hash:
        G.add_vertex(table)
        G.add_vertex(foreign_table)
        G.add_edge(table,foreign_table, edges_hash[(table,foreign_table)] )

    print ('SCHEMA CREATED')          
    return G
```

```python
mas_fk_constraints = [
        ('fk_author_organization', 'author', 'aid', 'organization', 'oid'),
        ('fk_publication_conference', 'publication', 'cid', 'conference', 'cid'),
        ('fk_publication_journal', 'publication', 'jid', 'journal', 'jid'),
        ('fk_domain_author_author', 'domain_author', 'aid', 'author', 'aid'),
        ('fk_domain_author_domain', 'domain_author', 'did', 'domain', 'did'),
        ('fk_domain_conference_conference','domain_conference','cid','conference','cid'),
        ('fk_domain_conference_domain', 'domain_conference', 'did', 'domain', 'did'),
        ('fk_domain_journal_journal', 'domain_journal', 'jid', 'journal', 'jid'),
        ('fk_domain_journal_domain', 'domain_journal', 'did', 'domain', 'did'),
        ('fk_domain_keyword_domain', 'domain_keyword', 'did', 'domain', 'did'),
        ('fk_domain_keyword_keyword', 'domain_keyword', 'kid', 'keyword', 'kid'),
        ('fk_publication_keyword_publication','publication_keyword','pid','publication','pid'),
        ('fk_publication_keyword_keyword','publication_keyword','kid','keyword','kid'),
        ('fk_writes_author', 'writes', 'aid', 'author', 'aid'),
        ('fk_writes_publication', 'writes', 'pid', 'publication', 'pid')
    ]
```

```python
if STEP_BY_STEP:
    
    fk_constraints = None
    if DEFAULT_DSCONFIG == mas_dsconfig:
        fk_constraints = mas_fk_constraints
    
    G = get_schema_graph(DEFAULT_DSCONFIG.conn_string,fk_constraints)  
    print(G,end='\n\n')
    for direction,level,vertex in G.leveled_dfs_iter():
        print(level*'\t',direction,vertex)
    #print([x for x in G.dfs_pair_iter(root_predecessor=True)])
```

# Processing Stage


## Keyword Matching


### Class KeywordMatch

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

### Value Filtering


#### VKMGen

```python
def VKMGen(Q,value_index,**kwargs):
    ignored_tables = kwargs.get('ignored_tables',[])
    ignored_attributes = kwargs.get('ignored_attributes',[])
    
    #Input:  A keyword query Q=[k1, k2, . . . , km]
    #Output: Set of non-free and non-empty tuple-sets Rq

    '''
    The tuple-set Rki contains the tuples of Ri that contain all
    terms of K and no other keywords from Q
    '''
    
    #Part 1: Find sets of tuples containing each keyword
    P = {}
    for keyword in Q:
        
        if keyword not in value_index:
            continue
        
        for table in value_index[keyword]:
            if table in ignored_tables:
                continue
                
            for (attribute,ctids) in value_index[keyword][table].items():
                
                if (table,attribute) in ignored_attributes:
                    continue
                
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
            
            joint_tuples = P[vkm_i] & P[vkm_j]
            
            if len(joint_tuples)>0:
                                
                joint_predicates = {}
                
                for attribute, keywords in vkm_i.value_filter:
                    joint_predicates.setdefault(attribute,set()).update(keywords)
                
                for attribute, keywords in vkm_j.value_filter:
                    joint_predicates.setdefault(attribute,set()).update(keywords)
                
                vkm_ij = KeywordMatch(vkm_i.table,value_filter=joint_predicates)
                P[vkm_ij] = joint_tuples
                                
                P[vkm_i].difference_update(joint_tuples)
                if len(P[vkm_i])==0:
                    del P[vkm_i]
                
                P[vkm_j].difference_update(joint_tuples)
                if len(P[vkm_j])==0:
                    del P[vkm_j]                

                return TSInterMartins(P)
    return
```

```python
if STEP_BY_STEP:
    print('FINDING TUPLE-SETS')
    Rq = VKMGen(Q,DEFAULT_DSCONFIG.value_index)
    print(len(Rq),'TUPLE-SETS CREATED\n')
    pp(Rq)
```

### Schema Filtering


#### Class Similarities

```python
class Similarities:    
    def __init__(self, model, schema_index,schema_graph,**kwargs):
        self.use_path_sim=kwargs.get('use_path_sim',True)
        self.use_wup_sim=kwargs.get('use_wup_sim',True)
        self.use_jaccard_sim=kwargs.get('use_jaccard_sim',True)
        self.use_emb_sim=kwargs.get('use_emb_sim',False)
        self.use_emb10_sim=kwargs.get('use_emb10_sim',False)  
        self.emb10_sim_type=kwargs.get('emb10_sim_type','B')
        
        self.model = model
        self.schema_index = schema_index
        self.schema_graph = schema_graph

        
        #self.porter = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        if self.use_emb10_sim:
            self.load_embedding_hashes()     
    
    def path_similarity(self,word_a,word_b):
        A = set(wn.synsets(word_a))
        B = set(wn.synsets(word_b))

        path_similarities = [0]
        
        for (sense1,sense2) in itertools.product(A,B):        
            path_similarities.append(wn.path_similarity(sense1,sense2) or 0)
            
        return max(path_similarities)
    
    def wup_similarity(self,word_a,word_b):
        A = set(wn.synsets(word_a))
        B = set(wn.synsets(word_b))

        wup_similarities = [0]
        
        for (sense1,sense2) in itertools.product(A,B):        
            wup_similarities.append(wn.wup_similarity(sense1,sense2) or 0)
            
        return max(wup_similarities)

    def jaccard_similarity(self,word_a,word_b):

        A = set(word_a)
        B = set(word_b)

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
                for neighbour_table in self.schema_graph.neighbours(table):

                    if neighbour_table not in self.model:
                        continue

                    sim_list |= self.EmbB[table][neighbour_table]   
        
        elif Emb == 'C':
            
            sim_list = self.EmbA[table][column]   
            
            if column != '*':
                sim_list |= self.EmbC[table][column]
        
        else:
            sim_list=[]
        
        
        #print('sim({},{}.{}) = {}'.format(word,table,column,sim_list))        
        #return self.porter.stem(word) in sim_list
        return self.lemmatizer.lemmatize(word) in sim_list
                
    
    
    def embedding_similarity(self,word_a,word_b):
        if word_a not in self.model or word_b not in self.model:
            return 0
        return self.model.similarity(word_a,word_b)
    
    
    def word_similarity(self,word,table,column = '*'):
        sim_list=[0]
    
        if column == '*':
            schema_term = table
        else:
            schema_term = column
            
        if self.use_path_sim:
            sim_list.append( self.path_similarity(schema_term,word) )
            
        if self.use_wup_sim:
            sim_list.append( self.wup_similarity(schema_term,word) )

        if self.use_jaccard_sim:
            sim_list.append( self.jaccard_similarity(schema_term,word) )

        if self.use_emb_sim:
            sim_list.append( self.embedding_similarity(schema_term,word) )

        sim = max(sim_list) 
        
        if self.use_emb10_sim:
            if self.embedding10_similarity(word,table,column,self.emb10_sim_type):
                if len(sim_list)==1:
                    sim=1
            else:
                sim=0     
        return sim    
    
    def __get_similar_set(self,word, input_type = 'word'):
        if input_type == 'vector':
            sim_list = self.model.similar_by_vector(word)
        else:
            sim_list = self.model.most_similar(word) 
        
        #return  {self.porter.stem(word.lower()) for word,sim in sim_list}
        return  {self.lemmatizer.lemmatize(word.lower()) for word,sim in sim_list}
    
    def load_embedding_hashes(self,weight=0.5):
        
        self.EmbA = {}
        self.EmbB = {}
        self.EmbC = {}
    
        for table in self.schema_graph.vertices():

            if table not in self.model:
                continue

            self.EmbA[table]={}
            self.EmbB[table]= {}
            self.EmbC[table]= {}
            
            self.EmbA[table]['*'] = self.__get_similar_set(table) 

            if table in self.schema_index:
                for column in self.schema_index[table]:
                    if column not in self.model or column=='id':
                        continue

                    self.EmbA[table][column]=self.__get_similar_set(column)

                    self.EmbB[table][column]=self.__get_similar_set( (table,column) )

                    avg_vec = (self.model[table]*weight + self.model[column]*(1-weight))                   
                    self.EmbC[table][column] = self.__get_similar_set(avg_vec, input_type = 'vector')
                
            for neighbor_table in self.schema_graph.neighbours(table):

                if neighbor_table not in self.model:
                    continue
                
                self.EmbB[table][neighbor_table] = self.__get_similar_set( (table,neighbor_table) )
```


```python
similarities=Similarities(word_embeddings_model,
                              DEFAULT_DSCONFIG.schema_index,
                              G)
```

#### SKMGen

```python
def SKMGen(Q,schema_index,similarities,**kwargs):    
    threshold = kwargs.get('threshold',1)
    keyword_matches_to_ignore = kwargs.get('keyword_matches_to_ignore',set())
    
    S = set()
    
    for keyword in Q:
        for table in schema_index:                       
            for attribute in ['*']+list(schema_index[table].keys()):
                
                if(attribute=='id'):
                    continue
                
                sim = similarities.word_similarity(keyword,table,attribute)
                
                if sim >= threshold:
                    skm = KeywordMatch(table,schema_filter={attribute:{keyword}})
                    
                    if skm not in keyword_matches_to_ignore:
                        S.add(skm)
                    
    return S
```

```python
if STEP_BY_STEP:    
    print('FINDING SCHEMA-SETS')        
    Sq = SKMGen(Q,DEFAULT_DSCONFIG.schema_index,similarities)
    print(len(Sq),' SCHEMA-SETS CREATED\n')
    pp(Sq)
```

## Query Matching


### Minimal Cover

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

### Merging Schema Filters

```python
def MergeSchemaFilters(QM):
    table_hash={}
    for keyword_match in QM:
        joint_schema_filter,value_keyword_matches = table_hash.setdefault(keyword_match.table,({},set()))

        for attribute, keywords in keyword_match.schema_filter:
            joint_schema_filter.setdefault(attribute,set()).update(keywords)

        if len(keyword_match.value_filter) > 0:
            value_keyword_matches.add(keyword_match)
    
    merged_qm = set()
    for table,(joint_schema_filter,value_keyword_matches) in table_hash.items():    
        if len(value_keyword_matches) > 0:
            joint_value_filter = {attribute:keywords 
                                  for attribute,keywords in value_keyword_matches.pop().value_filter}
        else:
            joint_value_filter={}

        joint_keyword_match = KeywordMatch(table,
                                           value_filter=joint_value_filter,
                                           schema_filter=joint_schema_filter)

        merged_qm.add(joint_keyword_match)
        merged_qm.update(value_keyword_matches) 

    return merged_qm
```

```python

```

### QMGen

```python
def groups_of_keywords(keyword_query):
    return [set(tokenize_string(group)) for group in keyword_query.split('"')[1::2]]
```

```python
groups = groups_of_keywords(keyword_query)
groups
```

```python
def contains_groups(M,groups_of_keywords):
    for group in groups_of_keywords:
        found = False
        for keyword_match in M:           
            km_keywords = set(keyword_match.keywords())
            
            num_common_elements = len(group &  km_keywords)
            
            # group must be either disjoint or a subset of km_keywords
            if num_common_elements!=0 and num_common_elements!=len(group) :
                return False
    return True
```

```python
def QMGen(Q,Rq,groups_of_keywords,**kwargs):    
    max_qm_size = kwargs.get('max_qm_size',3)
    #Input:  A keyword query Q, The set of non-empty non-free tuple-sets Rq
    #Output: The set Mq of query matches for Q
    
    '''
    Query match is a set of tuple-sets that, if properly joined,
    can produce networks of tuples that fulfill the query. They
    can be thought as the leaves of a Candidate Network.
    
    '''
    
    
    Mq = []
    for i in range(1,min(len(Q),max_qm_size)+1):
        for M in itertools.combinations(Rq,i):            
            if(MinimalCover(M,Q)):
                merged_qm = MergeSchemaFilters(M)
                
                if contains_groups(merged_qm,groups_of_keywords):
                    Mq.append(merged_qm)
                   
    return Mq 
```

```python
if STEP_BY_STEP:
    print('GENERATING QUERY MATCHES')
    TMaxQM = 3
    Mq = QMGen(Q,Rq|Sq,groups)
    print (len(Mq),'QUERY MATCHES CREATED\n')  
```

### QMRank

```python
def QMRank(Mq,value_index,schema_index,similarities,**kwargs):  
    show_log = kwargs.get('show_log',False)
    topk_qms = kwargs.get('topk_qms',10)
    
    Ranking = []  

    for M in Mq:
        #print('=====================================\n')
        value_prod = 1 
        schema_prod = 1
        score = 1
        
        there_is_schema_terms = False
        there_is_value_terms = False
        
        for keyword_match in M:
            
            for table, attribute, value_words in keyword_match.value_mappings():

                (Norm,num_distinct_words,num_words,max_frequency) = schema_index[table][attribute]                
                wsum = 0


                if show_log:
                    print('Norm: {}\nMaxFrequency {}\n'.format(Norm,max_frequency))


                for term in value_words:    

                    IAF = value_index.get_IAF(term)

                    frequency = len(value_index.get_mappings(term,table,attribute))
                    TF = (frequency/max_frequency)
                    wsum = wsum + TF*IAF
                    if show_log:
                        print('- Term: {}\n  Frequency:{}\n  TF:{}\n  IAF:{}\n'.format(term,frequency,TF,IAF))

                    there_is_value_terms = True

                cos = wsum/Norm
                value_prod *= cos     
        
        
            for table, attribute, schema_words in keyword_match.schema_mappings():
                schemasum = 0
                for term in schema_words:
                    sim = similarities.word_similarity(term,table,attribute)
                    schemasum += sim

                    if show_log:
                        print('- Term: {}\n  Sim:{}\n'.format(term,sim))

                    there_is_schema_terms = True

                schema_prod *= schemasum   
        
        value_score  = value_prod
        schema_score = schema_prod
        
        if there_is_value_terms:
            score *= value_score
        else:
            value_score = 0
            
            
        if there_is_schema_terms:
            score *= schema_score
        else:
            schema_score = 0
                
        Ranking.append( (M,score) )
        
    
    ranked_qms = sorted(Ranking,key=lambda x: x[1]/len(x[0]),reverse=True)
    if topk_qms!=-1:
        ranked_qms=ranked_qms[:topk_qms]
        
    return ranked_qms
```


```python
if STEP_BY_STEP:
    print('RANKING QUERY MATCHES')
    RankedMq = QMRank(Mq,DEFAULT_DSCONFIG.value_index,DEFAULT_DSCONFIG.schema_index,similarities)   
    
    top_k = 20
    num_pruned_qms = len(RankedMq)-top_k
    if num_pruned_qms>0:
        print(num_pruned_qms,' QUERY MATCHES SKIPPED (due to low score)')
    else:
        num_pruned_qms=0        
        
    for (j, (M,score) ) in enumerate(RankedMq[:top_k]):
        print(j+1,'ª QM')           
        pp(M)
        print('Score: ',"%.8f" % score)
        
        #print('\n----Details----\n')
        #QMRank([M],value_index,schema_index,similarities,show_log=True)

        print('----------------------------------------------------------------------\n')
```

## Candidate Networks


### Class CandidateNetwork

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
    
    def num_free_keyword_matches(self):
        i=0
        for keyword_match,alias in self.vertices():
            if keyword_match.is_free():
                i+=1
        return i
            
    def is_sound(self):
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
cnx=CandidateNetwork.from_str('''ORGANIZATION.s(*{organization})
	<IS_MEMBER
		>COUNTRY.v(name{oman})
	<IS_MEMBER
		>COUNTRY.v(name{panama})''')
CandidateNetwork.from_json(cnx.to_json())
```

### Translation of CNs


#### get_sql_from_cn

```python
def get_sql_from_cn(G,Cn,**kwargs):
    
    show_evaluation_fields=kwargs.get('show_evaluation_fields',False)
    rows_limit=kwargs.get('rows_limit',1000) 
    
    #Future feature
    multiple_edges=False
    
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
        
        if show_evaluation_fields:
            tables__search_id.append('{}.__search_id'.format(alias))

        if prev_vertex is None:
            selected_tables.append('{} {}'.format(keyword_match.table,alias))
        else:
            # After the second table, it starts to use the JOIN syntax
            prev_keyword_match,prev_alias = prev_vertex
            if direction == '>':
                constraint_keyword_match,constraint_alias = prev_vertex
                foreign_keyword_match,foreign_alias = vertex
            else:
                constraint_keyword_match,constraint_alias = vertex
                foreign_keyword_match,foreign_alias = prev_vertex
            
            edge_info = G.get_edge_info(constraint_keyword_match.table,
                                        foreign_keyword_match.table)
            
            for constraint in edge_info:
                join_conditions = []
                for (constraint_column,foreign_column) in edge_info[constraint]:                   
                    join_conditions.append('{}.{} = {}.{}'.format(constraint_alias,
                                                                  constraint_column,
                                                                  foreign_alias,
                                                                  foreign_column ))
                
                selected_tables.append('JOIN {} {} ON {}'.format(keyword_match.table,
                                                                 alias,
                                                                 '\n\t\tAND '.join(join_conditions)))
                if not multiple_edges:
                    break            
            
            if show_evaluation_fields:
                relationships__search_id.append('({}.__search_id, {}.__search_id)'.format(alias,prev_alias))


    for table,aliases in hashtables.items():        
        for i in range(len(aliases)):
            for j in range(i+1,len(aliases)):
                disambiguation_conditions.append('{}.ctid <> {}.ctid'.format(aliases[i],aliases[j]))
        
    if len(tables__search_id)>0:
        tables__search_id = ['({}) AS Tuples'.format(', '.join(tables__search_id))]
    if len(relationships__search_id)>0:
        relationships__search_id = ['({}) AS Relationships'.format(', '.join(relationships__search_id))]

    sql_text = '\nSELECT\n\t{}\nFROM\n\t{}\nWHERE\n\t{}\nLIMIT {};'.format(
        ',\n\t'.join( tables__search_id+relationships__search_id+list(selected_attributes) ),
        '\n\t'.join(selected_tables),
        '\n\tAND '.join( disambiguation_conditions+filter_conditions),
        rows_limit)
    return sql_text
```

#### exec_sql

```python
def exec_sql (conn_string,SQL,**kwargs):
    #print('RELAVANCE OF SQL:\n')
    #print(SQL)
    show_results=kwargs.get('show_results',True)
        
    with psycopg2.connect(conn_string) as conn:
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
                if show_results:
                    print(table)
            except:
                print('ERRO SQL:\n',SQL)
                raise
                        
            return cur.rowcount>0
```

```python
# exec_sql(DEFAULT_CONFIG.conn_string,get_sql_from_cn(G,Cn,rowslimit=1))
```

### Generation and Ranking of CNs


#### sum_norm_attributes

```python
sorted([(sum(Norm for (Norm,num_distinct_words,num_words,max_frequency) in DEFAULT_DSCONFIG.schema_index[table].values()),
  table) for table in DEFAULT_DSCONFIG.schema_index],reverse=True)
```

```python
sorted([(1,table) for table in DEFAULT_DSCONFIG.schema_index],reverse=True)
```

```python
def worst_neighbors_sorting(directed_neighbor):
        direction,adj_table = directed_neighbor
        if adj_table == 'movie_info':
            return 6
        if adj_table == 'cast_info':
            return 5
        if adj_table == 'role_type':
            return 4
        if adj_table == 'char_name':
            return 3
        if adj_table == 'name':
            return 2
        if adj_table == 'title':
            return 1
        return 0
```

#### CNGraphGen

```python
def factory_sum_norm_attributes(schema_index):
    
    def sum_norm_attributes(directed_neighbor):
        direction,adj_table = directed_neighbor
        if adj_table not in schema_index:
            return 0
        return sum(Norm for (Norm,
                             num_distinct_words,
                             num_words,max_frequency) in schema_index[adj_table].values())        
    
    return sum_norm_attributes
```

```python
def CNGraphGen(conn_string,schema_index,G,QM,**kwargs):  
    
    show_log = kwargs.get('show_log',False)
    max_cn_size = kwargs.get('max_cn_size',5)
    topk_cns_per_qm = kwargs.get('topk_cns_per_qm',1)
    directed_neighbor_sorting_function = kwargs.get('directed_neighbor_sorting_function',
                                                    factory_sum_norm_attributes(schema_index))
    non_empty_only = kwargs.get('non_empty_only',False)
    desired_cn = kwargs.get('desired_cn',None)
    gns_elapsed_time = kwargs.get('gns_elapsed_time',[])
    
    
    
    start_time = timer()
    
    if show_log:
        print('================================================================================\nSINGLE CN')
        print('max_cn_size ',max_cn_size)
        print('FM')
        pp(QM)
    
    CN = CandidateNetwork()
    CN.add_vertex(next(iter(QM)))
    
    if len(QM)==1:
        if non_empty_only:
            sql = get_sql_from_cn(G,
                                     CN,
                                     rowslimit=1)

            non_empty = exec_sql(conn_string,
                                 sql,
                                 show_results=False)

            
        if non_empty_only and non_empty==False:
            return {}
        return {CN}
    
    returned_cns = list()
    ignored_cns = list()
    
    table_hash={}
    for keyword_match in QM:
        table_hash.setdefault(keyword_match.table,set()).add(keyword_match)    
    
    F = deque()
    F.append(CN)
        
    while F:        
        CN = F.popleft()

        for vertex_u in CN.vertices():
            keyword_match,alias = vertex_u

            sorted_directed_neighbors = sorted(G.directed_neighbours(keyword_match.table),
                                               reverse=True,
                                               key=directed_neighbor_sorting_function) 
            
            for direction,adj_table in sorted_directed_neighbors:
                
                if adj_table in table_hash: 
                    for adj_keyword_match in table_hash[adj_table]:
                        
                        if adj_keyword_match not in CN.keyword_matches():
                            
                            new_CN = copy.deepcopy(CN)
                            vertex_v = new_CN.add_vertex(adj_keyword_match)
                            new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction) 

                            if (new_CN not in F and
                                new_CN not in returned_cns and
                                new_CN not in ignored_cns and
                                len(new_CN)<=max_cn_size and
                                new_CN.is_sound() and
                                len(list(new_CN.leaves())) <= len(QM) and
                                new_CN.num_free_keyword_matches()+len(QM) <= max_cn_size
                               ):

                                if new_CN.minimal_cover(QM):
                                    
                                    if non_empty_only == False:
                                        
                                        current_time = timer()
                                        gns_elapsed_time.append(current_time-start_time)
                                        
                                        returned_cns.append(new_CN)
                                        
                                        if new_CN == desired_cn:
                                            return returned_cns
                                    else:
                                        sql = get_sql_from_cn(G,
                                                                 new_CN,
                                                                 rowslimit=1)
                                            
                                        non_empty = exec_sql(conn_string,
                                                             sql,
                                                             show_results=False)
                                            
                                        if non_empty:
                                            current_time = timer()
                                            gns_elapsed_time.append(current_time-start_time)
                                            
                                            returned_cns.append(new_CN)
                                            
                                            if new_CN == desired_cn:
                                                return returned_cns
                                        else:
                                            ignored_cns.append(new_CN)                                        
                                    
                                    
                                    if len(returned_cns)>=topk_cns_per_qm:                                            
                                        return returned_cns
                                    
                                elif len(new_CN)<max_cn_size:
                                    F.append(new_CN)                
                
                
                new_CN = copy.deepcopy(CN)
                adj_keyword_match = KeywordMatch(adj_table)
                vertex_v = new_CN.add_vertex(adj_keyword_match)
                new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction)
                if (new_CN not in F and
                    len(new_CN)<max_cn_size and
                    new_CN.is_sound() and 
                    len(list(new_CN.leaves())) <= len(QM) and
                    new_CN.num_free_keyword_matches()+len(QM) <= max_cn_size
                   ):
                    F.append(new_CN)

    return returned_cns
```

```python
if STEP_BY_STEP:
    max_cn_size=5
   
    (QM,score) = RankedMq[0]
    print('GENERATING CNs FOR QM:',QM)
    
    Cns = CNGraphGen(DEFAULT_DSCONFIG.conn_string,
                     DEFAULT_DSCONFIG.schema_index,
                     G,QM,max_cn_size=max_cn_size,
                     directed_neighbor_sorting_function=worst_neighbors_sorting,
                     non_empty_only=True,
                     show_sql_runs=True,
                     topk_cns_per_qm=10,
                    )
    
    for j, Cn in enumerate(Cns):
        print('{}ª CN\n{}\nScore:{:.8f}'.format(j+1,Cn,score))
```

#### MatchCN

```python
def MatchCN(conn_string,schema_index,G,ranked_mq,**kwargs):
    topk_cns = kwargs.get('topk_cns',20)
    show_log = kwargs.get('show_log',False)
    CNGraphGen_kwargs = kwargs.get('CNGraphGen_kwargs',{})
    
    
    un_ranked_cns = []    
    generated_cns=[]
    
    num_cns_available = topk_cns
    
    for i,(QM,qm_score) in enumerate(ranked_mq):
        if topk_cns!=-1 and num_cns_available<=0:
            break
            
        if show_log:
            print('{}ª QM:\n{}\n'.format(i+1,QM))
        Cns = CNGraphGen(conn_string,schema_index,G,QM,**CNGraphGen_kwargs)
        if show_log:
            print('Cns:')
            pp(Cns)
        
    
        for i,Cn in enumerate(Cns):            
            if(Cn not in generated_cns):          
                if num_cns_available<=0:
                    break
                    
                generated_cns.append(Cn)

                #Dividindo score pelo tamanho da cn (SEGUNDA PARTE DO RANKING)                
                cn_score = qm_score/len(Cn)
                
                un_ranked_cns.append( (Cn,cn_score) )
                
                num_cns_available -=1
            
    #Ordena CNs pelo CnScore
    ranked_cns=sorted(un_ranked_cns,key=lambda x: x[1],reverse=True)
    
    return ranked_cns
```

```python
if STEP_BY_STEP:   
    print('GENERATING CANDIDATE NETWORKS')  
    RankedCns = MatchCN(DEFAULT_DSCONFIG.conn_string,
                        DEFAULT_DSCONFIG.schema_index,
                        G,RankedMq,
                        **{
                            'topk_cns':20,
                            'CNGraphGen_kwargs':{
                                'non_empty_only':False,
                                'topk_cns_per_qm':2,
                                'directed_neighbor_sorting_function':worst_neighbors_sorting,
                            },
                        }
                        
                       )
    print (len(RankedCns),'CANDIDATE NETWORKS CREATED AND RANKED\n')
    
    for (j, (Cn,score) ) in enumerate(RankedCns):
        print('{}ª CN\n{}\nScore:{:.8f}'.format(j+1,Cn,score))
```

## Keyword Search


### keyword_search

```python
def keyword_search(qsconfig,
                   word_embeddings_model,
                **kwargs
         ):
    dsconfig = qsconfig.dsconfig
    
    show_log = kwargs.get('show_log',False)
    output_results = kwargs.get('output_results',OrderedDict())
    queryset = kwargs.get('queryset', get_query_sets(qsconfig.queryset_filename))    
    
    
    VKMGen_kwargs = kwargs.get('VKMGen_kwargs',{})
    SKMGen_kwargs = kwargs.get('SKMGen_kwargs',{})
    QMGen_kwargs  = kwargs.get('QMGen_kwargs',{})
    QMRank_kwargs = kwargs.get('QMRank_kwargs',{})
    MatchCN_kwargs = kwargs.get('MatchCN_kwargs',{})
    schema_graph_kwargs = kwargs.get('schema_graph_kwargs', {})
    similarity_kwargs = kwargs.get('similarity_kwargs',{})   
    
    top_k=10
    
    G = get_schema_graph(dsconfig.conn_string,**schema_graph_kwargs)
    
    similarities=Similarities(word_embeddings_model,
                              dsconfig.schema_index,
                              G,
                              **similarity_kwargs)
    
    for (i,keyword_query) in enumerate(queryset):
        
        begin_query_processing_time = timer()
        
        print('{}ª QUERY: {}\n'.format(i+1,keyword_query))
        
        Q = tuple(tokenize_string(keyword_query))
        groups = groups_of_keywords(keyword_query)
        
        if Q in output_results:
            print('QUERY SKIPPED')
            continue
            
        print('FINDING VALUE-KEYWORD MATCHES')
        Rq = VKMGen(Q, dsconfig.value_index,**VKMGen_kwargs)
        print('{} VALUE-KEYWORD MATCHES CREATED\n'.format(len(Rq)))
        
        if show_log:
            pp(Rq)
            
        print('FINDING SCHEMA-KEYWORD MATCHES')        
        Sq = SKMGen(Q,dsconfig.schema_index,similarities,**SKMGen_kwargs)
        print('{} VALUE-KEYWORD MATCHES CREATED\n'.format(len(Sq)))
        
        if show_log:
            pp(Sq)
            
        end_keyword_matching_time = timer()
        
        print('GENERATING QUERY MATCHES')
        Mq = QMGen(Q,Rq|Sq,groups)
        print (len(Mq),'QUERY MATCHES CREATED\n')
                
        print('RANKING QUERY MATCHES')
        RankedMq = QMRank(Mq,
                          dsconfig.value_index,
                          dsconfig.schema_index,
                          similarities,**QMRank_kwargs)   
                
        print('{} QUERY MATCHES SKIPPED (due to low score)'.format(len(Mq)-len(RankedMq)))
        
        if show_log:
            for (j, (QM,score) ) in enumerate(RankedMq[:top_k]):
                print('{}ª Q, {}ª QM\n{}\nScore:{:.8f}'.format(i+1,j+1,QM,score))
            print('\n{}\n'.format('-'*80))
            
        end_query_matching_time = timer()
        
        
        print('GENERATING CANDIDATE NETWORKS')     
        RankedCns = MatchCN(dsconfig.conn_string,
                            dsconfig.schema_index,
                            G,RankedMq,**MatchCN_kwargs)
        
        print (len(RankedCns),'CANDIDATE NETWORKS CREATED RANKED\n')
        
        if show_log:
            for (j, (Cn,score) ) in enumerate(RankedCns):
                for (j, (Cn,score) ) in enumerate(RankedCns):
                    print('{}ª CN\n{}\nScore:{:.8f}\n{}\n'.format(j+1,Cn,score,'-'*80))
        
        end_candidate_networks_time = timer()
        
        
        keyword_matching_time   = end_keyword_matching_time   - begin_query_processing_time
        query_matching_time     = end_query_matching_time     - end_keyword_matching_time
        candidate_networks_time = end_candidate_networks_time - end_query_matching_time
        execution_time          = end_candidate_networks_time - begin_query_processing_time
        
        output_results[keyword_query]={'query_matches':RankedMq[:top_k],
                                       'candidate_networks':RankedCns,
                                       'keyword_matching_time':keyword_matching_time,
                                       'query_matching_time':query_matching_time,
                                       'candidate_networks_time':candidate_networks_time,
                                       'execution_time':execution_time,
                                       'num_query_matches':len(Mq),
                                       'num_keyword_matches':len(Rq)+len(Sq),
                          }
        print('Elapsed Time:{}'.format(str(datetime.timedelta(seconds=execution_time))))
        print('\n{}\n'.format('='*80))
        
    return output_results
```

### DEFAULT_PARAMS

```python
DEFAULT_PARAMS = {
                    'show_log':False,
                    'output_results':OrderedDict(),
                    'queryset':get_query_sets(DEFAULT_QSCONFIG.queryset_filename),
                    'schema_graph_kwargs': {
                        'fk_constraints':None,
                    },
                    'similarity_kwargs':{
                        'use_path_sim'   :True,
                        'use_wup_sim'    :True,
                        'use_jaccard_sim':True,
                        'use_emb_sim'    :False,
                        'use_emb10_sim'  :False,
                        'emb10_sim_type' :'B',
                    },
                    'VKMGen_kwargs':{
                        'ignored_tables':[]
                    },
                    'SKMGen_kwargs':{
                        'threshold':1,
                    },
                    'QMGen_kwargs':{
                        'max_qm_size':3,
                    },
                    'QMRank_kwargs':{
                        'topk_qms':10,
                        'show_log':False,
                    },
                    'MatchCN_kwargs':{
                        'show_log':False,
                        'topk_cns':20,
                        'CNGraphGen_kwargs':{
                            'max_cn_size':5,
                            'show_log':False,
                            'show_sql':False,
                            'show_sql_runs':False,
                            'topk_cns_per_qm':2,
                            'directed_neighbor_sorting_function':
                            factory_sum_norm_attributes(DEFAULT_DSCONFIG.schema_index),
                            'non_empty_only':False,
                        }
                    },
                }
```

### main

```python
DSCONFIGS_TO_PREPROCESS
```

```python
QSCONFIGS_TO_PROCESS
```

```python
QSCONFIGS_TO_PROCESS[:2]
```

```python
cns_results = {}
necns_results = {} 
```

```python
# QSCONFIGS_TO_PROCESS =  [mondial_coffman_original_qsconfig]
```

```python
mas_fk_constraints = [
    ('fk_author_organization', 'author', 'aid', 'organization', 'oid'),
    ('fk_publication_conference', 'publication', 'cid', 'conference', 'cid'),
    ('fk_publication_journal', 'publication', 'jid', 'journal', 'jid'),
    ('fk_domain_author_author', 'domain_author', 'aid', 'author', 'aid'),
    ('fk_domain_author_domain', 'domain_author', 'did', 'domain', 'did'),
    ('fk_domain_conference_conference', 'domain_conference', 'cid',
     'conference', 'cid'),
    ('fk_domain_conference_domain', 'domain_conference', 'did', 'domain',
     'did'),
    ('fk_domain_journal_journal', 'domain_journal', 'jid', 'journal', 'jid'),
    ('fk_domain_journal_domain', 'domain_journal', 'did', 'domain', 'did'),
    ('fk_domain_keyword_domain', 'domain_keyword', 'did', 'domain', 'did'),
    ('fk_domain_keyword_keyword', 'domain_keyword', 'kid', 'keyword', 'kid'),
    ('fk_publication_keyword_publication', 'publication_keyword', 'pid',
     'publication', 'pid'),
    ('fk_publication_keyword_keyword', 'publication_keyword', 'kid', 'keyword',
     'kid'), ('fk_writes_author', 'writes', 'aid', 'author', 'aid'),
    ('fk_writes_publication', 'writes', 'pid', 'publication', 'pid')
]

mas_jagadish_qsconfig.params = {
    'schema_graph_kwargs': {
        'fk_constraints': mas_fk_constraints,
    },
    'VKMGen_kwargs': {
        'threshold':0.8,
        'ignored_tables': ['cite', 'domain_publication', 'ids', 'keyword_variations'],
        'ignored_attributes': [('publication', 'abstract')]
    },
}

imdb_coffman_original_qsconfig.params = {
    'SKMGen_kwargs': {
        'keyword_matches_to_ignore':
        {KeywordMatch.from_str('CHAR_NAME.s(name{name})')},
    },
    'VKMGen_kwargs':{
        'ignored_tables':['cast_info']
    },
}

imdb_coffman_ci_qsconfig.params = {
    'SKMGen_kwargs': {
        'keyword_matches_to_ignore':
        {KeywordMatch.from_str('CHAR_NAME.s(name{name})')},
    },
    'VKMGen_kwargs':{
        'ignored_tables':['cast_info']
    },
}
```

```python
endpoint
```

```python
num_iterations = 10 

cns_results_list = []
for ir in range(num_iterations):
    cns_results_list.append({})
    
    for qsconfig in QSCONFIGS_TO_PROCESS:
        if qsconfig==mas_jagadish_qsconfig and ir>0:
            continue

        params = {
            'MatchCN_kwargs': {
                'CNGraphGen_kwargs': {
                    'non_empty_only': False,
                }
            }
        }

        # Note that qsconfig-specific parameters overwirites the experiment params
        params.update(qsconfig.params)

        print('{0}\n{0}\nCONFIG: {1}\n'.format('=' * 80, qsconfig))
        cns_results_list[ir][qsconfig] = keyword_search(qsconfig, word_embeddings_model,
                                               **params)
```

```python
cns_results = cns_results_list[0]
```

## Export Results to JSON


# ALTERAR SAÍDA DO MÉTODO KEYWORD SEARCH, COLOCA SCORES EM UMA CHAVE SEPARADA

```python
def export_results_json(cns_results,all_fields = True):
    
    for qsconfig in cns_results:
        data = []
        for i, keyword_query in enumerate(
                get_query_sets(qsconfig.queryset_filename)):
            if keyword_query == '':
                continue

            result = cns_results[qsconfig][keyword_query]
            
            obj = {}
            
            if all_fields:
                obj = result.copy()
            
#             obj['num_query_matches'] = obj['num_query_matches'][0]
            obj['keyword_query'] = keyword_query            
            obj['query_matches'] = [
                [km.to_json_serializable() for km in match]
                for (match, score) in result['query_matches']
            ]
            
            obj['candidate_networks'] = [
                cn.to_json_serializable()
                for (cn, score) in result['candidate_networks']
            ]
            
            data.append(obj)

        with open('results/{}.json'.format(qsconfig.name), 'w') as outfile:
            json.dump(data, outfile)
```

```python
# def export_results_json2(cns_results,all_fields = True):
    
#     for qsconfig in cns_results:
#         data = []
#         for i, keyword_query in enumerate(
#                 get_query_sets('querysets/queryset_imdb_coffman_revised.txt')):
#             if keyword_query == '':
#                 continue

#             result = cns_results[qsconfig][keyword_query]
            
#             obj = {}
            
#             if all_fields:
#                 obj = result.copy()
            
# #             obj['num_query_matches'] = obj['num_query_matches'][0]
#             obj['keyword_query'] = keyword_query            
#             obj['query_matches'] = [
#                 [km.to_json_serializable() for km in match]
#                 for match in result['query_matches']
#             ]
            
#             obj['candidate_networks'] = [
#                 cn.to_json_serializable()
#                 for cn in result['candidate_networks']
#             ]
            
#             data.append(obj)

#         with open('new_golden_standards/queryset_imdb_coffman_revised.json', 'w') as outfile:
#             json.dump(data, outfile)
```

```python
# export_results_json2({imdb_coffman_original_qsconfig:new_gs})
```

## Import Results from JSON

```python
def import_results_json(results_json_filename, all_fields = True):
    with open(results_json_filename, 'r') as f:
        json_serializable = json.load(f)
    
        results = {}
        for obj in json_serializable:
            
            result = {}
            
            if all_fields:
                result = obj.copy()
                del result['keyword_query']
            
            keyword_query = obj['keyword_query']
            
            result['query_matches'] = [
                set(map(KeywordMatch.from_json_serializable, json_query_match))
                for json_query_match in obj['query_matches']
            ]

            result['candidate_networks'] = list(
                map(CandidateNetwork.from_json_serializable,
                    obj['candidate_networks'])
            )

            results[keyword_query] = result
            
        if '' in results:
            del results['']
            
        return results
            
results_json_filename='new_golden_standards/queryset_imdb_coffman_revised.json'
imported_results = import_results_json(results_json_filename)
```

```python
imported_results = import_results_json('results/{}.json'.format(imdb_coffman_original_qsconfig.name))
```

```python
golden_standards = import_results_json('new_golden_standards/queryset_imdb_coffman_revised.json')
```

```python
imported_results['harrison ford george lucas']
```

```python
queryset = get_query_sets('querysets/queryset_imdb_coffman_revised.txt')
```

## Set Golden Standards

```python
results = imported_results
```

```python
def set_golden_standards(queryset,results,premade_golden_standards=None):
    from IPython.display import clear_output
    
    
    def custom_query_match():
        i=1
        query_match = set()
        answer = input('Insert the {}a Keyword Match from the QM (type f if QM is complete):'.format(i))
        while answer != 'f':
            keyword_match = KeywordMatch.from_str(answer)
            query_match.add(keyword_match)
            i+=1
            answer = input('Insert the {}a Keyword Match from the QM (type f if QM is complete):'.format(i))
        print('Generated QM:\n{}\n'.format(query_match))
        return query_match

    answer = None  
    golden_standards = {}

    for i,keyword_query in enumerate(queryset):
        
        
        if keyword_query in premade_golden_standards:
            golden_standards[keyword_query]=premade_golden_standards[keyword_query]
        else:
            golden_standards[keyword_query]={'query_matches':[],
                                             'candidate_networks':[]}
            
        
        # Generating Golden Query Matches
        if len(golden_standards[keyword_query]['query_matches']) == 0:
      
            
            for query_match in results[keyword_query]['query_matches']+['EOF']:
                
                if query_match == 'EOF':
                    
                    answer = input('The system did not generate any Query Match for the Query Q given.\
\nQ:\n{}\nWould you like to write the correct relevant QM for it?(type y if yes)'
                                   .format(keyword_query))
                    if answer == 'y':
                        golden_standards[keyword_query]['query_matches']=[custom_query_match()]
                    continue
                
                answer = input('Is the following Query Match relevant for the Query Q given?\
(type y if yes)\n{}a Q:\n{}\nQM:\n{}\n'
                               .format(i+1,keyword_query,query_match))
                clear_output()
                if answer == 'y':
                    golden_standards[keyword_query]['query_matches']=[query_match]
                    break
                elif answer == 'custom':
                    golden_standards[keyword_query]['query_matches']=[custom_query_match()]
                    break
                elif answer == 'skip':
                    break
                elif answer == 'stop':
                    return golden_standards
                
        # Generating Golden Query Matches
        if len(golden_standards[keyword_query]['candidate_networks']) == 0:
            
            for candidate_network in results[keyword_query]['candidate_networks']+['EOF']:
                
                if candidate_network == 'EOF':
                    
                    answer = input('The system did not generate any Candidate Network for the Query Q given.\
\nQ:\n{}\nWould you like to write the correct relevant CN for it?(type y if yes)'
                                   .format(keyword_query))
                    if answer == 'y':
                        candidate_network = CandidateNetwork.from_str(input("Write the golden Candidate Network:"))
                        golden_standards[keyword_query]['candidate_networks']=[candidate_network]
                    continue
                
                if len(golden_standards[keyword_query]['query_matches'])>0:
                    query_match = golden_standards[keyword_query]['query_matches'][0]
                    if query_match != candidate_network.non_free_keyword_matches():
                        continue
                
                answer = input('Is the following Candidate Network relevant for the Query Q given?\
(type y if yes)\n{}a Q:\n{}\nCN:\n{}\n'
                               .format(i+1,keyword_query,candidate_network))
                clear_output()
                if answer == 'y':
                    golden_standards[keyword_query]['candidate_networks']=[candidate_network]
                    break
                elif answer == 'custom':
                    candidate_network = CandidateNetwork.from_str(input("Write the golden Candidate Network:"))
                    golden_standards[keyword_query]['candidate_networks']=[candidate_network]
                    break
                elif answer == 'skip':
                    break
                elif answer == 'stop':
                    return golden_standards
                
    return golden_standards
```

```python
# new_gs = set_golden_standards(queryset,results,golden_standards)
```

```python
# get_query_sets(imdb_coffman_original_qsconfig.queryset_filename)
```

## Evaluation


### Number of Keyword Matches / Query Matches

```python
num_keyword_matches = {}
num_query_matches = {}
for qsconfig in QSCONFIGS_TO_PROCESS:
    
    num_keyword_matches[qsconfig.name]=[]
    num_query_matches[qsconfig.name]=[]
    
    for Q in cns_results[qsconfig]:
        
        num_keyword_matches[qsconfig.name].append(cns_results[qsconfig][Q]['num_keyword_matches'])
        num_query_matches[qsconfig.name].append(cns_results[qsconfig][Q]['num_query_matches'])
        
        
#         if cns_results[qsconfig][Q]['num_query_matches'][0] > 200:
#             print(Q, cns_results[qsconfig][Q]['num_query_matches'][0])
```

```python
label_order = [qsconfig.name for qsconfig in QSCONFIGS_TO_PROCESS]

box_plot_data=[num_keyword_matches[label] for label in label_order]
plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=label_order)

plt.xlabel('Keyword Matches')
plt.show()
```

```python
box_plot_data=[num_query_matches[label] for label in label_order]
plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=label_order)

plt.xlabel('Query Matches')

plt.show()
```

```python
box_plot_data=[num_query_matches[label] for label in label_order]

width_ratios = [4, 1]

fig,(ax, ax2)  = plt.subplots(nrows = 1, ncols = 2, sharey=True,gridspec_kw={'width_ratios': width_ratios})

# plot the same data on both axes
ax.boxplot(box_plot_data,vert=0,patch_artist=True,labels=label_order)
ax2.boxplot(box_plot_data,vert=0,patch_artist=True,labels=label_order)

ax.set_xlim(xmin=0, xmax=300)  # outliers only
ax2.set_xlim(xmin=2800, xmax=3000) # most of the data

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.tick_params(axis='y',labelleft=False,length=0)  # don't put tick labels at the top


dx = .015*(sum(width_ratios)/width_ratios[0])  # how big to make the diagonal lines in axes coordinates
dy = .015 

# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1 - dx, 1 + dx), (1 - dy, 1 + dy), **kwargs)  # top-left diagonal
ax.plot((1 - dx, 1 + dx), (0 - dy, 0 + dy), **kwargs)  # top-right diagonal


dx = .015*(sum(width_ratios)/width_ratios[1])  # how big to make the diagonal lines in axes coordinates
dy = .015

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((0 - dx, 0 + dx), (1 - dy, 1 + dy), **kwargs)  # bottom-left diagonal
ax2.plot((0 - dx, 0 + dx), (0 - dy, 0 + dy), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'


fig.text(0.5, 0.03, 'Query Matches', ha='center')

plt.show()
```

### Performance Evaluation

```python

```

```python
perfoe
for qsconfig in QSCONFIGS_TO_PROCESS[1:]:
    
    
```

```python
cn_results = {}
keyword_matching_time = {}
for qsconfig in QSCONFIGS_TO_PROCESS[1:]:
    
    cn_results[qsconfig.name] =  import_results_json('results/{}.json'.format(qsconfig.name))
    
#     for Q in cns_results[qsconfig]:
        
        
        
    
    
    
    
# num_keyword_matches = {}
# num_query_matches = {}
# for qsconfig in QSCONFIGS_TO_PROCESS:
    
#     num_keyword_matches[qsconfig.name]=[]
#     num_query_matches[qsconfig.name]=[]
    
#     for Q in cns_results[qsconfig]:
        
#         num_keyword_matches[qsconfig.name].append(cns_results[qsconfig][Q]['num_keyword_matches'])
#         num_query_matches[qsconfig.name].append(cns_results[qsconfig][Q]['num_query_matches'])
```

```python
performance_results = {
    'keyword_matching_time': {},
    'query_matching_time': {},
    'candidate_networks_time': {},
}

for execution_time in performance_results:
    for qsname in cn_results:
        performance_results[execution_time][qsname] = (0,0)
        
        for Q in cn_results[qsname]:
            time_sum, time_count = performance_results[execution_time][qsname]
            time_sum += cn_results[qsname][Q][execution_time]
            time_count += 1
            performance_results[execution_time][qsname] = (time_sum, time_count)

        time_sum, time_count = performance_results[execution_time][qsname]
        performance_results[execution_time][qsname] = time_sum / time_count
```

```python
performance_results.keys()
```

```python
performance_results
```

```python
performance_results ={'keyword_matching_time': {'IMDB': 0.17123486743934335,
  'IMDB-DI': 0.16311269146011909,
  'MONDIAL': 0.20620049766633505,
  'MONDIAL-DI': 0.24691154144416538},
 'query_matching_time': {'IMDB': 0.17705535780027276,
  'IMDB-DI': 0.20345495622023008,
  'MONDIAL': 0.003033677511686821,
  'MONDIAL-DI': 0.008975450955525351},
 'candidate_networks_time': {'IMDB': 0.061031114980432906,
  'IMDB-DI': 0.06724109777947888,
  'MONDIAL': 0.575027877888513,
  'MONDIAL-DI': 1.3994221708895769}}


label_order = ['IMDB', 'IMDB-DI', 'MONDIAL', 'MONDIAL-DI']

N = len(label_order)
ind = np.arange(N)

keyword_matching_time = [performance_results['keyword_matching_time'][label] for label in label_order]
query_matching_time = [performance_results['query_matching_time'][label] for label in label_order]
candidate_networks_time = [performance_results['candidate_networks_time'][label] for label in label_order]

width = 0.4

p1 = plt.bar(ind, keyword_matching_time, width, color='r')
p2 = plt.bar(ind, query_matching_time, width, color='b',
             bottom=keyword_matching_time )
p3 = plt.bar(ind, candidate_networks_time, width, color='g',
             bottom=np.array(keyword_matching_time)+np.array(query_matching_time))

plt.xticks(ind, label_order)
plt.legend((p1[0], p2[0],p3[0]), ('Keyword Match', 'Query Match','Candidate Network'))

plt.ylabel('Execution Time(s)')
plt.show()
```

### Query Matches Ranking

```python
imdb_coffman_original_qsconfig.golden_query_matches_directory
```

### get_golden_query_matches

```python
def get_golden_query_matches(golden_query_matches_directory):
    golden_cns = OrderedDict()
    for filename in sorted(glob.glob('{}/*.txt'.format(golden_query_matches_directory.rstrip('/')))):
        with open(filename) as f:
            json_serializable = json.load(f)
            golden_cns[tuple(json_serializable['query'])] = \
                {KeywordMatch.from_json_serializable(js) for js in json_serializable['query_match']}
    return golden_cns
```

```python
get_golden_query_matches(imdb_coffman_original_qsconfig.golden_query_matches_directory)
```

```python
# config = mondial_coffman_ci_qsconfig
# golden_cns=get_golden_candidate_networks(config.golden_candidate_networks_directory)
# golden_qms={ Q:cn.non_free_keyword_matches()
#             for Q,cn in golden_cns.items()}
```

### generate_golden_qm_files

```python
def generate_golden_qm_files(golden_qms,queryset,golden_query_matches_directory):         
    for i,Q in enumerate(queryset):
        
        filename = "{}/{:0>3d}.txt".format(golden_query_matches_directory.rstrip('/'),i+1) 
        
        if Q not in golden_qms:
                print("File {} not created because there is\n no golden standard set for the query\n {}".format(filename,Q))
                continue
        
        with open(filename,mode='w') as f:
            json_serializable = {'query_match':[keyword_match.to_json_serializable() 
                                                for keyword_match in golden_qms[Q]],
                                 'query':Q,} 
            f.write(json.dumps(json_serializable,indent=4))
```

```python
# generate_golden_qm_files(golden_qms,get_query_sets(config.queryset_filename),
#                          config.golden_query_matches_directory)
```

### get_golden_candidate_networks

```python
def get_golden_candidate_networks(golden_candidate_networks_directory):
    golden_cns = OrderedDict()
    for filename in sorted(glob.glob('{}/*.txt'.format(golden_candidate_networks_directory.rstrip('/')))):
        with open(filename) as f:
            json_serializable = json.load(f)
            golden_cns[tuple(json_serializable['query'])] = \
                CandidateNetwork.from_json_serializable(json_serializable['candidate_network'])
    return golden_cns
```

```python
golden_cns = get_golden_candidate_networks(
    mondial_coffman_original_qsconfig.golden_candidate_networks_directory)

golden_cns[('serb', 'europe')]= CandidateNetwork.from_str('''CONTINENT.v(name{europe})
	<ENCOMPASSES
		>COUNTRY
			<ETHNIC_GROUP.v(name{serb})''')


golden_cns[('uzbek', 'asia')]= CandidateNetwork.from_str('''CONTINENT.v(name{asia})
	<ENCOMPASSES
		>COUNTRY
			<ETHNIC_GROUP.v(name{uzbek})''')

for i, Q in enumerate(golden_cns):
    cn = golden_cns[Q]
    print('{}a Q:\n{}\n{}a CN:\n{}\n'.format(i + 1, ' '.join(Q), i + 1, cn))
```

```python
list(golden_cns[('mongolia', 'china')].non_free_keyword_matches())
```

```python
def export_results_json (cns_results):
    for qsconfig in cns_results:
        data = []
        for i, keyword_query in enumerate(
                get_query_sets(qsconfig.queryset_filename)):
            if keyword_query == '':
                continue

            obj = {
                'keyword_query': keyword_query,
                'query_matches':[
                    [km.to_json_serializable() for km in match]
                    for (match, score) in cns_results[qsconfig][keyword_query]['query_matches']
                ],
                'candidate_networks': [cn.to_json_serializable()
                                       for (cn, score) in cns_results[qsconfig][keyword_query]['candidate_networks']
                ],
            }

            data.append(obj)

        with open('results/{}.json'.format(qsconfig.name), 'w') as outfile:
            json.dump(data, outfile)

```

```python
QSCONFIGS_TO_PROCESS
```

```python
qsconfig = imdb_coffman_original_qsconfig

golden_cns = get_golden_candidate_networks(
    qsconfig.golden_candidate_networks_directory)
```

```python
qb=golden_cns.keys()
```

```python
qa={tuple(tokenize_string(keyword_query)) for keyword_query in get_query_sets(qsconfig.queryset_filename)}
```

```python
print(qa - qb)
```

```python
print(qb - qa)
```

```python
qsconfig.queryset_filename
```

```python


data = []
for keyword_query in get_query_sets('querysets/queryset_imdb_coffman_revised.txt'):

    Q = tuple(tokenize_string(keyword_query))
    candidate_network = golden_cns[Q]    
    serializable_query_match = [keyword_query.to_json_serializable()
                   for keyword_query in candidate_network.non_free_keyword_matches()]

    obj = {
        'keyword_query': keyword_query,
        'query_matches':[ serializable_query_match ],
        'candidate_networks': [candidate_network.to_json_serializable()],
    }

    data.append(obj)
with open('new_golden_standards/{}.json'.format(qsconfig.name), 'w') as outfile:
    json.dump(data, outfile)
```

```python

```

```python
if STEP_BY_STEP:
    golden_cns=get_golden_candidate_networks(DEFAULT_QSCONFIG.golden_candidate_networks_directory)
    
    for i,Q in enumerate(get_query_sets(DEFAULT_QSCONFIG.queryset_filename)):
        if Q not in golden_cns:
            continue
    
        print('{} Q: {}\nCN: {}\n'.format(i+1,Q,golden_cns[Q]))
```

```python
if STEP_BY_STEP:
    golden_cns=get_golden_candidate_networks(DEFAULT_QSCONFIG.golden_candidate_networks_directory)
    queryset =get_query_sets(DEFAULT_QSCONFIG.queryset_filename)
        
    
    for i,Q in enumerate(golden_cns):
        if  Q in border_queries:
            continue
    
        print('{} Q: {}\nCN: {}\n'.format(i+1,Q,golden_cns[Q]))
```

```python
golden_cns[('serb', 'europe')]= CandidateNetwork.from_str('''CONTINENT.v(name{europe})
	<ENCOMPASSES
		>COUNTRY
			<ETHNIC_GROUP.v(name{serb})''')


golden_cns[('uzbek', 'asia')]= CandidateNetwork.from_str('''CONTINENT.v(name{asia})
	<ENCOMPASSES
		>COUNTRY
			<ETHNIC_GROUP.v(name{uzbek})''')
```

```python
print(golden_cns[('serb', 'europe')])
print(golden_cns[('uzbek', 'asia')])
```

```python
golden_cns
```

### set_golden_candidate_networks

```python
def set_golden_candidate_networks(queryset,result,golden_cns=None):
    from IPython.display import clear_output
    if golden_cns is None:
        golden_cns = OrderedDict()
    for i,Q in enumerate(queryset):
        if golden_cns.setdefault(Q,None) is not None:
            continue
        
        answer = None
            
        if Q not in result or len(result[Q]['candidate_networks']) == 0:
            answer = input('The system did not generate any Candidate Network for the Query Q given.\nQ:\n{}\nWould you like to write the correct relevant CN for it?(type y if yes)\n'.format(Q))
            clear_output()
            if answer == 'y':
                candidate_network = CandidateNetwork.from_str(input("Write the golden Candidate Network:"))
                golden_cns[Q]=candidate_network
            continue
        
        for candidate_network,score in result[Q]['candidate_networks']:
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
# gs = set_golden_candidate_networks(queryset,necns_results[DEFAULT_QSCONFIG],golden_cns)
```

```python
# gs = set_golden_candidate_networks(get_query_sets(imdb_coffman_clear_intents_config.queryset_filename),
#                                    nepr_imdb_clear_itent_results,
#                                    golden_cns)
```

### generate_golden_cn_files

```python
def generate_golden_cn_files(golden_candidate_networks_directory,queryset,golden_standards):          
    for i,Q in enumerate(queryset):
        
        filename = "{}/{:0>3d}.txt".format(golden_candidate_networks_directory.rstrip('/'),i+1) 
        
        if Q not in golden_standards:
                print("File {} not created because there is\n no golden standard set for the query\n {}".format(filename,Q))
                continue
        
        with open(filename,mode='w') as f:
            json_serializable = {'candidate_network':golden_standards[Q].to_json_serializable(),
                                 'query':Q,} 
```

```python
#             if Q in [('serb', 'europe'),('uzbek', 'asia')]:
#                 print(filename)
#                 print(Q)
#                 print(golden_standards[Q])
#                 pp(json_serializable)
            f.write(json.dumps(json_serializable,indent=4))
```


```python
# generate_golden_cn_files(config.golden_candidate_networks_directory,
#                          get_query_sets(config.queryset_filename),
#                          golden_cns)
```

```python
for Q in gs:
    print(Q,'\n',gs[Q],'\n\n')
```

```python
generate_golden_cn_files(imdb_coffman_clear_intents_config.golden_candidate_networks_directory,
                         get_query_sets(imdb_coffman_clear_intents_config.queryset_filename),
                         gs,
                        )
```

```python
final_metrics = {}
config = list(CONFIGS_TO_PREPROCESS)[0]
```

```python
print(get_sql_from_cn(get_schema_graph(config.conn_string),
                [cn[0] for cn in non_empty_param_results[config][('harrison', 'ford', 'george', 'lucas')]['candidate_networks']][0]
               ))
```

```python
[cn[0] for cn in non_empty_param_results[config][('harrison', 'ford', 'george', 'lucas')]['candidate_networks']][0]
```

```python
normal_results[config][('harrison', 'ford', 'george', 'lucas')]['candidate_networks']
```

```python
results = normal_results[config]
# get golden_standards
golden_qms = get_golden_query_matches(config.golden_query_matches_directory)
golden_cns = get_golden_candidate_networks(config.golden_candidate_networks_directory)

result_query_matches = OrderedDict( (Q,[qm for qm,_,_,_ in results[Q]['query_matches']]) 
                               for Q in results )

result_candidate_networks = OrderedDict((Q,[cn for cn,_,_,_ in results[Q]['candidate_networks']])
                                for Q in results)  
```

```python
positions_query_matches = get_relevant_positions(result_query_matches,golden_qms)
positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns) 

positions_non_empty_candidate_networks = get_relevant_positions(result_candidate_networks,
                                                  golden_cns,
                                                  index_step_func=factory_index_step_non_empty_cn(config.conn_string)
                                                 )
```

```python
positions_candidate_networks
```

```python
final_metrics[config]={}
final_metrics[config]['query_matches']=metrics(positions_query_matches.values())
final_metrics[config]['candidate_networks']=metrics(positions_candidate_networks.values())
final_metrics[config]['non_empty_candidate_networks']=metrics(positions_non_empty_candidate_networks.values())
```

```python
next(iter(final_metrics.values()))
```

```python
border_queries = [('slovakia', 'hungary'),
 ('mongolia', 'china'),
 ('niger', 'algeria'),
 ('kuwait', 'saudi', 'arabia'),
 ('lebanon', 'syria')]
```

```python
def evaluation(results,golden_qms,golden_cns):
    result_query_matches = OrderedDict( (Q,[qm for qm,_,_,_ in results[Q]['query_matches']]) 
                                   for Q in results )
    
    result_candidate_networks = OrderedDict((Q,[cn for cn,_,_,_ in results[Q]['candidate_networks']])
                                        for Q in results)  
```

```python
def get_relevant_positions(results,golden_stantards, index_step_func = None):
    relevant_positions = OrderedDict()
    for Q,golden_standard in golden_stantards.items():
        idx = 0
        found = False
        
        if Q not in results:
            continue
            
        for element in results[Q]:
            if index_step_func is None or index_step_func(element):
                idx+=1
                
            if element==golden_standard:
                found=True
                break
        
        if not found:
            idx = -1
            
        relevant_positions[Q]=(idx)
    return relevant_positions
```

```python
positions_query_matches = get_relevant_positions(result_query_matches,golden_qms)
```

```python
positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns)
```

```python
def factory_index_step_non_empty_cn(conn_string):
    
    G = get_schema_graph(conn_string)
    
    def index_step_non_empty_cn(candidate_network):
        return exec_sql(conn_string,
                        get_sql_from_cn(G,candidate_network,rowslimit=1), 
                        show_results=False,)
    
    return index_step_non_empty_cn

# positions_non_empty_candidate_networks = get_relevant_positions(result_candidate_networks,
#                                                       golden_cns,
#                                                       index_step_func=factory_index_step_non_empty_cn(DEFAULT_CONFIG.conn_string)
#                                                      )
```

```python
def mrr(position_list):
    if len(position_list) == 0:
        return 0
    return sum(1/p for p in position_list if p != -1)/len(position_list)

def precision_at(position_list,threshold = 3):
    if len(position_list) == 0:
        return 0
    return len([p for p in position_list if p != -1 and p<=threshold])/len(position_list)
           


```

```python
def metrics(position_list, max_position_k=4):
    result = {}
    result['MRR']=mrr(position_list)
    result['LRR']=max(position_list) if len(position_list)>0 else -1
    
    for i in range(max_position_k):
        result['P@{}'.format(i+1)]=precision_at(position_list,threshold=i+1)
    return result
```

## FINAL METRICS

```python
skip_queries = ['poland cape verde organization',
 'saint kitts cambodia',
 'marshall islands grenadines organization',
 'czech republic cote divoire organization',
 'panama oman',
 'iceland mali',
 'guyana sierra leone',
 'mauritius india',
 'vanuatu afghanistan',
 'libya australia',]
        
queryset = [tuple(tokenize_string(keyword_query))
            for keyword_query in get_query_sets(qsconfig.queryset_filename) 
            if keyword_query not in skip_queries]

len(queryset)
```

```python
final_metrics = {'query_matches':{},
                 'candidate_networks':{},}

for qsconfig in QSCONFIGS_TO_PROCESS:
    
    final_metrics['candidate_networks'][qsconfig]={}
    
    for cn_approach,experiments in [('cn',cns_results)]:    
        
        skip_queries = ['poland cape verde organization',
                         'saint kitts cambodia',
                         'marshall islands grenadines organization',
                         'czech republic cote divoire organization',
                         'panama oman',
                         'iceland mali',
                         'guyana sierra leone',
                         'mauritius india',
                         'vanuatu afghanistan',
                         'libya australia',]
#         tuple(tokenize_string(keyword_query))
        queryset = [keyword_query
            for keyword_query in get_query_sets(qsconfig.queryset_filename) 
            if keyword_query not in skip_queries]
        
        
        if qsconfig not in final_metrics['query_matches']:
            golden_qms = get_golden_query_matches(qsconfig.golden_query_matches_directory)
            
            result_query_matches = OrderedDict( (tuple(tokenize_string(Q)), [qm for qm,score in experiments[qsconfig][Q]['query_matches']]) 
                                       for Q in experiments[qsconfig] if Q in queryset)
            
            positions_query_matches = get_relevant_positions(result_query_matches,golden_qms)

            final_metrics['query_matches'][qsconfig]=metrics(positions_query_matches.values(),
                                                             max_position_k=10)

        golden_cns = get_golden_candidate_networks(qsconfig.golden_candidate_networks_directory)
        
        result_candidate_networks = OrderedDict( (tuple(tokenize_string(Q)), [cn for cn,score in experiments[qsconfig][Q]['candidate_networks']] )
                                                for Q in experiments[qsconfig] if Q in queryset)  

        positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns) 
        
        final_metrics['candidate_networks'][qsconfig][cn_approach]=metrics(positions_candidate_networks.values())
        
    

print('FINISHED')
```

```python
golden_cns = get_golden_candidate_networks(qsconfig.golden_candidate_networks_directory)
        
result_candidate_networks = OrderedDict( (tuple(tokenize_string(Q)), [cn for cn,score in experiments[qsconfig][Q]['candidate_networks']] )
                                        for Q in experiments[qsconfig] if Q in queryset)  

positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns) 
```

```python
positions_candidate_networks
```

```python
final_metrics
```

```python
final_metrics = {'query_matches':{},
                 'candidate_networks':{},}

for qsconfig in QSCONFIGS_TO_PROCESS:
    
    final_metrics['candidate_networks'][qsconfig]={}
    
    for cn_approach,experiments in [('necn',necns_results),
                        ('cn',cns_results)]:    
        
        border_queries = [('slovakia', 'hungary'),
                         ('mongolia', 'china'),
                         ('niger', 'algeria'),
                         ('kuwait', 'saudi', 'arabia'),
                         ('lebanon', 'syria')]
        
        queryset = [Q 
                    for Q in get_query_sets(qsconfig.queryset_filename) 
                    if Q not in border_queries]
        
        
        if qsconfig not in final_metrics['query_matches']:
            golden_qms = get_golden_query_matches(qsconfig.golden_query_matches_directory)
            
            result_query_matches = OrderedDict( (Q, [qm for qm,score in experiments[qsconfig][Q]['query_matches']]) 
                                       for Q in experiments[qsconfig] if Q in queryset)
            
            positions_query_matches = get_relevant_positions(result_query_matches,golden_qms)

            final_metrics['query_matches'][qsconfig]=metrics(positions_query_matches.values(),
                                                             max_position_k=10)

        golden_cns = get_golden_candidate_networks(qsconfig.golden_candidate_networks_directory)
        
        result_candidate_networks = OrderedDict( (Q, [cn for cn,score in experiments[qsconfig][Q]['candidate_networks']] )
                                                for Q in experiments[qsconfig] if Q in queryset)  

        positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns) 
        
        final_metrics['candidate_networks'][qsconfig][cn_approach]=metrics(positions_candidate_networks.values())
        
        
        if cn_approach == 'cn':
#             print(qsconfig)
#             print(Q)
#             print(result_candidate_networks)
            
            positions_non_empty_candidate_networks = get_relevant_positions(result_candidate_networks,
                                                              golden_cns,
                                                              index_step_func=factory_index_step_non_empty_cn(qsconfig.dsconfig.conn_string)
                                                             )
            
            final_metrics['candidate_networks'][qsconfig]['cnne']=metrics(positions_non_empty_candidate_networks.values())
            

print('FINISHED')
```

```python
cn_approaches = ['cn','cnne','necn']
metrics_names = ['MRR', 'P@1', 'P@2', 'P@3', 'P@4']

formated_cn_exp = {}
for metric in metrics_names:
    formated_cn_exp[metric] = {}
    for qsconfig in final_metrics['candidate_networks']: 
        formated_cn_exp[metric][qsconfig.name] = [final_metrics['candidate_networks'][qsconfig][cn_approach][metric] for cn_approach in cn_approaches]

print(formated_cn_exp)
```

```python
pp(final_metrics['query_matches'])
```

## Verificar quais consultas são diferentes no MONDIAL E MONDIAL-DI

```python
final_metrics = {'query_matches':{},
                 'candidate_networks':{},}

aux =  {'query_matches':{},
                 'candidate_networks':{},}
for qsconfig in [mondial_coffman_original_qsconfig,
                 mondial_coffman_ci_qsconfig]:
    
    final_metrics['candidate_networks'][qsconfig]={}
    aux['candidate_networks'][qsconfig]={}
    
    for cn_approach,experiments in [('necn',necns_results),
                        ('cn',cns_results)]:    
        
        border_queries = [('slovakia', 'hungary'),
                         ('mongolia', 'china'),
                         ('niger', 'algeria'),
                         ('kuwait', 'saudi', 'arabia'),
                         ('lebanon', 'syria')]
        
        queryset = [Q 
                    for Q in get_query_sets(qsconfig.queryset_filename) 
                    if Q not in border_queries]
        
        
        if qsconfig not in final_metrics['query_matches']:
            golden_qms = get_golden_query_matches(qsconfig.golden_query_matches_directory)
            
            result_query_matches = OrderedDict( (Q, [qm for qm,score in experiments[qsconfig][Q]['query_matches']]) 
                                       for Q in experiments[qsconfig] if Q in queryset)
            
            positions_query_matches = get_relevant_positions(result_query_matches,golden_qms)

            aux['query_matches'][qsconfig]=positions_query_matches
            
            final_metrics['query_matches'][qsconfig]=metrics(positions_query_matches.values(),
                                                             max_position_k=10)

        golden_cns = get_golden_candidate_networks(qsconfig.golden_candidate_networks_directory)
        
        result_candidate_networks = OrderedDict( (Q, [cn for cn,score in experiments[qsconfig][Q]['candidate_networks']] )
                                                for Q in experiments[qsconfig] if Q in queryset)  

        positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns)
        
        aux['candidate_networks'][qsconfig][cn_approach]=positions_candidate_networks
        
        final_metrics['candidate_networks'][qsconfig][cn_approach]=metrics(positions_candidate_networks.values())
        
        
        if cn_approach == 'cn':
            positions_non_empty_candidate_networks = get_relevant_positions(result_candidate_networks,
                                                              golden_cns,
                                                              index_step_func=factory_index_step_non_empty_cn(qsconfig.dsconfig.conn_string)
                                                             )
            
            aux['candidate_networks'][qsconfig]['cnne']=positions_candidate_networks
            final_metrics['candidate_networks'][qsconfig]['cnne']=metrics(positions_non_empty_candidate_networks.values())
            

```


```python
not_found = {'query_matches':{},
                 'candidate_networks':{},}

exp_name = 'query_matches'

for exp_name in not_found:
    for qsconfig in aux[exp_name]:
        
        if exp_name == 'query_matches':
        
            for Q in aux[exp_name][qsconfig]:
                if aux[exp_name][qsconfig][Q]==-1:                
                    not_found[exp_name].setdefault(Q,[])
                    not_found[exp_name][Q].append(qsconfig.name)
                    
        if exp_name == 'candidate_networks': 
            for cn_approach in cn_approaches:
                for Q in aux[exp_name][qsconfig][cn_approach]:
                    if aux[exp_name][qsconfig][cn_approach][Q]==-1:
                        not_found[exp_name].setdefault(Q,{})
                        not_found[exp_name][Q].setdefault(qsconfig.name,[])
                        not_found[exp_name][Q][qsconfig.name].append(cn_approach)
            

            

```


```python
print('QUERY MATCHES\n')
for Q in sorted(not_found['query_matches']):
    print('Q:{}\nNot Found in:{}\n'.format(Q,not_found[exp_name][Q]))

print('CANDIDATE NETWORKS\n')
for Q in sorted(not_found['candidate_networks']):
    print('Q:{}\nNot Found in:{}\n'.format(Q,not_found[exp_name][Q]))
```

## Performance Experiments

```python
ir=0
performance_results = {}
for qsconfig in QSCONFIGS_TO_PROCESS:
        
    print('{0}\n{0}\nCONFIG: {1}\n'.format('='*80,qsconfig))       
    performance_results[qsconfig] = keyword_search(qsconfig,
                                            word_embeddings_model,
                                            **{
                                                'VKMGen_kwargs':{
                                                    'ignored_tables':['cast_info']
                                                },
                                                'SKMGen_kwargs':{
                                                    'keyword_matches_to_ignore':{KeywordMatch.from_str('CHAR_NAME.s(name{name})')},
                                                },
                                                'MatchCN_kwargs':{
                                                   'CNGraphGen_kwargs':{
                                                       'directed_neighbor_sorting_function':
                                                       worst_neighbors_sorting,
                                                       'non_empty_only':False,
                                                   }
                                               },
                                            })
```

```python
performance_results
```

```python
performance_metrics = {}
for qsconfig in performance_results:
    qsname=qsconfig.name
    
    time_list = [performance_results[qsconfig][Q]['execution_time'] for Q in performance_results[qsconfig]]
        
    performance_metrics[qsname] = time_list
```

```python
ordered_performance_results = {}
for qsconfig in performance_results:
    qsname = qsconfig.name
    
    ordered_performance_results[qsname] = OrderedDict(
        sorted(
            [(performance_results[qsconfig][Q]['execution_time'],Q) 
             for Q in performance_results[qsconfig]],
           reverse=True))
```

```python
pp(ordered_performance_results)
```

# Experimento top-k-cn-per-qm

```python
QSCONFIGS_TO_PROCESS
```

```python
border_queries = [('slovakia', 'hungary'),
                         ('mongolia', 'china'),
                         ('niger', 'algeria'),
                         ('kuwait', 'saudi', 'arabia'),
                         ('lebanon', 'syria')]


# cn_per_qm_results = {}

start_time = timer()

for qsconfig in QSCONFIGS_TO_PROCESS:
    qsname = qsconfig.name
    cn_per_qm_results[qsname] = {}
    
    print(qsname)

    for cn_approach,experiments in [('cn',cns_results),
                                    ('necn',necns_results)]:    
        
        print(cn_approach)
        print(str(datetime.timedelta(seconds=elapsed_time)))
        
        
        cn_per_qm_results[qsname][cn_approach] = {}
        
        golden_qms = get_golden_query_matches(qsconfig.golden_query_matches_directory)
        golden_cns = get_golden_candidate_networks(qsconfig.golden_candidate_networks_directory)
        
        border_queries = [('slovakia', 'hungary'),
                         ('mongolia', 'china'),
                         ('niger', 'algeria'),
                         ('kuwait', 'saudi', 'arabia'),
                         ('lebanon', 'syria')]
        
        dsconfig = qsconfig.dsconfig
        G = get_schema_graph(dsconfig.conn_string)
        
        queryset = [Q 
                    for Q in get_query_sets(qsconfig.queryset_filename) 
                    if Q not in border_queries]
        
        for Q in queryset:
            print(Q)
            print(str(datetime.timedelta(seconds=elapsed_time)))
            
#             if Q != ('cameroon', 'economy'):
#                 continue
            
#             print('Q:',Q)
#             print('Golden CN:\n',golden_cns[Q])
            
            QM = golden_qms[Q]
            
            non_empty_only = (cn_approach == 'necn')
            
            cns_elapsed_time = []
            
            CNGraphGen_start_time = timer()
            Cns = CNGraphGen(dsconfig.conn_string,
                 dsconfig.schema_index,
                 G,QM,
                 **{
                    'topk_cns_per_qm':10,
                    'directed_neighbor_sorting_function':
                    worst_neighbors_sorting,
                    'non_empty_only':non_empty_only,
                     'gns_elapsed_time':cns_elapsed_time,
                 }
                )
            CNGraphGen_end_time = timer()
            
            cnkm_total_elapsed_time = CNGraphGen_start_time-CNGraphGen_end_time
            
            cns_elapsed_time = cns_elapsed_time + [cnkm_total_elapsed_time]*(10-len(cns_elapsed_time))
                            
            
            if len(Cns)==0:
                rank=-1
            else:
                if golden_cns[Q] in Cns:
                    rank = len(Cns)
        
#             print('Rank:',rank)
#             print('CNs:')
#             for cn in Cns:
#                 print(cn)
#             print('='*80)
            
            cn_per_qm_results[qsname][cn_approach][Q] = (rank,cns_elapsed_time, Cns)
        
print('FINISHED')        
```

```python
aggregated = {}
for qsname in cn_per_qm_results:
    
    aggregated[qsname]={}
    
    for cn_approach in cn_per_qm_results[qsname]:    
        
        aggregated[qsname][cn_approach]=[]
        
        for Q in cn_per_qm_results[qsname][cn_approach]:
            
            rank,elapsed_time,generated_cns = cn_per_qm_results[qsname][cn_approach][Q]
            
            aggregated[qsname][cn_approach].append(rank)        
        
               
        position_list = aggregated[qsname][cn_approach]
        
        metric_hash = metrics(position_list, max_position_k=10)
        precision_at_k_list = []
        for i in range(10):
            precision_at_k_list.append(metric_hash['P@{}'.format(i+1)])   
            del metric_hash['P@{}'.format(i+1)]
        metric_hash['P@K'] = precision_at_k_list
        
        aggregated[qsname][cn_approach] = metric_hash
```

```python
print(aggregated)
```

```python
cn_per_qm_results['IMDB']['cn'][('angelina', 'jolie')]
```

```python
aggregated = {}
for qsname in cn_per_qm_results:
    
    aggregated[qsname]={}
    
    for cn_approach in cn_per_qm_results[qsname]:    
        
        aggregated[qsname][cn_approach]={'AVG':[],'MAX':[]}
        
        for i in range(10):        
            position_list=[]
            
            for Q in cn_per_qm_results[qsname][cn_approach]:

                rank,elapsed_time,generated_cns = cn_per_qm_results[qsname][cn_approach][Q]

                position_list.append(elapsed_time[i])        
                    
            aggregated[qsname][cn_approach]['AVG'].append(sum(position_list)/len(position_list))
            aggregated[qsname][cn_approach]['MAX'].append(max(position_list))
```

```python
print(aggregated)
```

```python
p_at_cns_per_qm= {'IMDB': {'cn': [0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          'necn': [0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
 'IMDB-DI': {'cn': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
             'necn': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
 'MONDIAL': {'cn': [0.7555555555555555,
                    0.7777777777777778,
                    0.7777777777777778,
                    0.7777777777777778,
                    0.7777777777777778,
                    0.7777777777777778,
                    0.7777777777777778,
                    0.7777777777777778,
                    0.8444444444444444,
                    1.0],
             'necn': [0.9777777777777777,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0]},
 'MONDIAL-DI': {'cn': [0.7555555555555555,
                       0.7777777777777778,
                       0.7777777777777778,
                       0.7777777777777778,
                       0.7777777777777778,
                       0.7777777777777778,
                       0.7777777777777778,
                       0.9111111111111111,
                       1.0,
                       1.0],
                'necn': [0.9777777777777777,
                         1.0,
                         1.0,
                         1.0,
                         1.0,
                         1.0,
                         1.0,
                         1.0,
                         1.0,
                         1.0]}}
```

```python
aggregated = {}
for qsname in cn_per_qm_results:
    
    aggregated[qsname]={}
    
    for cn_approach in cn_per_qm_results[qsname]:    
        
        aggregated[qsname][cn_approach]=[]
        
        for Q in cn_per_qm_results[qsname][cn_approach]:
            
            rank,elapsed_time,generated_cns = cn_per_qm_results[qsname][cn_approach][Q]
            
            aggregated[qsname][cn_approach].append(elapsed_time)        
            
        elapserd_time_list = aggregated[qsname][cn_approach]
        aggregated[qsname][cn_approach] = max(elapserd_time_list)
```

```python
avg_time_cn_per_qm = {'IMDB': {'cn': 0.011240729689598083, 'necn': 0.38074914962053297},
 'IMDB-DI': {'cn': 0.012587687522172928, 'necn': 0.28267604678869246},
 'MONDIAL': {'cn': 9.995028199421036, 'necn': 10.284027291834354},
 'MONDIAL-DI': {'cn': 2.609588401019573, 'necn': 2.7846566166314815}}


max_time_cn_per_qm={'IMDB': {'cn': 0.13095925375819206, 'necn': 1.9354509562253952},
 'IMDB-DI': {'cn': 0.1568550541996956, 'necn': 0.8087131939828396},
 'MONDIAL': {'cn': 61.13658142834902, 'necn': 62.68183133006096},
 'MONDIAL-DI': {'cn': 14.79405715316534, 'necn': 15.299424204975367}}
```
