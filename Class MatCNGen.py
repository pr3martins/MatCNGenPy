# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: md,ipynb,py
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

# +
from pprint import pprint as pp
import gc #garbage collector usado no create_inverted_index

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

from prettytable import PrettyTable # To print sql results


# -

class Configuration:
    def __init__(self,
                 dbname,user,password,
                 embedding_filename,
                 queryset_filename,
                 golden_candidate_networks_directory,
                 golden_query_matches_directory):
        self.dbname = dbname
        self.user = user
        self.password = password
        
        self.embedding_filename = embedding_filename
        self.queryset_filename = queryset_filename
        self.golden_candidate_networks_directory = golden_candidate_networks_directory
        self.golden_query_matches_directory = golden_query_matches_directory


mondial_config =  Configuration('mondial_coffman',
                                'imdb',
                                'imdb',
                                'word_embeddings/word2vec/GoogleNews-vectors-negative300.bin',
                                'querysets/queryset_mondial_coffman_original.txt',
                                'golden_candidate_networks/mondial_coffman',
                                'golden_query_matches/mondial_coffman'                                
                               )

imdb_coffman_config =  Configuration('imdb_subset_coffman',
                                'imdb',
                                'imdb',
                                'word_embeddings/word2vec/GoogleNews-vectors-negative300.bin',
                                'querysets/queryset_imdb_coffman_original.txt',
                                'golden_candidate_networks/imdb_coffman_revised',
                                'golden_query_matches/imdb_coffman_original'                                
                               )

imdb_ijs_config =  Configuration('imdb_ijs',
                                'imdb',
                                'imdb',
                                'word_embeddings/word2vec/GoogleNews-vectors-negative300.bin',
                                'querysets/queryset_imdb_martins_qualis.txt',
                                'golden_candidate_networks/imdb_ijs_martins',
                                'golden_query_matches/imdb_ijs_martins'                                
                               )

DEFAULT_CONFIG = mondial_config
STEP_BY_STEP = True
PREPROCESSING = False
CUSTOM_QUERY = None


def valid_schema_element(text,embmodel=set()): 
    if 'id' in text or 'index' in text or 'code' in text or 'nr' in text:
        return False
    return True    


def tokenize_string(text):     
    return [word.strip(string.punctuation)
            for word in text.lower().split() 
            if word not in stopwords.words('english') or word == 'will']
    return [word
            for word in text.translate({ord(c):' ' for c in string.punctuation if c!='_'}).lower().split() 
            if word not in stopwords.words('english') or word == 'will']


# # Preprocessing stage

def load_embeddings(config = None):
    if config is None:
        config=DEFAULT_CONFIG
    return KeyedVectors.load_word2vec_format(config.embedding_filename,
                                             binary=True, limit=500000)


if STEP_BY_STEP and PREPROCESSING:
    word_embeddings_model=load_embeddings()


# During the process of generating SQL queries, Lathe uses two data structures
# which are created in a **Preprocessing stage**: the Value Index and the Schema Index.
#
# The Value Index is an inverted index stores the occurrences of keyword in the database,
# indicating the relations and tuples a keyword appear and which attributes are mapped
# to that keyword. These occurrences are retrieved in the Query Matching phase. In
# addition, the Value Index is also used to calculate term frequencies for the Ranking of
# Query Matches. The Schema Index is an inverted index that stores information about
# the database schema and statics about ranking of attributes, which is also used in the
# Query Matches Ranking.

# ## Class BabelItemsIter

# +
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


# -

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
    
    def print_babel(self):
        print(self.__babel)


# ## Class WordHash

class WordHash(dict):      
        
    def __init__(self): 
        dict.__init__(self)
    
    def add_mapping(self,word,table,attribute,ctid):
        self.setdefault( word, (0, BabelHash() ) )                    
        self[word].setdefault(table , BabelHash() )       
        self[word][table].setdefault( attribute , [] ).append(ctid)        
        
    def get_mappings(self,word,table,attribute):
        return self[word][table][attribute]
    
    def get_IAF(self,key):
        return dict.__getitem__(self,key)[0]
    
    def set_IAF(self,key,IAF):

        old_IAF,old_value = dict.__getitem__(self,key)
        
        dict.__setitem__(self, key,  (IAF,old_value)  )
    
    def __getitem__(self,word):
        return dict.__getitem__(self,word)[1]
    
    def __setitem__(self,word,value): 
        old_IAF,old_value = dict.__getitem__(self,word)
        dict.__setitem__(self, word,  (old_IAF,value)  )


# ## Class DatabaseIter

class DatabaseIter:
    def __init__(self,embedding_model,config = None):
        if config is None:
            config=DEFAULT_CONFIG
        
        self.config=config   
        self.embedding_model=embedding_model

    def __iter__(self):
        with psycopg2.connect(dbname=self.config.dbname,
                              user=self.config.user,
                              password=self.config.password) as conn:
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
                                       
                    for i,row in enumerate(cur.fetchall()): 
                        ctid = row[0]
                        for col in range(1,len(row)):
                            column = cur.description[col][0]
                            for word in tokenize_string( str(row[col]) ):
                                yield table,ctid,column, word
                        
                        if i%100000==1:
                            print('*',end='')


# ## Create Inverted Index

def create_inverted_index(embedding_model,show_log=True):
    #Output: word_hash (Term Index) with this structure below
    #map['word'] = [ 'table': ( {column} , ['ctid'] ) ]

    '''
    The Term Index is built in a preprocessing step that scans only
    once all the relations over which the queries will be issued.
    '''

    
    wh = WordHash()
    ah = {}
    
    previous_table = None
    
    for table,ctid,column,word in DatabaseIter(embedding_model):        
        wh.add_mapping(word,table,column,ctid)
                
        ah.setdefault(table,{}).setdefault(column,{}).setdefault(word,1)
        ah[table][column][word]+=1
        
    for table in ah:
        for column in ah[table]:
            
            max_frequency = num_distinct_words = num_words = 0            
            for word, frequency in ah[table][column].items():
                
                num_distinct_words += 1
                
                num_words += frequency
                
                if frequency > max_frequency:
                    max_frequency = frequency
            
            norm = 0
            ah[table][column] = (norm,num_distinct_words,num_words,max_frequency)

    print ('\nINVERTED INDEX CREATED')
    gc.collect()
    return wh,ah

if STEP_BY_STEP and PREPROCESSING:
    (word_hash,attribute_hash) = create_inverted_index(word_embeddings_model)


# ## Processing IAF

def process_iaf(word_hash,attribute_hash):
    
    total_attributes = sum([len(attribute) for attribute in attribute_hash.values()])
    
    for (term, values) in word_hash.items():
        attributes_with_this_term = sum([len(attribute) for attribute in word_hash[term].values()])
        IAF = log1p(total_attributes/attributes_with_this_term)
        word_hash.set_IAF(term,IAF)        
        
    print('IAF PROCESSED')


if STEP_BY_STEP and PREPROCESSING:
    process_iaf(word_hash,attribute_hash)


# ## Processing Attribute Norms

def process_norms_of_attributes(word_hash,attribute_hash):    
    for word in word_hash:
        for table in word_hash[word]:
            for column, ctids in word_hash[word][table].items():
                   
                (prev_norm,num_distinct_words,num_words,max_frequency) = attribute_hash[table][column]

                IAF = word_hash.get_IAF(word)

                frequency = len(ctids)
                
                TF = frequency/max_frequency
                
                Norm = prev_norm + (TF*IAF)

                attribute_hash[table][column]=(Norm,num_distinct_words,num_words,max_frequency)
                
    print ('NORMS OF ATTRIBUTES PROCESSED')


if STEP_BY_STEP and PREPROCESSING:
    process_norms_of_attributes(word_hash,attribute_hash)


def pre_processing(config=None):
    if config is None:
        config = DEFAULT_CONFIG
    
    word_embeddings_model=load_embeddings(config)
    (word_hash,attribute_hash) = create_inverted_index(word_embeddings_model)
    process_iaf(word_hash,attribute_hash)
    process_norms_of_attributes(word_hash,attribute_hash)
    
    print('PRE-PROCESSING STAGE FINISHED')
    return (word_hash,attribute_hash,word_embeddings_model)


if not STEP_BY_STEP and PREPROCESSING:
    (word_hash,attribute_hash,word_embeddings_model) = pre_processing()


# ## Class Graph

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


# ## get_schema_graph

def get_schema_graph(config=None):
    if config is None:
        config=DEFAULT_CONFIG
    
    #Output: A Schema Graph G  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }
    
    G = Graph(has_edge_info=True)
    with psycopg2.connect(dbname=config.dbname,user=config.user,password=config.password) as conn:
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


if STEP_BY_STEP:
    G = get_schema_graph()  
    print(G)
    for direction,level,vertex in G.leveled_dfs_iter():
        print(level*'\t',direction,vertex)
    print([x for x in G.dfs_pair_iter(root_predecessor=True)])


# # Processing Stage

def get_query_sets(config=None):
    if config is None:
        config = DEFAULT_CONFIG
        
    QuerySet = []
    with open(config.queryset_filename,
              encoding='utf-8-sig') as f:
        for line in f.readlines():
            
            #The line bellow Remove words not in OLIVEIRA experiments
            #Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
            Q = tuple(tokenize_string(line))
            
            QuerySet.append(Q)
    return QuerySet


if STEP_BY_STEP:
    QuerySets = get_query_sets()
    if CUSTOM_QUERY is None:
        Q = QuerySets[0]
    else:
        Q = CUSTOM_QUERY
    print(Q)


# ## Keyword Matching

# ### Class KeywordMatch

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

kmx= KeywordMatch.from_str('CHARACTER.s(name{name}).v(name{scissorhands,edward},birthdate{1997})')
KeywordMatch.from_json(kmx.to_json())


# ### Value Filtering

# #### VKMGen

# +
def VKMGen(Q,word_hash):
    #Input:  A keyword query Q=[k1, k2, . . . , km]
    #Output: Set of non-free and non-empty tuple-sets Rq

    '''
    The tuple-set Rki contains the tuples of Ri that contain all
    terms of K and no other keywords from Q
    '''
    
    #Part 1: Find sets of tuples containing each keyword
    P = {}
    for keyword in Q:
        
        if keyword not in word_hash:
            continue
        
        for table in word_hash[keyword]:
            for (attribute,ctids) in word_hash[keyword][table].items():
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
# -

if STEP_BY_STEP:
    print('FINDING TUPLE-SETS')
    Rq = VKMGen(Q, word_hash)
    print(len(Rq),'TUPLE-SETS CREATED\n')
    pp(Rq)


# ### Schema Filtering

# #### Class Similarities

class Similarities:    
    def __init__(self, model, attribute_hash,schema_graph,**kwargs):
        self.use_path_sim=kwargs.get('use_path_sim',True)
        self.use_wup_sim=kwargs.get('use_wup_sim',True)
        self.use_jaccard_sim=kwargs.get('use_jaccard_sim',True)
        self.use_emb_sim=kwargs.get('use_emb_sim',False)
        self.use_emb10_sim=kwargs.get('use_emb10_sim',True)  
        self.emb10_sim_type=kwargs.get('emb10_sim_type','B')
        
        self.model = model
        self.attribute_hash = attribute_hash
        self.schema_graph = schema_graph

        
        #self.porter = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        if self.use_emb_sim or self.use_emb10_sim:
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

            if table in self.attribute_hash:
                for column in self.attribute_hash[table]:
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
        

if STEP_BY_STEP:
    similarities=Similarities(word_embeddings_model,
                              attribute_hash,
                              get_schema_graph(),
                              )


# #### SKMGen

def SKMGen(Q,attribute_hash,similarities,**kwargs):    
    threshold = kwargs.get('threshold',1)
    
    S = set()
    
    for keyword in Q:
        for table in attribute_hash:            
            for attribute in ['*']+list(attribute_hash[table].keys()):
                
                if(attribute=='id'):
                    continue
                
                sim = similarities.word_similarity(keyword,table,attribute)
                
                if sim >= threshold:
                    skm = KeywordMatch(table,schema_filter={attribute:{keyword}})
                    S.add(skm)
                    
    return S


if STEP_BY_STEP:    
    print('FINDING SCHEMA-SETS')        
    Sq = SKMGen(Q,attribute_hash,similarities)
    print(len(Sq),' SCHEMA-SETS CREATED\n')
    pp(Sq)


# ## Query Matching

# ### Minimal Cover

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


# ### Merging Schema Filters

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


# ### QMGen

def QMGen(Q,Rq,**kwargs):    
    max_qm_size = kwargs.get('max_qm_size',5)
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
                Mq.append(merged_qm)
                   
    return Mq 


if STEP_BY_STEP:
    print('GENERATING QUERY MATCHES')
    TMaxQM = 3
    Mq = QMGen(Q,Rq|Sq)
    print (len(Mq),'QUERY MATCHES CREATED\n')  


# ### QMRank

def QMRank(Q, Mq,word_hash,attribute_hash,similarities,**kwargs):  
    
    show_log = kwargs.get('show_log',False)
    
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

                (Norm,num_distinct_words,num_words,max_frequency) = attribute_hash[table][attribute]                
                wsum = 0


                if show_log:
                    print('Norm: {}\nMaxFrequency {}\n'.format(Norm,max_frequency))


                for term in value_words:    

                    IAF = word_hash.get_IAF(term)

                    frequency = len(word_hash.get_mappings(term,table,attribute))
                    TF = (frequency/max_frequency)
                    wsum = wsum + TF*IAF
                    if show_log:
                        print('- Term: {}\n  Frequency:{}\n  TF:{}\n  IAF:{}\n'.format(term,frequency,TF,IAF))

                    there_is_value_terms = True

                '''
                for i in range(len(Q)-1):
                    if Q[i] in value_words and Q[i+1] in value_words:
                        wsum = wsum * 3
                '''        
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
                
        Ranking.append( (M,score,value_score,schema_score) )
                            
    return sorted(Ranking,key=lambda x: x[1]/len(x[0]),reverse=True)
                

if STEP_BY_STEP:
    print('RANKING QUERY MATCHES')
    RankedMq = QMRank(Q,Mq,word_hash,attribute_hash,similarities)   
    
    top_k = 20
    num_pruned_qms = len(RankedMq)-top_k
    if num_pruned_qms>0:
        print(num_pruned_qms,' QUERY MATCHES SKIPPED (due to low score)')
    else:
        num_pruned_qms=0        
        
    for (j, (M,score,valuescore,schemascore) ) in enumerate(RankedMq[:top_k]):
        print(j+1,'ª QM')           

        print('Schema Score:',"%.8f" % schemascore,
            '\nValue Score: ',"%.20f" % valuescore,
            '\n|M|: ',"%02d (Não considerado para calcular o total score)" % len(M),
            '\nTotal Score: ',"%.8f" % score)
        pp(M)
        #print('\n----Details----\n')
        #QMRank(Q, [M],word_hash,attribute_hash,similarities,show_log=True)

        print('----------------------------------------------------------------------\n')


# ## Candidate Networks

# ### Class CandidateNetwork

class CandidateNetwork(Graph):
    def add_vertex(self, vertex, default_alias=True):
        if default_alias:
            vertex = (vertex, 't{}'.format(self.__len__()+1))
        return super().add_vertex(vertex)
        
    def keyword_matches(self):
        return {keyword_match for keyword_match,alias in self.vertices()}
    
    def non_free_keyword_matches(self):
        return {keyword_match for keyword_match,alias in self.vertices() if not keyword_match.is_free()}
            
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
   


cnx=CandidateNetwork.from_str('''TITLE.s(*{title})
	<CAST_INFO
		>NAME.v(name{bond,james})''')
CandidateNetwork.from_json(cnx.to_json())


# ### Generation and Ranking of CNs

# #### sum_norm_attributes

def sum_norm_attributes(directed_neighbor):
    direction,adj_table = directed_neighbor
    if adj_table not in attribute_hash:
        return 0
    return sum(Norm for (Norm,num_distinct_words,num_words,max_frequency) in attribute_hash[adj_table].values())


sorted([(sum(Norm for (Norm,num_distinct_words,num_words,max_frequency) in attribute_hash[table].values()),
  table) for table in attribute_hash],reverse=True)


# #### CNGraphGen

def CNGraphGen(QM,G,**kwargs):  
    
    max_cn_size = kwargs.get('max_cn_size',10)
    show_log = kwargs.get('show_log',False)
    directed_neighbor_sorting = kwargs.get('directed_neighbor_sorting',sum_norm_attributes)
    topk_cns_per_qm = kwargs.get('topk_cns_per_qm',2)
    
    if show_log:
        print('================================================================================\nSINGLE CN')
        print('Tmax ',TMax)
        print('FM')
        pp(QM)
    
    CN = CandidateNetwork()
    CN.add_vertex(next(iter(QM)))
    
    if len(QM)==1:
        return {CN}
    
    returned_cns = set()
    
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
                                new_CN not in returned_cns and
                                len(new_CN)<=max_cn_size and
                                new_CN.is_sound() and
                                len(list(new_CN.leaves())) <= len(QM)):
#                                 print('Adding ',adj_keyword_match,' to current CN')
                                if new_CN.minimal_cover(QM):
#                                     print('Found CN')
#                                     print(new_CN)
#                                     print('GENERATED THE FIRST ONE')
                                    if len(returned_cns)<topk_cns_per_qm:
                                        returned_cns.add(new_CN)
                                    
                                    if len(returned_cns)==topk_cns_per_qm:
                                        return returned_cns
                                elif len(new_CN)<max_cn_size:
#                                     print('Adding\n{}\n'.format(new_CN))
                                    F.append(new_CN)
                                    
                
                
                new_CN = copy.deepcopy(CN)
#                 print('FREE KEYWORD MATCHES')
                adj_keyword_match = KeywordMatch(adj_table)
                vertex_v = new_CN.add_vertex(adj_keyword_match)
                new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction)
                if (new_CN not in F and
                    new_CN not in returned_cns and
                    len(new_CN)<max_cn_size and
                    new_CN.is_sound() and 
                    len(list(new_CN.leaves())) <= len(QM)):
#                     print('Adding ',adj_keyword_match,' to current CN')
#                     print('Adding\n{}\n'.format(new_CN))
                    F.append(new_CN)
                        
    return returned_cns

if STEP_BY_STEP:
    max_cn_size=5
   
    (QM,score,valuescore,schemascore) = RankedMq[0]
    print('GENERATING CNs FOR QM:',QM)
    
    Cns = CNGraphGen(QM,G,max_cn_size=max_cn_size,topk_cns_per_qm=20)
    
    for j, Cn in enumerate(Cns):
        print(j+1,'ª CN',
              '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
              '\nTotal Score: ',"%.8f" % (score/len(Cn)))
        print(Cn)


# #### MatchCN

def MatchCN(attribute_hash,G,RankedMq,**kwargs):
    
    topk_cns = kwargs.get('topk_cns',20)
    show_log = kwargs.get('show_log',False)
    
    UnrankedCns = []    
    generated_cns=[]
    
    for i,(QM,score,valuescore,schemascore) in enumerate(RankedMq):
        if show_log:
            print('{}ª QM:\n{}\n'.format(i+1,QM))
        Cns = CNGraphGen(QM,G,**kwargs)
        if show_log:
            print('Cns:')
            pp(Cns)
        if len(UnrankedCns)>=topk_cns:
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


if STEP_BY_STEP:   
    print('GENERATING CANDIDATE NETWORKS')  
    RankedCns = MatchCN(attribute_hash,G,RankedMq,topk_cns_per_qm=0)
    print (len(RankedCns),'CANDIDATE NETWORKS CREATED AND RANKED\n')
    
    for (j, (Cn,score,valuescore,schemascore) ) in enumerate(RankedCns):
        print(j+1,'ª CN')
        print('Schema Score:',"%.8f" % schemascore,
              '\nValue Score: ',"%.8f" % valuescore,
              '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
              '\nTotal Score: ',"%.8f" % score)
        pp(Cn)


# ### Translation of CNs

# #### get_sql_from_cn

def get_sql_from_cn(G,Cn,**kwargs):
    
    show_evaluation_fields=kwargs.get('show_evaluation_fields',False)
    rows_limit=kwargs.get('rows_limit',1000)
    
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
                (prev_column,column) = G.get_edge_info(prev_keyword_match.table,keyword_match.table)
            else:
                (column,prev_column) = G.get_edge_info(keyword_match.table,prev_keyword_match.table)

            selected_tables.append('JOIN {} {} ON {}.{} = {}.{}'.format(keyword_match.table,
                                                                        alias,
                                                                        prev_alias,
                                                                        prev_column,
                                                                        alias,
                                                                        column ))
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

if STEP_BY_STEP:
    (Cn,score,valuescore,schemascore)= RankedCns[0]
    print(Cn)
    print(get_sql_from_cn(G,Cn,show_evaluation_fields=True))


# #### exec_sql

def exec_sql (SQL,**kwargs):
    #print('RELAVANCE OF SQL:\n')
    #print(SQL)
    config=kwargs.get('config',DEFAULT_CONFIG)
    show_results=kwargs.get('show_results',True)
        
    with psycopg2.connect(dbname=config.dbname,
                          user=config.user,
                          password=config.password) as conn:
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

exec_sql(get_sql_from_cn(G,Cn,rowslimit=1))


# ## Keyword Search

def keyword_search(word_hash,attribute_hash,word_embeddings_model,
                **kwargs
         ):
    show_log = kwargs.get('show_log',False)
    output_results = kwargs.get('output_results',OrderedDict())
 
    queryset = kwargs.get('output_results', get_query_sets())
    
    G = get_schema_graph()    
    
    similarity_kwargs = kwargs.get('similarity_kwargs',{})   
    similarities=Similarities(word_embeddings_model,
                              attribute_hash,
                              G,
                              **similarity_kwargs)
    
    
    
    SKMGen_kwargs = kwargs.get('SKMGen_kwargs',{})
    QMGen_kwargs  = kwargs.get('QMGen_kwargs',{})
    QMRank_kwargs = kwargs.get('QMRank_kwargs',{})
    MatchCN_kwargs = kwargs.get('MatchCN_kwargs',{})
    
    for (i,Q) in enumerate(queryset):
        print('{}ª QUERY: {}\n'.format(i+1,Q))
        
        if Q in output_results:
            print('QUERY SKIPPED')
            continue
            
        print('FINDING VALUE-KEYWORD MATCHES')
        Rq = VKMGen(Q, word_hash)
        print('{} VALUE-KEYWORD MATCHES CREATED\n'.format(len(Rq)))
        
        if show_log:
            pp(Rq)
            
        print('FINDING SCHEMA-KEYWORD MATCHES')        
        Sq = SKMGen(Q,attribute_hash,similarities,**SKMGen_kwargs)
        print('{} VALUE-KEYWORD MATCHES CREATED\n'.format(len(Sq)))
        
        if show_log:
            pp(Sq)
            
        print('GENERATING QUERY MATCHES')
        Mq = QMGen(Q,Rq|Sq)
        print (len(Mq),'QUERY MATCHES CREATED\n')
        
        print('RANKING QUERY MATCHES')
        RankedMq = QMRank(Q,Mq,word_hash,attribute_hash,similarities,**QMRank_kwargs)   
                
        num_pruned_qms = len(RankedMq)-top_k
        
        if num_pruned_qms>0:
            print(num_pruned_qms,' QUERY MATCHES SKIPPED (due to low score)')
        else:
            num_pruned_qms=0        
        
        if show_log:
            for (j, (QM,score,valuescore,schemascore) ) in enumerate(RankedMq[:top_k]):
                print(i+1,'ª Q ',j+1,'ª QM')           
                
                print('Schema Score:',"%.8f" % schemascore,
                      '\nValue Score: ',"%.8f" % valuescore,
                      '\n|QM|: ',"%02d (Não considerado para calcular o total score)" % len(QM),
                      '\nTotal Score: ',"%.8f" % score)
                pp(QM)
                
                print('----------------------------------------------------------------------\n')
        
        
        print('GENERATING CANDIDATE NETWORKS')     
        RankedCns = MatchCN(attribute_hash,G,RankedMq,**MatchCN_kwargs)
        
        print (len(RankedCns),'CANDIDATE NETWORKS CREATED RANKED\n')
        
        if show_log:
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
        output_results[Q]={'query_matches':RankedMq[:top_k],
                      'candidate_networks':RankedCns}
    return output_results

results = keyword_search(word_hash,
                         attribute_hash,
                         word_embeddings_model,)


def get_golden_candidate_networks(path=None):
    if path is None:
        path = GOLDEN_CANDIDATE_NETWORKS_PATH
    golden_cns = OrderedDict()
    for filename in sorted(glob.glob('{}/*.txt'.format(GOLDEN_CANDIDATE_NETWORKS_PATH.rstrip('/')))):
        with open(filename) as f:
            json_serializable = json.load(f)
            golden_cns[tuple(json_serializable['query'])] = \
                CandidateNetwork.from_json_serializable(json_serializable['candidate_network'])
    return golden_cns



if STEP_BY_STEP:
    golden_cns=get_golden_candidate_networks()
    
    for i,Q in enumerate(get_query_sets(QUERYSETFILE)):
        if Q not in golden_cns:
            continue
    
        print('{} Q: {}\nCN: {}\n'.format(i+1,Q,golden_cns[Q]))

[(i+1,Q) for i,Q in enumerate(get_query_sets(QUERYSETFILE)) if Q not in golden_cns]


def set_golden_candidate_networks(result,golden_cns=None):
    from IPython.display import clear_output
    if golden_cns is None:
        golden_cns = OrderedDict()
    for i,Q in enumerate(get_query_sets(QUERYSETFILE)):
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

gs = set_golden_candidate_networks(results,golden_cns)


def generate_golden_cn_files(golden_standards,path=None):  
    if path is None:
        path = GOLDEN_CANDIDATE_NETWORKS_PATH
        
    for i,Q in enumerate(get_query_sets(QUERYSETFILE)):
        
        filename = "{}/{:0>3d}.txt".format(GOLDEN_CANDIDATE_NETWORKS_PATH.rstrip('/'),i+1) 
        
        if Q not in golden_standards:
                print("File {} not created because there is\n no golden standard set for the query\n {}".format(filename,Q))
                continue
        
        with open(filename,mode='w') as f:
            json_serializable = {'candidate_network':golden_standards[Q].to_json_serializable(),
                                 'query':Q,} 
            f.write(json.dumps(json_serializable,indent=4))


# +
#generate_golden_cn_files(golden_cns)
# -

def get_golden_query_matches(path=None):
    if path is None:
        path = GOLDEN_QUERY_MATCHES_PATH
    golden_cns = OrderedDict()
    for filename in sorted(glob.glob('{}/*.txt'.format(GOLDEN_QUERY_MATCHES_PATH.rstrip('/')))):
        with open(filename) as f:
            json_serializable = json.load(f)
            golden_cns[tuple(json_serializable['query'])] = \
                {KeywordMatch.from_json_serializable(js) for js in json_serializable['query_match']}
    return golden_cns


golden_qms=get_golden_query_matches()


# +
# golden_qms={ Q:cn.non_free_keyword_matches()
#             for Q,cn in golden_cns.items()}
# -

def generate_golden_qm_files(golden_qms,path=None):  
    if path is None:
        path = GOLDEN_QUERY_MATCHES_PATH
        
    for i,Q in enumerate(get_query_sets(QUERYSETFILE)):
        
        filename = "{}/{:0>3d}.txt".format(GOLDEN_QUERY_MATCHES_PATH.rstrip('/'),i+1) 
        
        if Q not in golden_qms:
                print("File {} not created because there is\n no golden standard set for the query\n {}".format(filename,Q))
                continue
        
        with open(filename,mode='w') as f:
            json_serializable = {'query_match':[keyword_match.to_json_serializable() 
                                                for keyword_match in golden_qms[Q]],
                                 'query':Q,} 
            f.write(json.dumps(json_serializable,indent=4))


# +
#generate_golden_qm_files(golden_qms)
# -

[(i,Q) for i,Q in enumerate(e) if e[Q] == -1]



for Q in e:
    if e[Q] == -1:
        print('\nQ:\n{}\nGS:\n{}\n'.format(Q,golden_cns[Q]))
        for i,x in enumerate(results[Q]['query_matches']):
            print('{}ª CN:\n{}'.format(i+1,x[0]))

result_query_matches = OrderedDict( (Q,[qm for qm,_,_,_ in results[Q]['query_matches']]) 
                        for Q in results )

result_candidate_networks = OrderedDict((Q,[cn for cn,_,_,_ in results[Q]['candidate_networks']]) for Q in results)

for x in result_candidate_networks:
    print('\n\n',x)
    for y in result_candidate_networks[x]:
        print(y)


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


positions_query_matches = get_relevant_positions(result_query_matches,golden_qms)

positions_candidate_networks = get_relevant_positions(result_candidate_networks,golden_cns)


# +
def index_step_non_empty_cn(candidate_network):
    return exec_sql(get_sql_from_cn(G,candidate_network,rowslimit=1), show_results=False,)

positions_non_empty_candidate_networks = get_relevant_positions(result_candidate_networks,
                                                      golden_cns,
                                                      index_step_func=index_step_non_empty_cn
                                                     )


# +
def mrr(position_list):
    return sum(1/p for p in position_list if p != -1)/len(position_list)

def precision_at(position_list,threshold = 3):
    return len([p for p in position_list if p != -1 and p<=threshold])/len(position_list)
           


# -

def metrics(position_list):
    result = OrderedDict()
    result['mrr']=mrr(position_list),
    result['p@1']=precision_at(position_list,threshold=1),
    result['p@2']=precision_at(position_list,threshold=2),
    result['p@3']=precision_at(position_list,threshold=3),
    result['p@4']=precision_at(position_list,threshold=4),
    result['max']=max(position_list),
    return result


metrics(positions_query_matches.values())

metrics(positions_candidate_networks.values())

metrics(positions_non_empty_candidate_networks.values())
