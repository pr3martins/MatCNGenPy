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
DBNAME = 'imdb_subset_pericles'
DBUSER = 'imdb'
DBPASS = 'imdb'
EMBEDDINGFILE = "word_embeddings/word2vec/GoogleNews-vectors-negative300.bin"
QUERYSETFILE ='querysets/queryset_imdb_martins_qualis.txt'
GONDELSTANDARDS ='golden_standards/imdb_coffman_revised'
GOLDENMAPPINGS ='golden_mappings/golden_mappings_imdb_martins.txt'

STEP_BY_STEP = False


# -

def validSchemaElement(text,embmodel=set()): 
    if 'id' in text or 'index' in text or 'code' in text:
        return False
    return True    


# +
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


# -

def tokenizeString(text):     
    return [word.strip(string.punctuation)
            for word in text.lower().split() 
            if word not in stopwords.words('english') or word == 'will']


def loadWordEmbeddingsModel(filename = EMBEDDINGFILE):
    model = KeyedVectors.load_word2vec_format(filename,
                                                       binary=True, limit=500000)
    return model


if STEP_BY_STEP:
    wordEmbeddingsModel=loadWordEmbeddingsModel(EMBEDDINGFILE)


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
    
    def printBabel(self):
        print(self.__babel)


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


class DatabaseIter:
    def __init__(self,embeddingModel,dbname=DBNAME,user=DBUSER,password=DBPASS):
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

                    if not validSchemaElement(table_name,embmodel=self.embeddingModel):
                        print('TABLE ',table_name, 'SKIPPED')
                        continue

                    print('\nINDEXING TABLE ',table_name)

                    #Get all tuples for this tablename
                    cur.execute(
                        sql.SQL("SELECT * FROM {} LIMIT 1;").format(sql.Identifier(table_name))
                        #NOTE: sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
                    )
                    
                    indexable_columns = []
                    
                    for column in range(1,len(cur.fetchone())):
                        column_name = cur.description[column][0]
                        if not validSchemaElement(column_name,embmodel=self.embeddingModel):
                            print('\tCOLUMN ',column_name,' SKIPPED')
                        else:
                            indexable_columns.append(sql.Identifier(column_name))
                            print('\tCOLUMN ',column_name,' NOT SKIPPED')
                    
                    if len(indexable_columns)==0:
                        continue
                                
                    #Get all tuples for this tablename
                    cur.execute(
                        sql.SQL("SELECT ctid, {} FROM {};").format(sql.SQL(', ').join(indexable_columns)
                                                                   ,sql.Identifier(table_name))
                        #NOTE: sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
                    )
                                       
                    for i,row in enumerate(cur.fetchall()): 
                        ctid = row[0]
                        for column in range(1,len(row)):
                            column_name = cur.description[column][0]
                            for word in tokenizeString( str(row[column]) ):
                                yield table_name,ctid,column_name, word
                        
                        if i%100000==1:
                            print('*',end='')


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

    print ('INVERTED INDEX CREATED')
    gc.collect()
    return wh,ah

if STEP_BY_STEP:
    (wordHash,attributeHash) = createInvertedIndex(wordEmbeddingsModel)


def processIAF(wordHash,attributeHash):
    
    total_attributes = sum([len(attribute) for attribute in attributeHash.values()])
    
    for (term, values) in wordHash.items():
        attributes_with_this_term = sum([len(attribute) for attribute in wordHash[term].values()])
        IAF = log1p(total_attributes/attributes_with_this_term)
        wordHash.setIAF(term,IAF)        
        
    print('IAF PROCESSED')


if STEP_BY_STEP:
    processIAF(wordHash,attributeHash)


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


if STEP_BY_STEP:
    processNormsOfAttributes(wordHash,attributeHash)


def preProcessing(emb_file=EMBEDDINGFILE):
    wordEmbeddingsModel=loadWordEmbeddingsModel(emb_file)
    (wordHash,attributeHash) = createInvertedIndex(wordEmbeddingsModel)
    processIAF(wordHash,attributeHash)
    processNormsOfAttributes(wordHash,attributeHash)
    
    print('PRE-PROCESSING STAGE FINISHED')
    return (wordHash,attributeHash,wordEmbeddingsModel)


# # Processing Stage

def getQuerySets(filename=QUERYSETFILE):
    QuerySet = []
    with open(filename,encoding='utf-8-sig') as f:
        for line in f.readlines():
            
            #The line bellow Remove words not in OLIVEIRA experiments
            #Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
            Q = tuple(tokenizeString(line))
            
            QuerySet.append(Q)
    return QuerySet


if STEP_BY_STEP:
    QuerySets = getQuerySets(QUERYSETFILE)
    Q = QuerySets[0]
    print(Q)


# ## class SchemaGraph

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
    
    def get_outgoing_neighbours(self,vertex): 
        return self.__graph_dict[vertex][0]
           
    def get_incoming_neighbours(self,vertex):
        return self.__graph_dict[vertex][1]
    
    def get_neighbours(self,vertex):
        return self.get_outgoing_neighbours(vertex) | self.get_incoming_neighbours(vertex)

    def add_edge(self, vertex1, vertex2,edge_info = None, edge_direction='>'):
        if edge_direction=='>':        
            self.get_outgoing_neighbours(vertex1).add(vertex2)
            self.get_incoming_neighbours(vertex2).add(vertex1)

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
            for neighbour in self.get_outgoing_neighbours(vertex):
                yield (vertex,)+ (neighbour,)
    
    def pp(self):
        pp(self.__graph_dict)
    
    def __repr__(self):
        return repr(self.__graph_dict)
    
    def __len__(self):
         return len(self.__graph_dict)
        
    def str_graph_dict(self):
        return str(self.__graph_dict)


def getSchemaGraph(dbname=DBNAME,user=DBUSER,password=DBPASS):
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


if STEP_BY_STEP:
    G = getSchemaGraph()  
    print(G)


# ## Class KeywordMatch

class KeywordMatch:
   
    def __init__(self, km_type,table, predicates = None, tuples = None):  
        self.__slots__ =['table','type','predicates','tuples']
        self.type = km_type
        self.table = table
        self.predicates= predicates if predicates is not None else {}
        self.tuples= tuples if tuples is not None else set()
        
    def addTuple(self, tuple_id):
        self.tuples.add(tuple_id)
        
    def addTuples(self, tuple_ids):
        self.tuples.update(tuple_ids)
        
    def addAttribute(self,attribute):
        self.attributes[attribute].setdefault( (set(),set()) )   
        
    def addMapping(self,keyword,attribute='*'):
        self.predicates.setdefault( attribute,set() )  
        self.predicates[attribute].add(keyword)
         
    def getMappings(self):
        return [(self.table,attribute,keywords) 
                for attribute, keywords in self.predicates.items()]
            
    def getAttributes(self):
        return [attr for attr in self.predicates.keys()]
                
    def getKeywords(self):
        keywords = set()
        for attribute in self.predicates.keys():
            keywords.update(self.predicates[attribute])
        return frozenset(keywords)
        
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
            keywords = self.predicates[attribute]
            
            if keywords == set():
                keywords = '{}'
                
            str_predicates.append (attribute + str(keywords))
            
        result += "(" + ','.join(str_predicates) + ")"
        return result        
    
    def __eq__(self, other):
        return isinstance(other, KeywordMatch) and self.table == other.table and self.predicates == other.predicates and self.tuples == other.tuples  
    
    def __hash__(self):
        return hash(frozenset(self.__repr__()))


# +
class ValueKeywordMatch(KeywordMatch):
    def __init__(self, table, predicates = None, tuples = None):
        self.__slots__ =['table','type','predicates','tuples']
        KeywordMatch.__init__(self, 1 ,table, predicates, tuples)
    
    def union(self, otherVKMatch, changeSources = False):
              
        if self.table != otherVKMatch.table:
            return None
        
        if self.table == None:
            return None
        
        if len(self.getKeywords() & otherVKMatch.getKeywords())>0:
            #tuple sets com palavras repetidas
            return None

        jointTuples = self.tuples & otherVKMatch.tuples
        
        jointPredicates = {}
        
        jointPredicates.update(copy.deepcopy(self.predicates))
        
        for attribute, keywords in otherVKMatch.predicates.items():  
            jointPredicates.setdefault(attribute, set() ).update(keywords)
            
        jointTupleset = ValueKeywordMatch(self.table, jointPredicates , jointTuples)
        
        if changeSources:
            self.tuples.difference_update(jointTuples)
            otherVKMatch.tuples.difference_update(jointTuples)
        
        return jointTupleset 

class SchemaKeywordMatch(KeywordMatch):
    def __init__(self, table, attribute, tuples = None):
        self.__slots__ =['table','type','predicates','tuples','__attribute']
        KeywordMatch.__init__(self, 2 ,table, predicates={attribute:set()}, tuples=tuples) 
        self.__attribute=attribute
    
    def addAttribute(self,attribute):
        pass 
        
    def addMapping(self,keyword):
        self.predicates[self.__attribute].add(keyword)
        
class FreeKeywordMatch(KeywordMatch):
    def __init__(self, table):
        self.__slots__ =['table','type','predicates','tuples']
        KeywordMatch.__init__(self, 3 ,table)
    def addAttribute(self,attribute):
        pass 
        
    def addMapping(self,keyword):
        pass


# +
def VKMGen(Q,wordHash):
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
                vkm = ValueKeywordMatch(table)
                vkm.addMapping(keyword,attribute)
                vkm.addTuples(ctids)                
                P.add(vkm)
    
    #Part 2: Find sets of tuples containing larger termsets
    TSInterMartins(P)
    
    
    #Part 3: Clean tuples
    for vkm in P:
        vkm.clearTuples()
    
    
    return P

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
# -

if STEP_BY_STEP:
    print('FINDING TUPLE-SETS')
    Rq = VKMGen(Q, wordHash)
    print(len(Rq),'TUPLE-SETS CREATED\n')
    pp(Rq)


# ## Class Similarities

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
        
        if table not in self.EmbA or column not in self.EmbA[table]:
            return False
        
        if Emb == 'A':
            sim_list = self.EmbA[table][column]
        
        elif Emb == 'B':
            sim_list = self.EmbA[table][column]   
            
            if column != '*':
                sim_list |= self.EmbB[table][column]
            else:                
                for neighbourTable in self.schemaGraph.get_neighbours(table):

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
    
        for table in self.attributeHash:

            if table not in self.model:
                continue

            self.EmbA[table]={}
            self.EmbB[table]= {}
            self.EmbC[table]= {}
            
            self.EmbA[table]['*'] = self.__getSimilarSet(table) 

            for column in self.attributeHash[table]:
                if column not in self.model or column=='id':
                    continue
                
                self.EmbA[table][column]=self.__getSimilarSet(column)
                
                self.EmbB[table][column]=self.__getSimilarSet( (table,column) )
                  
                avg_vec = (self.model[table]*weight + self.model[column]*(1-weight))                   
                self.EmbC[table][column] = self.__getSimilarSet(avg_vec, inputType = 'vector')
                
        for tableA in self.schemaGraph.vertices():

            if tableA not in self.attributeHash or tableA not in self.model:
                continue

            for tableB in self.schemaGraph.get_neighbours(tableA):

                if tableB not in self.attributeHash or tableB not in self.model:
                    continue
                
                self.EmbB[tableA][tableB] = self.__getSimilarSet( (tableA,tableB) )
        

if STEP_BY_STEP:
    similarities=Similarities(wordEmbeddingsModel,attributeHash,getSchemaGraph())


def SKMGen(Q,attributeHash,similarities,threshold=0.8):    
    S = set()
    
    for keyword in Q:
        for table in attributeHash:            
            for attribute in ['*']+list(attributeHash[table].keys()):
                
                if(attribute=='id'):
                    continue
                
                sim = similarities.word_similarity(keyword,table,attribute)
                
                if sim >= threshold:
                    skm = SchemaKeywordMatch(table,attribute)
                    skm.addMapping(keyword)
                    S.add(skm)
                    
    return S


if STEP_BY_STEP:    
    print('FINDING SCHEMA-SETS')        
    Sq = SKMGen(Q,attributeHash,similarities)
    print(len(Sq),' SCHEMA-SETS CREATED\n')
    pp(Sq)


# ## Query Matching

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
                Mq.append(M)
                   
    return Mq 


if STEP_BY_STEP:
    print('GENERATING QUERY MATCHES')
    TMaxQM = 3
    Mq = QMGen(Q,Rq|Sq,TMaxQM=TMaxQM)
    print (len(Mq),'QUERY MATCHES CREATED\n')  


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
            
            if isinstance(keyword_match,ValueKeywordMatch):
                if showLog:
                    print(keyword_match)
                
                for table, attribute, valueWords in keyword_match.getMappings():

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
        
        
        for keyword_match in M:
            
            if isinstance(keyword_match,SchemaKeywordMatch):
                if showLog:
                    print(keyword_match)
                
                for table, attribute, schemaWords in keyword_match.getMappings():
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
                            
    return sorted(Ranking,key=lambda x: x[1],reverse=True)
                

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
            '\nValue Score: ',"%.8f" % valuescore,
            '\n|M|: ',"%02d (Não considerado para calcular o total score)" % len(M),
            '\nTotal Score: ',"%.8f" % score)
        pp(M)
        #print('\n----Details----\n')
        #QMRank(Q, [M],wordHash,attributeHash,similarities,showLog=True)

        print('----------------------------------------------------------------------\n')


class CandidateNetwork(Graph):
    def add_vertex(self, vertex):
        return super().add_vertex((vertex, 't{}'.format(self.__len__()+1)))
        
    def keyword_matches(self):
        return {keyword_match for keyword_match,alias in self.vertices()}
            
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
                
    def leveled_dfs_iter(self,start_vertex=None,visited = None, level=0):
        if len(self)>0:
            if start_vertex is None:
                start_vertex = self.get_starting_vertex()             
            if visited is None:
                visited = set()
            visited.add(start_vertex)

            yield( (level,start_vertex) )

            for neighbour in self.get_neighbours(start_vertex):
                if neighbour not in visited:
                    yield from self.leveled_dfs_iter(neighbour,visited,level=level+1)  
    
    def dfs_iter(self,start_vertex=None,visited = None, level=0):
        for level,vertex in self.leveled_dfs_iter():
            yield vertex    
    
    def get_starting_vertex(self):
        vertex = None
        for vertex in self.vertices():
            keyword_match,alias = vertex
            if not isinstance(keyword_match,FreeKeywordMatch):
                break
        return vertex
    
    def __repr__(self):
        if len(self)==0:
            return 'EmptyCN'            
        print_string = ['\t'*level+str(vertex[0])  for level,vertex in self.leveled_dfs_iter()]            
        return '\n'.join(print_string)
    
    def remove_vertex(self,vertex):
        print('vertex:\n{}\n_Graph__graph_dict\n{}'.format(vertex,self._Graph__graph_dict))
        outgoing_neighbours,incoming_neighbours = self._Graph__graph_dict[vertex]
        for neighbour in incoming_neighbours:
            self._Graph__graph_dict[neighbour][0].remove(vertex)
        self._Graph__graph_dict.pop(vertex)
        
    def minimal_cover(self,QM):
        for vertex in self.vertices():
            keyword_match,alias = vertex
            if isinstance(keyword_match,FreeKeywordMatch):
                visited = {vertex}
                start_node = next(iter( self.vertices() - visited ))
                
                for vertex in self.leveled_dfs_iter(start_node,visited=visited):
                    #making sure that the dfs algorithm runs until the end of iteration
                    continue
                
                if visited == self.vertices():
                    return False
        return True
                


def CNGraphGen(QM,G,TMax=10,showLog=False,tuplesetSortingOrder = None,topKCNs=5):  
    if showLog:
        print('================================================================================\nSINGLE CN')
        print('Tmax ',TMax)
        print('FM')
        pp(QM)
        #print('\n\n')
        #print('\n\nGts')
        #pp(Gts)
        #print('\n\n')
    
    table_hash = {}
    
    returnedCNs = []
    
    for keyword_match in QM:
        table_hash.setdefault(keyword_match.table,([],[]))
        if isinstance(keyword_match,ValueKeywordMatch):
            table_hash[keyword_match.table][0].append(keyword_match)
        else:
            table_hash[keyword_match.table][1].append(keyword_match)
       
    new_QM = set()
    for table,(vk_matches,sk_matches) in table_hash.items():       
        if len(vk_matches)>0:
            table_hash[table]=vk_matches
        else:
            table_hash[table]=sk_matches[0:1]
        
        new_QM.update(set(table_hash[table]))
    
    F = deque()
    
    first_element = next(iter(new_QM))
        
    CN = CandidateNetwork()
    CN.add_vertex(first_element)
    
    if len(new_QM)==1:
        returnedCNs.append(CN)
    else:    
        F.append(CN)
        
    while F:        
        CN = F.popleft()
#         print('///////////////')
#         pp([x for x in CN.vertices()])
        
        
        for vertex_u in CN.vertices():
            keyword_match,alias = vertex_u
            
            for adj_table in G.get_neighbours(keyword_match.table):
#                 print('CHECKING TABLE ',adj_table)
                
                if adj_table in G.get_outgoing_neighbours(keyword_match.table):
                    direction = '>'
                else:
                    direction = '<'
                
#                 print('NON-FREE KEYWORD MATCHES')
                if adj_table in table_hash:                    
                    for adj_keyword_match in table_hash[adj_table]:
                        if adj_keyword_match not in CN.keyword_matches():
                            new_CN = copy.deepcopy(CN)
                            vertex_v = new_CN.add_vertex(adj_keyword_match)
                            new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction)         

                            if new_CN not in F and len(new_CN)<=TMax and new_CN.isSound():
#                                 print('Adding ',adj_keyword_match,' to current CN')
                                if new_CN.minimal_cover(new_QM):
#                                     print('GENERATED THE FIRST ONE')
                                    if len(returnedCNs)<topKCNs:
                                        returnedCNs.append(new_CN)
                                    else:
                                        return returnedCNs
                                else:
                                    F.append(new_CN)
                                    
                
                
                new_CN = copy.deepcopy(CN)
#                 print('FREE KEYWORD MATCHES')
                adj_keyword_match = FreeKeywordMatch(adj_table)
                vertex_v = new_CN.add_vertex(adj_keyword_match)
                new_CN.add_edge(vertex_u,vertex_v,edge_direction=direction)
                if new_CN not in F and len(new_CN)<=TMax and new_CN.isSound():
#                     print('Adding ',adj_keyword_match,' to current CN')
                    F.append(new_CN)
#                 else:
#                     print('Did not add ',adj_keyword_match,' to current CN ({},{},{})'.format(new_CN not in F,len(new_CN)<=TMax,new_CN.isSound()))
                        
    return returnedCNs

if STEP_BY_STEP:
    TMax=5
    topK = 20    
    tuplesetSortingOrder = {table : 1/sum([Norm for (Norm,numDistinctWords,numWords,maxFrequency) in attributeHash[table].values()]) for table in attributeHash}
    
    (QM,score,valuescore,schemascore) = RankedMq[0]
    print('GENERATING CNs FOR QM:',QM)
    
    Cns = CNGraphGen(QM,G,TMax=TMax,tuplesetSortingOrder=tuplesetSortingOrder)
    
    for j, Cn in enumerate(Cns):
        print(j+1,'ª CN',
              '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
              '\nTotal Score: ',"%.8f" % (score/len(Cn)))
        pp(Cn)


def MatchCN(attributeHash,G,RankedMq,TMax=10,maxNumCns=20,tuplesetSortingOrder=None):    
    UnrankedCns = []    
    generated_cns=set()
    
    for  (QM,score,valuescore,schemascore) in RankedMq:
        Cns = CNGraphGen(QM,G,TMax=TMax,tuplesetSortingOrder=tuplesetSortingOrder)
        if len(UnrankedCns)>maxNumCns:
            break
    
        for Cn in Cns:
            if(Cn not in generated_cns):          
                generated_cns.add(Cn)

                #Dividindo score pelo tamanho da cn (SEGUNDA PARTE DO RANKING)                
                CnScore = score/len(Cn)

                UnrankedCns.append( (Cn,CnScore,valuescore,schemascore) )
    
    #Ordena CNs pelo CnScore
    RankedCns=sorted(UnrankedCns,key=lambda x: x[1],reverse=True)
    
    return RankedCns


# # NOTE: SABER SE UMA CN É UMA COBERTURA MÍNIMA NÃO É MAIS TRIVIAL

if STEP_BY_STEP:   
    print('GENERATING CANDIDATE NETWORKS')  
    RankedCns = MatchCN(attributeHash,G,RankedMq,TMax=TMax,maxNumCns=topK,tuplesetSortingOrder=tuplesetSortingOrder)
    print (len(RankedCns),'CANDIDATE NETWORKS CREATED AND RANKED\n')
    
    for (j, (Cn,score,valuescore,schemascore) ) in enumerate(RankedCns):
        print(j+1,'ª CN')
        print('Schema Score:',"%.8f" % schemascore,
              '\nValue Score: ',"%.8f" % valuescore,
              '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
              '\nTotal Score: ',"%.8f" % score)
        pp(Cn)


# ## getSQLfromCN

def getSQLfromCN(G,Cn,showEvaluationFields=False,rowslimit=1000):
    
    hashtables = {} # used for disambiguation

    selected_attributes = []
    filter_conditions = []
    disambiguation_conditions = []
    selected_tables = []

    tables__search_id = []
    relationships__search_id = []

    prev_vertex = None

    for vertex in Cn.dfs_iter():
        keyword_match, alias = vertex
        for _ ,attr,keywords in keyword_match.getMappings():
            selected_attributes.append('{}.{}'.format(alias,attr))
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


            if keyword_match.table in G.get_outgoing_neighbours(prev_keyword_match.table):
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

        prev_vertex = vertex


    for table,aliases in hashtables.items():        
        for i in range(len(aliases)):
            for j in range(i+1,len(aliases)):
                disambiguation_conditions.append('{}.ctid <> {}.ctid'.format(aliases[i],aliases[j]))
        
    if len(tables__search_id)>0:
        tables__search_id = ['({}) AS Tuples'.format(', '.join(tables__search_id))]
    if len(relationships__search_id)>0:
        relationships__search_id = ['({}) AS Relationships'.format(', '.join(relationships__search_id))]

    sqlText = '\nSELECT\n\t{}\nFROM\n\t{}\nWHERE\n\t{}\nLIMIT {}'.format(
        ',\n\t'.join( tables__search_id+relationships__search_id+selected_attributes ),
        ',\n\t'.join(selected_tables),
        '\n\tAND '.join( disambiguation_conditions+filter_conditions),
        rowslimit)
    print(sqlText)

if STEP_BY_STEP:
    (Cn,score,valuescore,schemascore)= RankedCns[15]
    print(Cn)
    getSQLfromCN(G,Cn,showEvaluationFields=True)


def execSQL (SQL,dbname=DBNAME,user=DBUSER,password=DBPASS,showResults=True):
    #print('RELAVANCE OF SQL:\n')
    #print(SQL)
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

import pandas as pd
URI = 'postgres://imdb:imdb@localhost/'+DBNAME
def  execSQLPandas(SQL,dbname=DBNAME,user=DBUSER,password=DBPASS):
    df=pd.read_sql(SQL, URI)
    if len(df)>0:
        display(df)
        return True
    return False


def keywordSearch(wordHash,attributeHash,wordEmbeddingsModel,
         showLog=False,
         SimilarityThreshold=0.9,
         querySetFileName=QUERYSETFILE,
         goldenStandardsFileName=GONDELSTANDARDS, numQueries=11,
         goldenMappingsFileName=GOLDENMAPPINGS,
         evaluation = True,
         topK=15,
         TMax=10,
         TMaxQM=3,
         tuplesetSortingOrder = None,
         similarities = None
         ):
    QuerySets = getQuerySets(querySetFileName)
    G = getSchemaGraph()    
    
    
    if tuplesetSortingOrder is None:
        tuplesetSortingOrder = {table : 1/sum([Norm for (Norm,numDistinctWords,numWords,maxFrequency) in attributeHash[table].values()]) for table in attributeHash}
    if similarities is None:
        similarities=Similarities(wordEmbeddingsModel,attributeHash,G)
    
    
    for (i,Q) in enumerate(QuerySets):
       
        print('QUERY-SET ',Q,'\n')
        
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
                print(j+1,'ª QM')           
                
                print('Schema Score:',"%.8f" % schemascore,
                      '\nValue Score: ',"%.8f" % valuescore,
                      '\n|QM|: ',"%02d (Não considerado para calcular o total score)" % len(QM),
                      '\nTotal Score: ',"%.8f" % score)
                pp(QM)
                
                print('----------------------------------------------------------------------\n')
        
        
        print('GENERATING CANDIDATE NETWORKS')     
        RankedCns = MatchCN(attributeHash,G,RankedMq,TMax=TMax,maxNumCns=topK,tuplesetSortingOrder=tuplesetSortingOrder)
        
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

(wordHash,attributeHash,wordEmbeddingsModel) = preProcessing()

Result = keywordSearch(wordHash,attributeHash,wordEmbeddingsModel,
                       evaluation=False,showLog=True,
                       querySetFileName=QUERYSETFILE,
                       goldenStandardsFileName=GONDELSTANDARDS,
                       numQueries=50,
                       topK=10,
                       TMaxQM=3,
                       #tuplesetSortingOrder = {'movie_info':6,'char_name':3,'role_type':4,'cast_info':5,'title':1,'name':2},
                       #tuplesetSortingOrder = {'movie_info':6,'character':3,'role':4,'casting':5,'movie':1,'person':2},
                      )
