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
            if word not in stopwords.words('english') + ['will']]


def loadWordEmbeddingsModel(filename = "word_embeddings/word2vec/GoogleNews-vectors-negative300.bin"):
    model = KeyedVectors.load_word2vec_format(filename,
                                                       binary=True, limit=500000)
    return model


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
    def __init__(self,embeddingModel,dbname='imdb',user='imdb',password='imdb'):
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

                            for word in tokenizeString( str(row[column]) ):
                                yield table_name,ctid,column_name, word

                        printSkippedColumns=False


def createInvertedIndex(embeddingModel,dbname='imdb',user='imdb',password='imdb',showLog=True):
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


def processIAF(wordHash,attributeHash):
    
    total_attributes = sum([len(attribute) for attribute in attributeHash.values()])
    
    for (term, values) in wordHash.items():
        attributes_with_this_term = sum([len(attribute) for attribute in wordHash[term].values()])
        IAF = log1p(total_attributes/attributes_with_this_term)
        wordHash.setIAF(term,IAF)        
        
    print('IAF PROCESSED')


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


# ## Class Tupleset

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
            if self.isValueFreeTupleset()==False and otherTupleset.isValueFreeTupleset() == False:
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
        return hash(frozenset(self.__repr__()))


# +
def TSFind(Q,wordHash):
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


# -

def getQuerySets(filename='querysets/queryset_imdb_martins_new.txt'):
    QuerySet = []
    with open(filename,encoding='utf-8-sig') as f:
        for line in f.readlines():
            
            #The line bellow Remove words not in OLIVEIRA experiments
            #Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
            Q = tuple(tokenizeString(line))
            
            QuerySet.append(Q)
    return QuerySet


# ## class SchemaGraph

class SchemaGraph:
    
    def __init__(self):
        self.__graph = {}
    
    def addRelationship(self,tableA,columnA,tableB, columnB, direction = 1):        
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
        return [ts.table for ts in self.__graph.keys()]
    
    def tuplesets(self):
        return self.__graph.keys()
        
    def getAdjacentTables(self, table, sort = False):
        return [ts.table for ts in self.getByTableName(table).keys()]

    def getAdjacentTuplesets(self, table, sort = False):
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
            
            if tsA.table == tsC.table:
                            
                for edge_info in self.__graph[tsA][tsB]:
                    (columnA,columnB,direction) = edge_info
                    
                    if direction == -1:
                        return False
        return True    
    
        
    def __repr__(self):
        return pprint.pformat(self.__graph)
    
    def __str__(self):
        return repr(self.__graph)


def getSchemaGraph(dbname='imdb',user='imdb',password='imdb'):
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


# ## Class Similarities

class Similarities:
    
    def __init__(self, model, attributeHash,schemaGraph):

        self.model = model
        self.attributeHash = attributeHash
        self.schemaGraph = schemaGraph
        self.porter = PorterStemmer()
        
        
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
        
        if Emb == 'A':
            sim_list = self.EmbA[table][column]            
        
        elif Emb == 'B':
            
            sim_list = self.EmbA[table][column]   
            
            if column != '*':
                sim_list |= self.EmbB[table][column]
                
            for neighbourTable in self.schemaGraph.getAdjacentTables(table):
                
                if neighbourTable not in self.attributeHash or neighbourTable not in self.model:
                    continue
                
                sim_list |= self.EmbB[table][neighbourTable]   
        
        elif Emb == 'C':
            
            sim_list = self.EmbA[table][column]   
            
            if column != '*':
                sim_list |= self.EmbB[table][column]
        
        else:
            sim_list=[]
        
        
        #print('sim({},{}.{}) = {}'.format(word,table,column,sim_list))        
        return self.porter.stem(word) in sim_list
                
    
    
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
                
        
        
        return sim    
    
    def __getSimilarSet(self,word, inputType = 'word'):
        if inputType == 'vector':
            sim_list = self.model.similar_by_vector(word)
        else:
            sim_list = self.model.most_similar(word) 
        
        return  {self.porter.stem(word.lower()) for word,sim in sim_list}
    
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
                
        for tableA in self.schemaGraph.tables():

            if tableA not in self.attributeHash or tableA not in self.model:
                continue

            for tableB in self.schemaGraph.getAdjacentTables(tableA):

                if tableB not in self.attributeHash or tableB not in self.model:
                    continue
                
                
                self.EmbB[tableB][tableA] = self.EmbB[tableA][tableB] = self.__getSimilarSet( (tableA,tableB) )
        

def SchSFind(Q,attributeHash,similarities,threshold=0.8, 
             sim_args={}):    
    S = set()
    
    for keyword in Q:
        for table in attributeHash:            
            for attribute in ['*']+list(attributeHash[table].keys()):
                
                if(attribute=='id'):
                    continue
                
                sim = similarities.word_similarity(keyword,table,attribute,**sim_args)
                
                if sim >= threshold:
                    ts = Tupleset(table)
                    ts.addSchemaMapping(keyword,attribute)
                    S.add(ts)
                    
    return S


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


# +
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
#                 print('----------------------------------------------\nM')
#                 pp(set(subset))
#                 print('\n')
                
                M = MInter(set(subset))
#                 print('subset')
#                 pp(M)
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


# -

def QMRank(Mq, wordHash,attributeHash,similarities):
    Ranking = []  

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
                    sim = similarities.word_similarity(term,table,attribute)
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
                

def SingleCN(FM,Gts,TMax=10,showLog=False):  
  
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
        
        sortedAdjacents = Gts.getAdjacentTuplesets(tsu.table,sort = True)
        
        if showLog:
            print('--------------------------------------------\nParctial CN')
            print('J ',J,'\n')

            print('\nAdjacents:')
            pp(Gts.getAdjacentTuplesets(tsu.table))
            
            #print('F:')
            #pp(F)
        
        for tsv in sortedAdjacents:
            
            if showLog:
                print('Checking adj:')
                pp(tsv)
                print()

            if (tsv.isFreeTupleset()) or (tsv not in J):
                
                Ji = J + [tsv]
                
                if (Ji not in F) and (len(Ji)<=TMax) and (Gts.isJNTSound(Ji)):
                    
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


def MatchCN(G,RankedMq,TMax=10):    
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


# ## getSQLfromCN

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
    sqlText +=' ('+', '.join(tables_id)+') AS Tuples,\n '
    if len(relationships)>0:
        sqlText +='('+', '.join(relationshipsText)+') AS Relationships,\n '
        
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


def getGoldenStandards(goldenStandardsFileName='golden_standards/imdb_martins',numQueries=11):
    goldenStandards = {}
    for i in range(1,numQueries+1):
        filename = goldenStandardsFileName+'/'+str(i).zfill(3) +'.txt'
        with open(filename) as f:

            listOfTuples = []
            Q = ()
            for j, line in enumerate(f.readlines()):
                
                splitedLine = line.split('#')
                
                line_without_comment=splitedLine[0]
                
                if len(splitedLine)>1:
                    comment_of_line=splitedLine[1]
                
                    if(j==2):
                        query = comment_of_line
                        Q = tuple(tokenizeString(query))
                    
                if line_without_comment:                    
                    
                    relevantResult = eval(line_without_comment)
                    listOfTuples.append( relevantResult )
            
            goldenStandards[Q]=listOfTuples
            
    return goldenStandards



def getGoldenMappings(goldenMappingsFileName='golden_mappings/golden_mappings_imdb_martins.txt'):
    
    goldenMappings = []
    with open(goldenMappingsFileName) as f:
        for j, line in enumerate(f.readlines()):

            splitedLine = line.split('#')

            line_without_comment=splitedLine[0]

            if len(splitedLine)>1:
                comment_of_line=splitedLine[1]

            if line_without_comment:                    
                tupleset = eval(line_without_comment)
                goldenMappings.append(tupleset)

    return goldenMappings


# +
def evaluateCN(CnResult,goldenStandard):
    '''
    print('Verificar se são iguais:\n')
    print('Result: \n',CnResult)
    print('Golden Result: \n',goldenStandard)
    '''
    
    tuplesOfCNResult =  set(CnResult[0])
    
    tuplesOfStandard =  set(goldenStandard[0])
        
    #Check if the CN result have all tuples in golden standard
    if tuplesOfCNResult.issuperset(tuplesOfStandard) == False:
        return False
    
    
    relationshipsOfCNResult = CnResult[1]
    
    relationshipsOfStandard = goldenStandard[1]
    
    if len(relationshipsOfCNResult)!=len(relationshipsOfStandard):
        #print('TAM OF JOIN PATHS DIFFERENT')
        
        #print('relationshipsOfCNResult')
        #pp(relationshipsOfCNResult)
        
        #print('\relationshipsOfStandard')
        #pp(relationshipsOfStandard)
        
        return False
    
    for goldenRelationship in relationshipsOfStandard:
        
        (A,B) = goldenRelationship
        
        if (A,B) not in relationshipsOfCNResult and (B,A) not in relationshipsOfCNResult:
            return False
        
    return True


def evaluanteResult(Result,Query,goldenStandards):
    
    goldenStandard = goldenStandards[tuple(Query)]
    
    #print('RESULT')
    #pp(Result)
    
    #print('STANDARD')
    #pp(goldenStandard)
    
    for goldenRow in goldenStandard:

        found = False

        for row in Result:
            if evaluateCN(row,goldenRow):
                found = True

        if not found:
            return False
        
    return True
            

def normalizeResult(ResultFromDatabase,Description):
    normalizedResult = []
    
    if Description[1].name=='relationships':
        hasRelationships = True
    else:
        hasRelationships = False
    
    for row in ResultFromDatabase:       
        if type(row[0]) == int:
            tuples = [row[0]]
        else:
            tuples = eval(str(row[0]))
        
        if hasRelationships:
            relationships = eval(row[1])
            #print('RELATIONSHIPS')
            #pp(relationships)
            if type(relationships[0]) != int:
                relationships = [eval(element) for element in relationships]
            else:
                relationships = [relationships]
        else:
            relationships=[]
        
        normalizedResult.append( (tuples,relationships) )
    return normalizedResult


# -

def getRelevantPosition(RankedCns,Q,goldenStandards):
    
    position=0
    nonEmptyPosition=0
    
    print(Q,'\n')
    
    for (Cn,Gts,score,schemascore,valuescore) in RankedCns:
        
        print('*',end='')
        
        #print('CN:\n')
        #pp(Cn)
        SQL1 = getSQLfromCN(Gts,Cn,contract=True)
        SQL2 = getSQLfromCN(Gts,Cn,contract=False)
        #print('\nSQL1\n')
        #print(SQL1)
        #print('\nSQL2\n')
        #print(SQL2)
        
        def getRelevanceFromSQL(SQL,dbname='imdb',user='imdb',password='imdb'):
            #print('RELAVANCE OF SQL:\n')
            #print(SQL)
            with psycopg2.connect(dbname=dbname,user=user,password=password) as conn:
                with conn.cursor() as cur:
                
                    cur.execute(SQL)

                    Results = cur.fetchall()
                    Description = cur.description

                    isEmpty = (len(Results)==0)

                    NResults = normalizeResult(Results, Description)

                    Relevance = evaluanteResult(NResults,Q,goldenStandards)

                    return (Relevance, isEmpty)
        
        (Relevance, isEmpty)=getRelevanceFromSQL(SQL1)
        if Relevance==False:
            (Relevance, isEmpty)=getRelevanceFromSQL(SQL2)
    
        position+=1
        if not isEmpty:
            nonEmptyPosition+=1
        
        if Relevance:
            print()
            return (position,nonEmptyPosition)
    print()
    return (-1,-1)


# +
def preProcessing(emb_file="word_embeddings/word2vec/GoogleNews-vectors-negative300.bin"):
    
    wordEmbeddingsModel=loadWordEmbeddingsModel(emb_file)
    (wordHash,attributeHash) = createInvertedIndex(wordEmbeddingsModel)
    processIAF(wordHash,attributeHash)
    processNormsOfAttributes(wordHash,attributeHash)
    
    print('PRE-PROCESSING STAGE FINISHED')
    return (wordHash,attributeHash,wordEmbeddingsModel)


# -

wordHash,attributeHash,wordEmbeddingsModel=preProcessing()


def keywordSearch (wordHash,attributeHash,wordEmbeddingsModel,sim_args={},
         showLog=False,
         SimilarityThreshold=0.9,
         querySetFileName='querysets/queryset_imdb_martins_new.txt',
         goldenStandardsFileName='golden_standards/imdb_martins', numQueries=11,
         goldenMappingsFileName='golden_mappings/golden_mappings_imdb_martins.txt',
         evaluation = True,
         topK=10,
         TMax=10):
    
    QuerySets = getQuerySets(querySetFileName)
    goldenStandards = getGoldenStandards(goldenStandardsFileName=goldenStandardsFileName,numQueries=numQueries)
    goldenMappings = getGoldenMappings()
    
    TP=[]
    FP=[]
    FN=[]
    
    listSkippedCN=[]
    
    relevantPositions = []
    nonEmptyRelevantPositions = []
    
    G = getSchemaGraph()  
    similarities = Similarities(wordEmbeddingsModel,attributeHash,G)
    
    for (i,Q) in enumerate(QuerySets):
       
        print('QUERY-SET ',Q,'\n')
        
        print('FINDING TUPLE-SETS')
        Rq = TSFind(Q, wordHash)
        print(len(Rq),'TUPLE-SETS CREATED\n')
        
        print('FINDING SCHEMA-SETS')        
        Sq = SchSFind(Q,attributeHash,similarities)

        print(len(Sq),' SCHEMA-SETS CREATED\n')

        for schema_mapping in Sq:
            if schema_mapping in goldenMappings:
                TP.append(schema_mapping)
                goldenMappings.remove(schema_mapping)
            else:
                FP.append(schema_mapping)       
        
        print('GENERATING QUERY MATCHES')
        Mq = QMGen(Q,Rq|Sq)
        print (len(Mq),'QUERY MATCHES CREATED\n')
        
        print('RANKING QUERY MATCHES')
        RankedMq = QMRank(Mq,wordHash,attributeHash,similarities)   
        
        if showLog:
            for (j, (M,score,schemascore,valuescore) ) in enumerate(RankedMq[:topK]):
                print(j+1,'ª QM')
                print('Schema Score:',"%.8f" % schemascore,
                      '\nValue Score: ',"%.8f" % valuescore,
                      '\n|M|: ',"%02d (Não considerado para calcular o total score)" % len(M),
                      '\nTotal Score: ',"%.8f" % score)
                pp(M)
                print('----------------------------------------------------------------------\n')
        
        
        if topK<=0:
            topKMq=RankedMq
        else:
            topKMq=RankedMq[:topK]
        
        numSkippedCNs = len(RankedMq)-topK
        
        
        RankedMq=[]
        gc.collect()
        
        if numSkippedCNs>0:
            print(numSkippedCNs,' QUERY MATCHES SKIPPED (due to low score)')
        else:
            numSkippedCNs=0
            
        
        print('GENERATING CANDIDATE NETWORKS')     
        
        RankedCns = MatchCN(G,topKMq,TMax=TMax)
        
        listSkippedCN.append(numSkippedCNs)
        
        print (len(RankedCns),'CANDIDATE NETWORKS CREATED AND RANKED\n')
        
        if showLog:
            for (j, (Cn,Gts,score,schemascore,valuescore) ) in enumerate(RankedCns):
                print(j+1,'ª CN')
                print('Schema Score:',"%.8f" % schemascore,
                      '\nValue Score: ',"%.8f" % valuescore,
                      '\n|Cn|: ',"%02d (Considerado para o Total Score)" % len(Cn),
                      '\nTotal Score: ',"%.8f" % score)
                pp(Cn)
                print()
                print(getSQLfromCN(Gts,Cn))
                print('\nsem arvore')
                print(getSQLfromCN(Gts,Cn,contract=False))
                print('----------------------------------------------------------------------\n')
        
        print('CHECKING RELEVANCE')
        
        if(evaluation):
            (pos,nonEmptyPos)=getRelevantPosition(RankedCns,Q,goldenStandards)
        else:
            (pos,nonEmptyPos) = -1,-1
        
        if pos<0:
            print('NO RELEVANT CN FOUND')
        else:
            (Cn,_,_,_,_) = RankedCns[pos-1]
            print('RELEVANT CN IN %d POSITION'%(pos))
            pp(Cn)
                        
        relevantPositions.append(pos)
        nonEmptyRelevantPositions.append(nonEmptyPos)
        
        print('==========================================================================\
==========================================================================\
==========================================================================\
==========================================================================\
==========================================================================\
==========================================================================')
    FN=goldenMappings
    return (relevantPositions,nonEmptyRelevantPositions,listSkippedCN,TP,FP,FN)

Result = keywordSearch(wordHash,attributeHash,wordEmbeddingsModel,
                       evaluation=False,showLog=True)

(relevantPositions,nonEmptyRelevantPositions,listSkippedCN,TP,FP,FN) = Result

x = getQuerySets()
for i,Q in enumerate(x):
    print('{} Query:{}\nrelevant:{}\nnon-empty relevant:{}\nskippedCN:{}\nTP:{}\nFP:{}\nFN:{}\n\n'.format(i,Q,relevantPositions,nonEmptyRelevantPositions,listSkippedCN,TP,FP,FN))

similarities = Similarities(wordEmbeddingsModel,attributeHash,getSchemaGraph())

similarities.EmbB
