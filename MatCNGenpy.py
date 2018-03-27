
# coding: utf-8

# # MatCNGenPy
# 
# This is a python implementation of the algorithms described in 
# **Efficient Match-Based Candidate Network Generation for Keyword 
# Queries over Relational Databases** paper.
# 
# 
# ## Installation
# - Install virtalenv
# - Run ```source bin/activate``` to enter in the virtual enviroment
# - Run ```pip install -r requirements.txt```
# - Run ```python ModCNGen.py

# In[1]:


import psycopg2
from psycopg2 import sql
from pprint import pprint as pp
from collections import defaultdict
import string
import itertools
import copy
from math import log1p
from queue import deque
import ast
import gc
from queue import deque

import nltk 
#nltk.download('wordnet')
#nltk.download('omw')
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stw_set = set(stopwords.words('english')) - {'will'}

# Connect to an existing database
conn = psycopg2.connect("dbname=imdb user=postgres")

# Open a cursor to perform database operations
cur = conn.cursor()


# In[2]:


def createInvertedIndex():
    #Output: wordHash (Term Index) with this structure below
    #map['word'] = [ 'table': ( {column} , ['ctid'] ) ]

    '''
    The Term Index is built in a preprocessing step that scans only
    once all the relations over which the queries will be issued.
    '''
    
    wordHash = {}
    attributeHash = {}
    
    # Get list of tablenames
    cur.execute("SELECT DISTINCT tablename FROM pg_tables WHERE schemaname!='pg_catalog' AND schemaname !='information_schema';")
    for table in cur.fetchall():
        table_name = table[0]
        print('INDEXING TABLE ',table_name)
        
        attributeHash[table_name] = {}
        
        #Get all tuples for this tablename
        cur.execute(
            sql.SQL("SELECT ctid, * FROM {};").format(sql.Identifier(table_name))
            #NOTE: sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
        )

        for row in cur.fetchall():
            for column in range(1,len(row)):
                column_name = cur.description[column][0]   
                ctid = row[0]

                for word in [word.strip(string.punctuation) for word in str(row[column]).lower().split()]:
                    
                    #Ignoring STOPWORDS
                    if word in stw_set:
                        continue

                    #If word entry doesn't exists, it will be inicialized (setdefault method),
                    #Append the location for this word
                    wordHash.setdefault(word, {})                    
                    wordHash[word].setdefault( table_name , {} )
                    wordHash[word][table_name].setdefault( column_name , [] ).append(ctid)
                    
                    attributeHash[table_name].setdefault(column_name,(0,set()))
                    attributeHash[table_name][column_name][1].add(word)
        
        #Count words
        
        for (column_name,(norm,wordSet)) in attributeHash[table_name].items():
            num_distinct_words = len(wordSet)
            wordSet.clear()
            attributeHash[table_name][column_name] = (norm,num_distinct_words)
        

    print ('INVERTED INDEX CREATED')
    return (wordHash,attributeHash)

#(wordHash,attributeHash) = createInvertedIndex()


# In[3]:


def processIAF(wordHash,attributeHash):
    
    total_attributes = sum([len(attribute) for attribute in attributeHash.values()])
    
    for (term, values) in wordHash.items():
        
        attributes_with_this_term = sum([len(attribute) for attribute in wordHash[term].values()])
        
        IAF = log1p(total_attributes/attributes_with_this_term)
                
        wordHash[term] = (IAF,values)
    print('IAF PROCESSED')
#processIAF(wordHash,attributeHash)


# In[4]:


def processNormsOfAttributes(wordHash,attributeHash):
  
    # Get list of tablenames
    cur.execute("SELECT DISTINCT tablename FROM pg_tables WHERE schemaname!='pg_catalog' AND schemaname !='information_schema';")
    for table in cur.fetchall():
        table_name = table[0]
        print('PROCESSING TABLE ',table_name)
        
        #Get all tuples for this tablename
        cur.execute(
            sql.SQL("SELECT ctid, * FROM {};").format(sql.Identifier(table_name))
            #NOTE: sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
        )

        for row in cur.fetchall():
            for column in range(1,len(row)):
                column_name = cur.description[column][0]   
                ctid = row[0]

                for word in [word.strip(string.punctuation) for word in str(row[column]).lower().split()]:
                    
                    #Ignoring STOPWORDS
                    if word in stw_set:
                        continue
                    
                    (prevNorm,num_distinct_words)=attributeHash[table_name][column_name]
                    
                    IAF = wordHash[word][0]
                    
                    Norm = prevNorm + IAF
                    
                    attributeHash[table_name][column_name]=(Norm,num_distinct_words)
                    

    print ('NORMS OF ATTRIBUTES PROCESSED')

#processNormsOfAttributes(wordHash,attributeHash)


# In[5]:


def wordNetSimilarity(wordA,wordB):
    
    A = set(wn.synsets(wordA))
    B = set(wn.synsets(wordB))
    
    similarities = [0]
    for (sense1,sense2) in itertools.product(A,B):        
        similarities.append(wn.wup_similarity(sense1,sense2) or 0)
        similarities.append(wn.path_similarity(sense1,sense2) or 0)
    return max(similarities)

def jaccard_similarity(wordA,wordB):
    
    A = set(wordA)
    B = set(wordB)
    
    return len(A & B ) / len(A | B)
    
def wordSimilarity(wordA,wordB):
    return max( (jaccard_similarity(wordA,wordB),wordNetSimilarity(wordA,wordB)) )


# In[102]:


def getQuerySets():
    QuerySet = []
    with open('querysets/queryset_imdb_martins.txt') as f:
        for line in f.readlines():
            
            #The line bellow Remove words not in OLIVEIRA experiments
            Q = [word.strip(string.punctuation) for word in line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set]  
            
            #Q = [word.strip(string.punctuation) for word in line.split() if word not in stw_set]  
            
            QuerySet.append(Q)
    return QuerySet
        
QuerySet = getQuerySets()
QuerySet


# In[7]:


def SchSFind(Q,threshold):
    S = []
    SchemaQuery = set()
    
    for (position,keyword) in enumerate(Q):
        for (table,values) in attributeHash.items():
            
            sim = wordSimilarity(keyword,table)
            if sim >= threshold:
                SchemaQuery.add(keyword)
                S.append( (table,'*',{keyword},position,sim) )
            
            for attribute in values.keys():
                
                if(attribute=='id'):
                    continue
                
                sim = wordSimilarity(keyword,attribute)
                
                if sim >= threshold:
                    SchemaQuery.add(keyword)
                    S.append( (table,attribute,{keyword},position,sim) )
    
    S = SchSInter(S)
    
    #The line below show similarity
    #Sq = {(table,attribute,frozenset(keywords),sim) for (table,attribute,keywords,position,sim) in S}
    
    Sq = {(table,attribute,frozenset(keywords)) for (table,attribute,keywords,position,sim) in S}
    
    Q = [element for element in Q if element not in SchemaQuery]
    
    return (Q,list(SchemaQuery),Sq)

def SchSInter(S):
    
    Scurr= S.copy()
    
    somethingChanged = False

    combinations = [x for x in itertools.combinations(Scurr,2)]
    
    for ( A , B ) in combinations:    
    
        (tableA,attributeA,wordsA,positionA,simA) = A
        (tableB,attributeB,wordsB,positionB,simB) = B
        
        
        if tableA == tableB and abs(positionA-positionB)<=1:
            AB = (tableA, '*' , wordsA | wordsB, max((positionA,positionB)) , max((simA,simB)) )
            
            Scurr.remove(A)
            Scurr.remove(B)
            Scurr.append(AB)
            
            somethingChanged = True 
            
    if somethingChanged:
        return SchSInter(Scurr)
    
    return Scurr

'''
#Q = ['actor','russel','crowe','gladiator','char','name']
Q = ['title', 'atticus', 'finch']

print(Q,'\n\n')
(Q,SchemaQuery,Sq) = SchSFind(Q,1)
pp(Sq)
print('\n\n\n\n=====================================================================\n')
Q,SchemaQuery
'''


# In[8]:


S = [('person', '*', {'actor'}, 0, 0.8),
 ('char', '*', {'char'}, 4, 1.0),
 ('movie', 'title', {'name'}, 5, 0.9230769230769231),
 ('char', 'name', {'name'}, 5, 1.0),
 ('person', 'name', {'name'}, 5, 1.0)]

SchSInter(S)


# In[9]:


def SMGen(SchemaQuery,Sq):
    #Input:  A keyword query Q, The set of non-empty non-free tuple-sets Rq
    #Output: The set Mq of query matches for Q
    
    '''
    Query match is a set of tuple-sets that, if properly joined,
    can produce networks of tuples that fulfill the query. They
    can be thought as the leaves of a Candidate Network.
    
    '''
    
    SMq = []
    for i in range(1,len(SchemaQuery)+1):
        for subset in itertools.combinations(Sq,i):
            if(MinimalCover(subset,SchemaQuery)):
                SMq.append(set(subset))
    return SMq


def MinimalCover(MC, Q):
    #Input:  A subset MC (Match Candidate) to be checked as total and minimal cover
    #Output: If the match candidate is a TOTAL and MINIMAL cover

    '''
    Total:   every keyword is contained in at least one tuple-set of the match
    
    Minimal: we can not remove any tuple-set from the match and still have a
             total cover.    
    '''
    Subset = [termset for table,attribute,termset in MC]
    u = set().union(*Subset)    
    
    isTotal = (u == set(Q))
    for element in Subset:
        
        new_u = list(Subset)
        new_u.remove(element)
        
        new_u = set().union(*new_u)
        
        if new_u == set(Q):
            return False
    
    return isTotal
'''
SMq = SMGen(SchemaQuery,Sq)
print (len(SMq),'Schema MATCHES CREATED')
for SM in SMq:
    print(SM,'\n\n')
'''


# In[10]:


def TSFind(Q):
    #Input:  A keyword query Q=[k1, k2, . . . , km]
    #Output: Set of non-free and non-empty tuple-sets Rq

    '''
    The tuple-set Rki contains the tuples of Ri that contain all
    terms of K and no other keywords from Q
    '''
    
    #Part 1: Find sets of tuples containing each keyword
    global P
    P = {}
    for keyword in Q:
        tupleset = set()
        
        if keyword not in wordHash:
            continue
        
        for (table,attributes) in wordHash.get(keyword)[1].items():
            for (attribute,ctids) in attributes.items():
                for ctid in ctids:
                    tupleset.add( (table,attribute,ctid) )
        P[frozenset([keyword])] = tupleset
    
    #Part 2: Find sets of tuples containing larger termsets
    P = TSInter(P)

    #Part 3:Build tuple-sets
    Rq = set()
    for keyword , tuples in P.items():
        for (table,attribute,ctid) in tuples:
            Rq.add( (table,attribute,keyword) )
    print ('TUPLE SETS CREATED')
    return Rq


def TSInter(P):
    #Input: A Set of non-empty tuple-sets for each keyword alone P 
    #Output: The Set P, but now including larger termsets (process Intersections)

    '''
    Termset is any non-empty subset K of the terms of a query Q        
    '''
    
    Pprev = {}
    Pprev=copy.deepcopy(P)
    Pcurr = {}

    combinations = [x for x in itertools.combinations(Pprev.keys(),2)]
    for ( Ki , Kj ) in combinations:
        Tki = Pprev[Ki]
        Tkj = Pprev[Kj]
        
        X = Ki | Kj
        Tx = Tki & Tkj        
        
        if len(Tx) > 0:            
            Pcurr[X]  = Tx            
            Pprev[Ki] = Tki - Tx         
            Pprev[Kj] = Tkj - Tx
            
    if Pcurr != {}:
        Pcurr = copy.deepcopy(TSInter(Pcurr))
        
    #Pprev = Pprev U Pcurr
    Pprev.update(Pcurr)     
    return Pprev       

'''
Rq = TSFind(Q)
pp(Rq)
'''


# In[11]:


#Rq[frozenset({'denzel', 'washington'})]
#Mq = QMGen(Q,Rq)
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
                Mq.append(set(subset))
    return Mq
'''
Mq = QMGen(Q,Rq)
print (len(Mq),'QUERY MATCHES CREATED')
for M in Mq:
    print(M,'\n\n')
'''


# In[12]:


def FMGen(SMq,QueryMatch):
    
    if len(SMq)==0:
        return [QueryMatch]
    
    FMq = []
    
    for SchemaMatch in SMq:
        FullMatch = QueryMatch.copy()

        for SchemaSet in  SchemaMatch:

            schemaTable = SchemaSet[0]

            schemaTableFound = False

            for TupleSet in QueryMatch:

                if type(TupleSet) is str:
                    tupleSetTable = TupleSet
                else:
                    tupleSetTable = TupleSet[0]

                if schemaTable ==tupleSetTable:
                    schemaTableFound = True

            if not schemaTableFound:
                FullMatch.add( (schemaTable,'__search_id',frozenset([''])) )

        FMq.append(FullMatch) 
            
    return FMq
                    


# In[13]:


def getSchemaGraph():
    #Output: A Schema Graph G  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }
    
    
    G = {} 
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname!='pg_catalog' AND schemaname !='information_schema';")
    for table in cur.fetchall():
        G.setdefault(table[0],{})
    
    sql = "SELECT DISTINCT                 tc.table_name, kcu.column_name,                 ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name             FROM information_schema.table_constraints AS tc              JOIN information_schema.key_column_usage AS kcu                 ON tc.constraint_name = kcu.constraint_name             JOIN information_schema.constraint_column_usage AS ccu                 ON ccu.constraint_name = tc.constraint_name             WHERE constraint_type = 'FOREIGN KEY'"
    cur.execute(sql)
    relations = cur.fetchall()
    
    for (table,column,foreign_table,foreign_column) in relations:
        G[table][foreign_table] = (1,column, foreign_column)
        G[foreign_table][table] = (-1,foreign_column,column)
    print ('SCHEMA CREATED')
    return G

def MatchGraph(Rq, G, M):
    #Input:  The set of non-empty non-free tuple-sets Rq,
    #        The Schema Graph G,
    #        A Query Match M
    #Output: A Schema Graph Gts  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }

    '''
    A Match Subgraph Gts[M] is a subgraph of G that contains:
        The set of free tuple-sets of G
        The query match M
    '''
    
    Gts = copy.deepcopy(G)
    
    tables = set()
    #Insert non-free nodes
    for (table ,attribute, keywords) in M:
        Gts[(table,attribute,keywords)]=copy.deepcopy(Gts[table])
        for foreign_table , (direction,column,foreign_column) in Gts[(table,attribute,keywords)].items():
            Gts[foreign_table][(table,attribute,keywords)] = (direction*(-1),foreign_column,column)

    return Gts 

G = getSchemaGraph()
pp(G)
'''
print ('\nEXAMPLE OF MATCH GRAPH')
Gts = MatchGraph(Rq, G, Mq[0])
pp(Gts)
'''


# In[14]:


xCn = [('person', 'name', frozenset({'ford', 'harrison'})),
 'casting',
 ('person', 'name', frozenset({'lucas', 'george'}))]



xGts = {'casting': {'char': (1, 'person_role_id', 'id'),
             'movie': (1, 'movie_id', 'id'),
             'person': (1, 'person_id', 'id'),
             'role': (1, 'role_id', 'id'),
             ('person', 'name', frozenset({'ford', 'harrison'})): (1,
                                                                   'person_id',
                                                                   'id'),
             ('person', 'name', frozenset({'lucas', 'george'})): (1,
                                                                  'person_id',
                                                                  'id')},
 'char': {'casting': (-1, 'id', 'person_role_id')},
 'movie': {'casting': (-1, 'id', 'movie_id')},
 'person': {'casting': (-1, 'id', 'person_id')},
 'role': {'casting': (-1, 'id', 'role_id')},
 ('person', 'name', frozenset({'ford', 'harrison'})): {'casting': (-1,
                                                                   'id',
                                                                   'person_id')},
 ('person', 'name', frozenset({'lucas', 'george'})): {'casting': (-1,
                                                                  'id',
                                                                  'person_id')}}

def isJNTSound(Gts,Ji):
    if len(Ji)<3:
        return True
    
    for i in range(len(Ji)-2):
        
        if type(Ji[i]) is str:
            tableA = Ji[i]
        else:
            tableA = Ji[i][0]
            
        if type(Ji[i+2]) is str:
            tableB = Ji[i+2]
        else:
            tableB = Ji[i+2][0]
            
            
            
        if tableA==tableB:
            edge_info = Gts[Ji[i]][Ji[i+1]]
            if(edge_info[0] == -1):
                return False
    return True

isJNTSound(xGts,xCn)


# In[40]:


F = deque()
F.append(1)
F.append(2)
F.append(3)
F.pop()


# In[64]:


def containsMatch(Ji,M):
    for relation in M:
        if relation not in Ji:
            return False
    return True

def isJNTSound(Gts,Ji):
    if len(Ji)<3:
        return True
    
    for i in range(len(Ji)-2):
        
        if type(Ji[i]) is str:
            tableA = Ji[i]
        else:
            tableA = Ji[i][0]
            
        if type(Ji[i+2]) is str:
            tableB = Ji[i+2]
        else:
            tableB = Ji[i+2][0]
            
            
            
        if tableA==tableB:
            edge_info = Gts[Ji[i]][Ji[i+1]]
            if(edge_info[0] == -1):
                return False
    return True

def SingleCN(FM,Gts,Tmax):    
    '''
    print('================================================================================\nSINGLE CN')
    print('Tmax ',Tmax)
    print('FM')
    pp(FM)
    
    print('\n\nGts')
    pp(Gts)
    print('\n\n')
    '''
    F = deque()

    first_element = list(FM)[0]
    J = [first_element]
    
    if len(FM)==1:
        return J
    
    F.append(J)
    
    while F:
        J = F.popleft()           
        u = J[-1]
        '''
        print('--------------------------------------------\nParctial CN')
        print('J ',J,'\n')
        
        print('\nAdjacents:')
        pp(Gts[u].items())
        '''
        for (adjacent,edge_info) in Gts[u].items():
            if (type(adjacent) is str) or (adjacent not in J):
                Ji = J + [adjacent]
                if (Ji not in F) and (len(Ji)<Tmax) and (isJNTSound(Gts,Ji)):
                    if(containsMatch(Ji,FM)):
                        '''
                        print('--------------------------------------------\nGenerated CN')
                        print('J ',Ji,'\n')
                        '''
                        return Ji
                    else:
                        F.append(Ji)

def MatchCN(G,Rq,Mq,SMq):    
    
    Cns = []                        
    for M in Mq: 
        
        FullMatches = FMGen(SMq,M) 
        
        for FM in FullMatches: 
            
            Gts =  MatchGraph(Rq,G,FM)
            
            Cn = SingleCN(FM,Gts,10)

            if(Cn is not None):
                Cns.append( (Cn,Gts,FM) )
    return Cns

'''
Cns = MatchCN(G,Rq,Mq,SMq)

for (Cn,Gts,FM) in Cns:
    print('\n\n--------------------------------------------------\nGts\n')
    pp(Gts)
    print('\nFM\n')
    pp(FM)
    print('\nCN\n')
    pp(Cn)
'''
#Cn=[('movie_info', frozenset({'gangster'})), 'title', ('cast_info', frozenset({'washington', 'denzel'}))] 
'''

Gts = {'casting': {'char': (1, 'person_role_id', 'id'),
             'movie': (1, 'movie_id', 'id'),
             'person': (1, 'person_id', 'id'),
             'role': (1, 'role_id', 'id'),
             ('char', 'name', frozenset({'finch', 'atticus'})): (1,
                                                                 'person_role_id',
                                                                 'id'),
             ('movie', '__search_id', frozenset({''})): (1, 'movie_id', 'id')},
 'char': {'casting': (-1, 'id', 'person_role_id')},
 'movie': {'casting': (-1, 'id', 'movie_id')},
 'person': {'casting': (-1, 'id', 'person_id')},
 'role': {'casting': (-1, 'id', 'role_id')},
 ('char', 'name', frozenset({'finch', 'atticus'})): {'casting': (-1,
                                                                 'id',
                                                                 'person_role_id')},
 ('movie', '__search_id', frozenset({''})): {'casting': (-1, 'id', 'movie_id')}}

FM = {('char', 'name', frozenset({'finch', 'atticus'})),
 ('movie', '__search_id', frozenset({''}))}

SingleCN(FM,Gts,10)
'''


# In[16]:



def CNRank(Cns,mi):
 Ranking = []
 for (Cn,Gts,M) in Cns:
     cosprod = 1
     
     for relation in Cn:
         if(type(relation) is str):
             continue

         (table,attribute,predicates) = relation
         
         if predicates == frozenset(['']):
             continue
         
         (norm_attribute,distinct_terms) = attributeHash[table][attribute]
         
         wsum = 0
         
         for term in predicates:
             
             IAF = wordHash[term][0] 
             
             ctids = wordHash[term][1][table][attribute]
             fkj = len(ctids)
             
             if fkj>0:
                 
                 TF = log1p(fkj) / log1p(distinct_terms)
                 
                 wsum = wsum + TF*IAF
                     
         cos = wsum/norm_attribute
         cosprod *= cos

     score = mi * cosprod * 1/len(Cn)
     Ranking.append((Cn,Gts,M,score))
     
 return sorted(Ranking,key=lambda x: x[-1],reverse=True)


# In[17]:


'''
RankedCns=CNRank(Cns,2700000000000)

for (Cn,Gts,M,Score) in RankedCns:
    print(Score)
    print(Cn)
'''


# In[18]:


def getSQLfromCN(Gts,Cn):
    #print('CN:\n',Cn)
    
    selected_attributes = [] 
    tables = []
    conditions=[]
    relationships = []
    
    for i in range(len(Cn)):
        
        if(type(Cn[i]) is str):
            tableA = Cn[i]
            attrA=''
            keywords=[]
        else:
            (tableA,attrA,keywords) = Cn[i]  
                
        A = 't' + str(i)
        
        
        if(attrA != ''):
            selected_attributes.append(A +'.'+ attrA)
        
        tables.append(tableA+' '+A)
            
        #tratamento de keywords
        for term in keywords:
            condition = 'CAST('+A +'.'+ attrA + ' AS VARCHAR) ILIKE \'%' + term.replace("'","''") + '%\''
            conditions.append(condition)
        
        if(i<len(Cn)-1):
            if(type(Cn[i+1]) is str):
                tableB = Cn[i+1]
                attrB=''
            else:
                (tableB,attrB,keywords)=Cn[i+1]
                  
            B = 't'+str(i+1)
            
            edge_info = Gts[Cn[i]][Cn[i+1]]
            (direction,joining_attrA,joining_attrB) = edge_info
            
            relationships.append( (A,B) )
            
            condition = A + '.' + joining_attrA + ' = ' + B + '.' + joining_attrB         
            conditions.append(condition)
    
    tables_id = ['t'+str(i)+'.__search_id' for i in range(len(tables))]
    
    relationshipsText = ['('+str(a)+'.__search_id'+','+str(b)+'.__search_id'+')' for (a,b) in relationships]
    
    
    sqlText = 'SELECT '
    sqlText +=' ('+', '.join(tables_id)+') AS Tuples '
    if len(relationships)>0:
        sqlText +=', ('+', '.join(relationshipsText)+') AS Relationships'
        
    sqlText += ' , ' + ' , '.join(selected_attributes)
    
    sqlText +=' FROM ' + ', '.join(tables)
    sqlText +=' WHERE ' + ' AND '.join(conditions)
    '''
    print('SELECT:\n',selected_attributes)
    print('TABLES:\n',tables)
    print('CONDITIONS:')
    pp(conditions)
    print('RELATIONSHIPS:')
    pp(relationships)
    '''    
    #print('SQL:\n',sql)
    return sqlText

'''
print('CN:\n',Cns[0][0])
getSQLfromCN(Cns[0][1],Cns[0][0])
'''


# In[100]:


def getGoldenStandards():
    goldenStandards = {}
    for i in range(1,51):
        filename = 'golden_standards/0'+str(i).zfill(2) +'.txt'
        with open(filename) as f:

            listOfTuples = []
            Q = ()
            for i, line in enumerate(f.readlines()):
              
                line_without_comment =line.split('#')[0]
                
                if(i==2):
                    comment_of_line = line.split('#')[1]
                    
                    #The line bellow Remove words not in OLIVEIRA experiments
                    Q = tuple([word.strip(string.punctuation) for word in comment_of_line.split() if word not in ['title','dr.',"here's",'char','name'] and word not in stw_set])

                    #Q = tuple([word.strip(string.punctuation) for word in comment_of_line.split() if word not in stw_set])
                
                if line_without_comment:                    
                    
                    relevantResult = eval(line_without_comment)
                    listOfTuples.append( relevantResult )
            
            goldenStandards[Q]=listOfTuples
            
    return goldenStandards


goldenStandards = getGoldenStandards()
goldenStandards


# In[20]:


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
    
    for goldenRelationship in goldenStandard[1]:
        
        (A,B) = goldenRelationship
        
        if (A,B) not in relationshipsOfCNResult and (B,A) not in relationshipsOfCNResult:
            return False
        
    return True


def evaluanteResult(Result,Query):
    
    goldenStandard = goldenStandards[tuple(Query)]
    
    for goldenRow in goldenStandard:

        found = False

        for row in Result:
            if evaluateCN(row,goldenRow):
                found = True

        if not found:
            return False
        
    return True
            
            
x=[('(39292828,5360667,21231023)', '("(39292828,5360667)","(5360667,21231023)")', 'Hamill, Mark', 'Luke Skywalker'), ('(39292828,5360749,21231023)', '("(39292828,5360749)","(5360749,21231023)")', 'Hamill, Mark', 'Luke Skywalker'), ('(39292828,5360752,21231023)', '("(39292828,5360752)","(5360752,21231023)")', 'Hamill, Mark', 'Luke Skywalker'), ('(39292828,5360753,21231023)', '("(39292828,5360753)","(5360753,21231023)")', 'Hamill, Mark', 'Luke Skywalker')]
q = ['hamill', 'skywalker']

def normalizeResult(ResultFromDatabase):
    normalizedResult = []
    
    for row in ResultFromDatabase:        
        if type(row[0]) == int:
            tuples = [row[0]]
        else:
            tuples = eval(str(row[0]))
        
        try:
            relationships = eval(row[1])
            relationships = [eval(element) for element in relationships]
        except:
            relationships = []
            
        
        normalizedResult.append( (tuples,relationships) )
    return normalizedResult

'''
normX = normalizeResult(x)

evaluanteResult(normX,q)
'''


# In[67]:


def getRelevantPosition(RankedCns,Q):
    countCNsWithResults = 0
    for (position,(Cn,Gts,M,score)) in enumerate(RankedCns):
        
        if position > 20:
            return (-1,-1)
        '''
        print('EXECUTING CN:\n')
        pp(Cn)
        '''
        SQL = getSQLfromCN(Gts,Cn)
        
        #print(SQL)
        
        cur.execute(SQL)
        Results = cur.fetchall()

        if len(Results)>0:
            countCNsWithResults +=1 
        
        NResults = normalizeResult(Results)

        Relevance = evaluanteResult(NResults,Q)

        if Relevance == True:
            return (position+1 , countCNsWithResults)
        
        gc.collect()

    return (-1,-1)


# In[88]:


QuerySets = getQuerySets()
QuerySets[35:36]


# In[96]:


def preProcessing():
    global wordHash
    global attributeHash
    (wordHash,attributeHash) = createInvertedIndex()
    processIAF(wordHash,attributeHash)
    processNormsOfAttributes(wordHash,attributeHash)
    print('PRE-PROCESSING STAGE FINISHED')
    

def main():   
    QuerySets = getQuerySets()
    goldenStandards = getGoldenStandards()
    
    EVALUATION_RESULTS = []
    
    for (i,Q) in enumerate(QuerySets[0:30]+QuerySets[35:]):
        oritinalQ = Q.copy()
        
        print('QUERY-SET ',Q,'\n')
        
        print('FINDING SCHEMA-SETS')
        (Q,SchemaQuery,Sq) = SchSFind(Q,1)
        print(len(Sq),' SCHEMA-SETS CREATED\n')
        
        print('GENERATING SCHEMA MATCHES')
        SMq = SMGen(SchemaQuery,Sq)
        print(len(SMq),' SCHEMA MATCHES CREATED')
        
        print('\nNEW QUERY-SET ',Q,'\n')
        
        print('FINDING TUPLE-SETS')
        Rq = TSFind(Q)
        print(len(Rq),'TUPLE-SETS CREATED\n')
        
        print('GENERATING QUERY MATCHES')
        Mq = QMGen(Q,Rq)
        print (len(Mq),'QUERY MATCHES CREATED\n')
        '''
        for M in Mq[:20]:
            pp(M)
            print('\n\n')
        '''
        print('GENERATING CANDIDATE NETWORKS')
        G = getSchemaGraph()
        
        Cns = MatchCN(G,Rq,Mq,SMq)
        
        print (len(Cns),'CANDIDATE NETWORKS CREATED\n')
        
        '''
        for Cn in Cns[:20]:
            pp(Cn[0])
            print('\n\n')
            #pp(Cn[1])
            #print('\n\n\n==================================================================================\n')
        '''
        print('RANKING CANDIDATE NETWORKS')
        RankedCns=CNRank(Cns,2700000000000)
        
        print('\nEVALUATING ANSWER')
        Position = getRelevantPosition(RankedCns,oritinalQ)
        print('\nRELEVANT ANSWER IN POSITION ', Position,'\n\n')
        
        EVALUATION_RESULTS.append(Position)
        
        print('EvaluationResults: \n',EVALUATION_RESULTS)
        
        gc.collect()
        
    return EVALUATION_RESULTS


# In[24]:


preProcessing()


# In[103]:


main()


# In[106]:


x = [(1, 1),
 (1, 1),
 (1, 1),
 (2, 2),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (1, 1),
 (2, 2),
 (1, 1),
 (1, 1),
 (7, 3),
 (13, 7),
 (19, 14),
 (11, 6),
 (14, 10),
 (4, 3),
 (13, 5),
 (3, 3),
 (14, 1),
 (-1, -1),
 (1, 1),
 (1, 1),
 (-1, -1),
 (-1, -1),
 (1, 1),
 (1, 1),
 (-1, -1),
 (-1, -1),
 (4, 2),
 (-1, -1),
 (14, 1),
 (8, 1),
 (8, 1),
 (6, 2),
 (1, 1)]
y = [positionWithResults for (position,positionWithResults) in x]
y

