import psycopg2
from psycopg2 import sql
from pprint import pprint as pp
from collections import defaultdict
import string
import itertools
import copy

# Connect to an existing database
conn = psycopg2.connect("dbname=imdb user=postgres")

# Open a cursor to perform database operations
cur = conn.cursor()

wordHash = {}


def createInvertedList():
    #Output: wordHash (Term Index) with this structure below
    #map['word'] = [ 'table': ( {column} , ['ctid'] ) ]

    '''
    The Term Index is built in a preprocessing step that scans only
    once all the relations over which the queries will be issued.
    '''

    # Get list of tablenames
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname!='pg_catalog' AND schemaname !='information_schema';")
    for table in cur.fetchall():
        table_name = table[0]
        
        print('INDEXING TABLE ',table_name)
        
        #Get all tuples for this tablename
        cur.execute(
            sql.SQL("SELECT ctid, * FROM {};").format(sql.Identifier(table_name))
            #NOTE: sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
        )

        for row in cur.fetchall():
            for column in range(1,len(row)):
                column_name = cur.description[column][0]
                ctid = row[0]

                #NOTE: Need to remove stopwords
                for word in [word.strip(string.punctuation) for word in str(row[column]).lower().split()]:

                    #If word entry doesn't exists, it will be inicialized (setdefault method),
                    #Append the location for this word
                    wordHash.setdefault(word, {})
                    wordHash[word].setdefault( table_name , (set(),[]) )
                    wordHash[word][table_name][0].add(column_name)
                    wordHash[word][table_name][1].append(ctid)
    print ('INVERTED INDEX CREATED')


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
        for table, (columns,ctids) in wordHash.get(keyword).items():
            for ctid in ctids:
                tupleset.add( (table,ctid) )
        P[frozenset([keyword])] = tupleset

    #Part 2: Find sets of tuples containing larger termsets
    P = TSInter(P)

    #Part 3:Build tuple-sets
    Rq = set()
    for keyword , tuples in P.items():
        for (table,ctid) in tuples:
            Rq.add( (table,keyword) )
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
    for ( Ki , Kj ) in combinations[0:4]:
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
                Mq.append(subset)
    print ('QUERY MATCHES CREATED')
    return Mq


def MinimalCover(MC, Q):
    #Input:  A subset MC (Match Candidate) to be checked as total and minimal cover
    #Output: If the match candidate is a TOTAL and MINIMAL cover

    '''
    Total:   every keyword is contained in at least one tuple-set of the match
    
    Minimal: we can not remove any tuple-set from the match and still have a
             total cover.    
    '''
    
    Subset = [termset for table,termset in MC]
    u = set().union(*Subset)    
    
    isTotal = (u == set(Q))
    for element in Subset:
        
        new_u = list(Subset)
        new_u.remove(element)
        
        new_u = set().union(*new_u)
        
        if new_u == set(Q):
            return False
    
    return isTotal


def getSchemaGraph():
    #Output: A Schema Graph G  with the structure below:
    # G['node'] = edges
    # G['table'] = { 'foreign_table' : (direction, column, foreign_column) }
    
    G = {} 
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname!='pg_catalog' AND schemaname !='information_schema';")
    for table in cur.fetchall():
        G.setdefault(table[0],{})
    
    sql = "SELECT DISTINCT \
                tc.table_name, kcu.column_name, \
                ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name \
            FROM information_schema.table_constraints AS tc  \
            JOIN information_schema.key_column_usage AS kcu \
                ON tc.constraint_name = kcu.constraint_name \
            JOIN information_schema.constraint_column_usage AS ccu \
                ON ccu.constraint_name = tc.constraint_name \
            WHERE constraint_type = 'FOREIGN KEY'"
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
    for (table , keywords) in M:
        Gts[(table,keywords)]=copy.deepcopy(Gts[table])
        for foreign_table , (direction,column,foreign_column) in Gts[(table,keywords)].items():
            Gts[foreign_table][(table,keywords)] = (direction*(-1),foreign_column,column)

    return Gts 

def isJNTTotal(Gts,Ji,Q):
    keywords = set()
    for node in Ji:
        if (type(node) is str):
            continue
        keywords = keywords | node[1]
    return (keywords == set(Q))

def isJNTSound(Gts,Ji):
    if len(Ji)<3:
        return True
    for i in range(len(Ji)-2):
        if (Ji[i],)[0] == (Ji[i+2],)[0]:
            edge_info = Gts[Ji[i]][Ji[i+1]]
            if(edge_info[0] == -1):
                return False
    return True

#CN = SingleCN(Mq[0],Gts,5,Q)
def SingleCN(M,Gts,Tmax,Q):
    from queue import deque
    F = deque()
    J = [M[0]]
    F.append(J)
    while F:
        J = F.pop()
        u = J[-1]
        for (adjacent,edge_info) in Gts[u].items():
            if (type(adjacent) is str) or (adjacent not in J):
                Ji = J + [adjacent]
                if (Ji not in F) and (len(Ji)<Tmax) and (isJNTSound(Gts,Ji)):
                    if(isJNTTotal(Gts,Ji,Q)):
                       return Ji
                    else:
                       F.append(Ji)

def getSQLfromCN(Gts,Cn):
    sql = 'SELECT * FROM ' + (Cn[0],)[0]
    for i in range(len(Cn)):
        

createInvertedList()
Q = ['denzel','washington','gangster']
Rq = TSFind(Q)
Mq = QMGen(Q,Rq)
G = getSchemaGraph()
Gts =  MatchGraph(Rq,G,Mq[0])
