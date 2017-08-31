import psycopg2
from psycopg2 import sql
import pprint
from collections import defaultdict
import string
import itertools


# Connect to an existing database
conn = psycopg2.connect("dbname=imdb user=postgres")

# Open a cursor to perform database operations
cur = conn.cursor()

wordHash = {}
tupleHash ={}

def createInvertedList():
    #Create an index list with map['word'] = [ ('table','column') : ['ctid'] ]

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
                #print('column: ',column)
                column_name = cur.description[column][0]
                #print('column_name: ',column_name)
                ctid = row[0]

                #print('row[column]:',row[column])  
                for word in [word.strip(string.punctuation) for word in str(row[column]).lower().split()]:
                    #print('word: ',word)
                    term_index = (table_name,column_name,ctid)
                    #print(term_index)

                    #If word entry doesn't exists, it will be inicialized (setdefault method),
                    #Append the location for this word
                    wordHash.setdefault(word, {})
                    wordHash[word].setdefault( (table_name,column_name) , [] ).append(ctid)
    print ('INVERTED INDEX CREATED')


def loadQuerySets():
    with open('querysets/queryset_imdb_inex.txt') as f:
        querysets = []
        for line in f.readlines():
            querysets.append(line.lower().split())
        return querysets

'''
    Termset is any non-empty subset K of the terms of a query Q
    
    Query match is a set of tuple-sets that, if properlyjoined,
    can produce networks of tuples that fulfill the query. They
    can be thought as the leaves of a Candidate Network.

        
'''
def TSFind(Q):
    #Part 1: Find sets of tuples containing each keyword
    P = {}
    for keyword in Q:

        tupleset = []
        
        for table_column, ctids in wordHash.get(keyword).items():
            for ctid in ctids:
                tupleset.append( (table_column,ctid) )
        
        P[frozenset([keyword])] = set(tupleset)

    #Part 2: Find sets of tuples containing larger termsets
    P = TSInter(P)

    return P
    '''
    #Part 3: Build tuple-sets
    Rq = {}
    for keyword , tupleset in P.items():

        Rq.setdefault(keyword,{})
        
        for (table_column,ctid) in tupleset:
    '''
            


def TSInter(P):
    Pprev = P
    Pcurr = {}
    for ( (Ki,Tki), (Kj,Tkj) ) in itertools.combinations(P.items(),2):
        X = Ki | Kj
        Tx = Tki & Tkj
        if len(Tx) > 0:
            Pcurr[X]  = Tx
            Pprev[Ki] = Tki
            Pprev[Kj] = Tkj
    if Pcurr != {}:
        Pcurr = TSInter(Pcurr)

    #Pprev = Pprev U Pcurr
    Pprev.update(Pcurr)     

    return Pprev


def QMGen(Q,Rq):
    Mq = []
    for i in range(1,len(Q)+1):
        for subset in itertools.combinations(Rq.keys(),i):
            print('---------------------------------------------------')
            pprint.pprint(subset)
            if(MinimalCover(subset,Q)):
                print('TOTAL MINIMAL COVER\n==============================================')                
                Mq.append(subset)
                print('\n')
    return Mq


def MinimalCover(Subset, Q):
    u = set().union(*Subset)

    isTotal = (u == set(Q))
    for element in Subset:
        new_u = list(Subset)
        new_u.remove(element)
        pprint.pprint(new_u)
        
        new_u = set().union(*new_u)
        
        if new_u == set(Q):
            return False
    
    return isTotal


def getSchemaGraph():
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
    return G

#Gts = MatchGraphs(Rq,G,Mq[0])
def MatchGraphs(Rq, G, match):
    import copy
    Gts = copy.deepcopy(G)
    
    tables = set()
    #Insert non-free nodes
    for tupleset in match:
        for ( (table,column) , ctid) in Rq[tupleset]:
            tables.add( (table,tupleset) )

    #Update edges
    for (table,tupleset) in tables:
        Gts[(table,tupleset)]=copy.deepcopy(Gts[table])
        for foreign_table , (direction,column,foreign_column) in Gts[(table,tupleset)].items():
            Gts[foreign_table][(table,tupleset)] = (direction*(-1),foreign_column,column)
    return Gts


def MatchCN(Mq,G):
    C = []
    for M in Mq:
        Gts = MatchGraphs(Rq,G,M)
        Cn = SingleCN(M,Gts)
        C.append(Cn)
    return C

def SingleCN(M, Gts):
    #Input:  A query match M; A match graph Gts[M]
    #Output: A single candidate network C

    import queue
    F = queue.Queue()

    J = Gts
    
    

createInvertedList()
Q = ['denzel','washington','gangster']
Rq = TSFind(Q)
Mq = QMGen(Q,Rq)
G = getSchemaGraph()    

def pp(Object):
    pprint.pprint(Object)

'''
# Make the changes to the database persistent
conn.commit()

# Close communication with the database
cur.close()
conn.close()
'''
