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
    for keyword , (table_column,ctid) in P.items():
        Rq
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
    Mq = set()
    for i in range(1,len(Q)+1):
        for subset in itertools.combinations(Rq.items(),i):
            if( isMinimalCover(subset)):
                Mq.add(subset)
'''
def MinimalCover(Q):
    for i in range(1,len(Q)+1):
        for subset in itertools.combinations(Rq.items(),i):
'''         
            
createInvertedList()
print(loadQuerySets())

'''
def getTupleSets(queryset):
    for term in queryset:
        print('term ',term)
        for table_column,freq_ctids in wordHash.get(term).items():
            print('table_column ',table_column,'freq_ctids ',ctids)
            for ctid in ctids:
                print('ctid ', ctid)
                tupleHash.setdefault(ctid, {})
                tupleHash[ctid].setdefault(term, []).append(table_column)
    
    for ctid, term_list_of_table_column in tupleHash.items():
        new_key = []
        new_value = {}
        for term, list_of_table_column in term_list_of_table_column.items():
            for table_column in list_of_table_column:
                print('term ',term,'table_column ',table_column)
                new_key.append(term)
                new_value.setdefault(table_column, [0,]).append(ctid)
                new_value[table_column][0]+=1
        print('new_key ',new_key, 'new_value ',new_value)
        #new_key  ['denzel', 'washington'] new_value  {('name', 'name'): [2, '(1257,105)', '(1257,105)']}
        break
'''


'''
# Make the changes to the database persistent
conn.commit()

# Close communication with the database
cur.close()
conn.close()
'''
