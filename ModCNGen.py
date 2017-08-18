import psycopg2
from psycopg2 import sql
import pprint
from collections import defaultdict

# Connect to an existing database
conn = psycopg2.connect("dbname=imdb user=postgres")

# Open a cursor to perform database operations
cur = conn.cursor()

def createInvertedList():
    #Creta a index list ( 'index':[('table','column','ctid')] )

    # Get list of tablenames
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname!='pg_catalog' AND schemaname !='information_schema';")

    for table in cur.fetchall():
        table_name = table[0]

        cur.execute(
            #sql.SQL is needed to specify this parameter as table name (can't be passed as execute second parameter)
            sql.SQL("SELECT ctid, * FROM {};").format(sql.Identifier(table_name))
        )

        for row in cur.fetchall():

            for column in range(1,len(row)):
                print('column: ',column)
                column_name = cur.description[column][0]
                print('column_name: ',column_name)
                ctid = row[0]

                print('row:',row)
                print('row[column]:',row[column])
                for word in row[column].split(' '):
                    print('word: ',word)
                    term_index = (word,table_name,column_name,ctid)
                    print(term_index)

        '''
        for row in cur.fetchall():



        #Create inverted index
        index = defaultdict(list)
        for i, tokens in enumerate(rows[0:2]):
            for token in tokens:
                index[token].append(i)

        print(index['actor'])
        '''
        break

createInvertedList()

# Make the changes to the database persistent
conn.commit()

# Close communication with the database
cur.close()
conn.close()
