import pandas as pd
from pandas import DataFrame
from mysql.connector.cursor import MySQLCursor
from zlib import crc32

def dfq(q: str, cur: MySQLCursor) -> DataFrame:
    '''
    dfq - "Data Frame Query"
    helper function to pass query strings to, and return pandas data frames

    Params:
    * q (str) = mySql query
    * cur = MySQLCursor object.  This should be created from an established database connection
    '''
    cur.execute(q)
    res = cur.fetchall()
    cols = [x[0] for x in cur.description]
    data = [{x[0]:x[1] for x in zip(cols, r)} for r in res]
    return pd.DataFrame(data)

def patientIdToFloat(s, encoding="utf-8"):
    return float(crc32(s.encode(encoding)) & 0xffffffff) / 2**32