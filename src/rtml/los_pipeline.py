from rtml.data_functions import dfq, patientIdToFloat
from mysql.connector.connection import MySQLConnection
from pandas import DataFrame
import numpy as np

class losPipeline:
    def __init__(self, cnx: MySQLConnection, db:str, patientId:str="PATIENT_ID"):
        self.cursor = cnx.cursor()
        self.cursor.execute(f'USE {db}')
        self.patientId = patientId
        self.cleanAdmitSql = '''clean_admits AS (
                    SELECT *, DATEDIFF(DISCHARGE_TIME, ADMIT_TIME) AS LOS
                    FROM encounters
                    WHERE PAT_CLASS = "Inpatient"
                    AND ADMIT_TIME IS NOT NULL
                    AND DISCHARGE_TIME IS NOT NULL
                )'''

    def trainTestValidationSplits(self, q: str=None, inPlace:bool=True, testPercent:float=0.2, validationFolds:int=3):
        '''
        trainTestValidationSplits:
            Creates lists of patient IDs for tests (a single list) and training (broken into roughly equal lists for each validation fold).

        params:
            * q (str): a query that creates a unique list of patient IDs.
            * inPlace (bool, optional): a boolean that indicates if you would like to simply create patient lists as internal variables.
        '''
        if not q:
            q='''
                SELECT DISTINCT PATIENT_ID 
                FROM encounters
                WHERE PAT_CLASS = "Inpatient"
                AND ADMIT_TIME IS NOT NULL
                AND DISCHARGE_TIME IS NOT NULL
            '''
        df = dfq(q, self.cursor)
        df['hash'] = df[self.patientId].map(lambda x: patientIdToFloat(x))

        test_ids = df[df['hash']>=(1-testPercent)][self.patientId].to_list()
        train_ids = [df[(df['hash']<((testPercent/validationFolds)*(1+x)))&(df['hash']>=(testPercent/validationFolds)*x)][self.patientId].to_list() for x in range(validationFolds)]
        if inPlace:
            self.test_ids = test_ids
            self.train_ids = train_ids
            return None
        else:
            return (test_ids, train_ids)

    def getOutcome(self, q:str=None, max_days:int=40)->DataFrame:
        '''
        getOutcome:
            returns a dataframe with the following columns:
            * PATIENT_ID - unique identifier number for a patient
            * ADMIT_TIME - Index time for the admission
            * LOS - Length of stay for the inpatient admission

        params:
            * q (str): a query to override the default.  What is here **should** work with RT extracts
            * max_days (int, nullable): Applies a cap to the LOS; pass in None to disable.

        returns:
            * a data frame indexed to patient_id and index time
        '''
        if not q:
            losCalc = 'DATEDIFF(DISCHARGE_TIME, ADMIT_TIME)'
            if max_days:
                losCalc = 'LEAST(' + losCalc + ', ' + str(max_days) + ")"
            q = f'''
                    SELECT PATIENT_ID, ADMIT_TIME, {losCalc} AS LOS
                    FROM encounters
                    WHERE PAT_CLASS = "Inpatient"
                    AND ADMIT_TIME IS NOT NULL
                    AND DISCHARGE_TIME IS NOT NULL
                '''
        return dfq(q, self.cursor).set_index(['PATIENT_ID', 'ADMIT_TIME'])

    def getPastUtil(self, q:str=None, lookback:int=365, suffix:str="_1Y")->DataFrame:
        '''
        TODO
        '''
        if not q:
            q = f'''
                WITH {self.cleanAdmitSql}, 
                admit_join AS (
                    SELECT 
                        l.PATIENT_ID, 
                        l.LOS,
                        l.ADMIT_TIME, l.DISCHARGE_TIME,
                        r.ADMIT_TIME AS Previous_Admit,
                        r.DISCHARGE_TIME AS Previous_Discharge,
                        r.HOSP_SERVICE,
                        DATEDIFF(l.ADMIT_TIME, r.DISCHARGE_TIME) AS Days_Since_Discharge,
                        r.LOS AS Prev_los
                    FROM clean_admits l
                    INNER JOIN clean_admits r
                    ON l.PATIENT_ID = r.PATIENT_ID
                    WHERE l.ADMIT_TIME > r.DISCHARGE_TIME
                ),
                prev_admits AS (
                    SELECT 
                        PATIENT_ID, 
                        ADMIT_TIME,
                        COUNT(DISTINCT Days_Since_Discharge) Number_Admissions{suffix},
                        COUNT(DISTINCT HOSP_SERVICE) Number_Admission_Types{suffix},
                        SUM(Prev_los) AS Inpatient_Days{suffix}
                    FROM admit_join
                    WHERE Days_Since_Discharge < {lookback}
                    GROUP BY PATIENT_ID, ADMIT_TIME
                )
                SELECT 
                    a.PATIENT_ID, 
                    a.ADMIT_TIME,
                    COALESCE(p.Number_Admissions{suffix}, 0) AS Number_Admissions{suffix},
                    COALESCE(p.Number_Admission_Types{suffix}, 0) AS Number_Admission_Types{suffix},
                    COALESCE(p.Inpatient_Days{suffix}, 0) AS Inpatient_Days_1Y
                FROM clean_admits a
                LEFT JOIN prev_admits p
                ON a.PATIENT_ID=p.PATIENT_ID AND a.ADMIT_TIME=p.ADMIT_TIME   
                ORDER BY PATIENT_ID, ADMIT_TIME
                '''
        return dfq(q, self.cursor).set_index(['PATIENT_ID', 'ADMIT_TIME'])

    # def getEncounterClass(self, q:str=None, )