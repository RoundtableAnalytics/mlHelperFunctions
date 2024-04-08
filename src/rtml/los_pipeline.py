from rtml.data_functions import dfq, patientIdToFloat
from mysql.connector.connection import MySQLConnection
from pandas import DataFrame
import numpy as np

class losPipeline:
    def __init__(self, cnx: MySQLConnection, db:str, patientId:str="PATIENT_ID", cvFolds:int=3, testPercent:float=0.2):
        self.cursor = cnx.cursor()
        self.cursor.execute(f'USE {db}')
        self.patientId = patientId
        self.cvFolds = cvFolds
        self.testPercent = testPercent
        self.cleanAdmitSql = '''clean_admits AS (
                    SELECT *, DATEDIFF(DISCHARGE_TIME, ADMIT_TIME) AS LOS
                    FROM encounters
                    WHERE PAT_CLASS = "Inpatient"
                    AND ADMIT_TIME IS NOT NULL
                    AND DISCHARGE_TIME IS NOT NULL
                )'''

    def getTrainingDatasets(self, trainIndex:list, verbose:bool=False):
        if len(trainIndex)==0:
            raise Exception("Must provide at least one training index")
        if verbose:
            print("Preparing DataFrame")
        df = self.getOutcome()
        df = df.join(self.getPastUtil())
        self.createSplits()
        self.trainEncounterClassEncoding(trainIndex=trainIndex)
        df = df.join(pipe.getEncounterClassFeature())
        if verbose:
            print("Splitting Dataframe into train & test sets")
        xss = [self.trainIds[y] for y in trainIndex]
        train = df.loc[[x for xs in xss for x in xs]]

        iss = set([x for x in range(self.cvFolds)]).difference(trainIndex)
        if len(iss)==0:
            if verbose:
                print("All folds selected for training, test data will be withheld test set")
            test = df.loc[pipe.testIds]
        else:
            if verbose:
                print(f"Tests will use hold out folds: {iss}")
            xss = [self.trainIds[y] for y in iss]
            test = df.loc[[x for xs in xss for x in xs]]
        return train, test


    def getFilteredAdmitSql(self, id_list:list):
        return self.cleanAdmitSql[:-1]+'\tAND PATIENT_ID IN ("' + '", "'.join(id_list) + '")\n\t\t)'

    def createSplits(self, q: str=None, inPlace:bool=True):
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

        testIds = df[df['hash']>=(1-self.testPercent)][self.patientId].to_list()
        trainIds = [df[(df['hash']<((self.testPercent/self.cvFolds)*(1+x)))&(df['hash']>=(self.testPercent/self.cvFolds)*x)][self.patientId].to_list() for x in range(self.cvFolds)]
        if inPlace:
            self.testIds = testIds
            self.trainIds = trainIds
            return None
        else:
            return (testIds, trainIds)

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
                    COALESCE(CAST(p.Inpatient_Days{suffix} AS SIGNED), 0) AS Inpatient_Days_1Y
                FROM clean_admits a
                LEFT JOIN prev_admits p
                ON a.PATIENT_ID=p.PATIENT_ID AND a.ADMIT_TIME=p.ADMIT_TIME   
                ORDER BY PATIENT_ID, ADMIT_TIME
                '''
        return dfq(q, self.cursor).set_index(['PATIENT_ID', 'ADMIT_TIME'])

    def trainEncounterClassEncoding(self, trainIndex:list, q:str=None, thresh:int=30):
        '''
        TODO
        '''
        if not q:
            xss = [self.trainIds[y] for y in trainIndex]
            idList = [x for xs in xss for x in xs]
            q = f'''
                WITH {self.getFilteredAdmitSql(idList)},
                fill_nulls AS (
                    SELECT COALESCE(HOSP_SERVICE, "") AS HOSP_SERVICE, LOS
                    FROM clean_admits
                ),
                grouped_service AS (
                    SELECT HOSP_SERVICE, COUNT(*) AS Count, AVG(LOS) AS mean_LOS
                    FROM fill_nulls
                    GROUP BY HOSP_SERVICE
                ),
                others AS (
                    SELECT 'Other' AS HOSP_SERVICE, LOS
                    FROM clean_admits l
                    INNER JOIN grouped_service r
                    ON l.HOSP_SERVICE = r.HOSP_SERVICE
                    WHERE r.Count <= {thresh}
                ),
                grouped_others AS (
                    SELECT HOSP_SERVICE, COUNT(*) AS Count, AVG(LOS) AS mean_LOS
                    FROM others
                    GROUP BY HOSP_SERVICE
                )
                SELECT HOSP_SERVICE, mean_LOS
                FROM grouped_others
                UNION
                SELECT CONVERT(HOSP_SERVICE USING utf8), mean_LOS
                FROM grouped_service
                WHERE Count > {thresh}
                
            '''
        df = dfq(q, self.cursor)
        tabs = "\t\t\t\t\t"
        switch = tabs + "CASE\n"
        for i, line in df.loc[1:].iterrows():
            if line.HOSP_SERVICE == "":
                switch = switch + tabs + f'\tWHEN (HOSP_SERVICE IS NULL) OR (HOSP_SERVICE = "") THEN {line.mean_LOS}\n'
            else:
                switch = switch + tabs + f'\tWHEN HOSP_SERVICE = "{line.HOSP_SERVICE}" THEN {line.mean_LOS}\n'
            # print(line.HOSP_SERVICE, line.mean_LOS)

        switch = switch + tabs + f"\tELSE {df.loc[0].mean_LOS}\n"
        switch = switch + tabs + "END AS mean_service_los"
        self.hosServiceSwitch = switch

    def getEncounterClassFeature(self, q:str=None,):
        '''
        TODO
        '''
        if not q:
            q = f'''
                WITH {self.cleanAdmitSql} 
                SELECT 
                    PATIENT_ID, 
                    ADMIT_TIME,
                    {self.hosServiceSwitch}
                FROM clean_admits 
                '''
        df = dfq(q, self.cursor).set_index(['PATIENT_ID', 'ADMIT_TIME'])
        df['mean_service_los'] = df['mean_service_los'].astype(float)
        return df

        
        