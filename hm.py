# test
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import sys
import sas7bdat
from sas7bdat import SAS7BDAT
import time
import pathlib
import os.path
import pyarrow as pa
import pyarrow.parquet as pq
#!pip install pgeocode
import pgeocode
import step1
import time
#!pip uninstall -y splink
#!pip install splink
import splink
from splink.duckdb.duckdb_linker import DuckDBLinker
import splink.duckdb.duckdb_comparison_library as cl
import splink.duckdb.comparison_template_library as ctl

blocking_rule_prov = [
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.sex=r.sex and l.ssn=r.ssn",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.ssn=r.ssn",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.sex=r.sex",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ssn=r.ssn",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ssn=r.ssn",
    "l.DOB = r.DOB and l.sex=r.sex and l.ln=r.ln and l.ssn=r.ssn",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ln=r.ln and l.ssn=r.ssn",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ssn=r.ssn",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln",
    "l.DOB = r.DOB and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex",
    "l.DOB = r.DOB and l.ssn=r.ssn and l.sex=r.sex",
    "l.DOB = r.DOB and l.ssn=r.ssn and l.ln=r.ln",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.ssn=r.ssn",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.sex=r.sex",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ssn=r.ssn",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ln=r.ln",
    "l.DOB = r.DOB and l.ssn=r.ssn",
    "l.DOB = r.DOB"
]


testa = pd.DataFrame({
    'sex':[1,2,1,2,1,2,1,2,1,2],
    'dob':['2012-1-1','2011-12-1','1999-1-1','1998-11-1','2012-11-1','1984-1-1','1982-1-1','1975-1-1','1967-1-1','1954-1-1'],
    'ssn':[1111,2222,3333,4444,5555,6666,7777,8888,9999,1010],
    'ln':["as",'ss','zz','rr','ww','wa','tr','tt','hh','gq'],
    'PROVIDER_NUMBER':[2,1,1,1,1,1,1,1,2,1]
})
testb = pd.DataFrame({
    'sex':[2,2,1,1,1,2,1,2,1,1],
    'dob':['2012-1-1','2001-12-1','1999-1-1','1998-11-1','2012-11-1','1984-1-1','1982-1-1','1975-1-1','1967-1-1','1954-1-1'],
    'ssn':[1111,2222,3333,4444,5555,6666,7777,8888,9999,1010],
    'ln':["as",'ls','zz','rr','wb','wa','tr','tt','ha','gq'],
    'PROVIDER_NUMBER':[2,1,1,1,1,1,1,1,2,1]
})


   
def hm(df_a=testa, df_b=testb, col_a=None, col_b=None,
       match_prob_threshold=0.001, 
       iteration=20,
        blocking_rule_prov = blocking_rule_prov,
       iteration_input = 20,
       model2 = False,
       blocking_rule_for_training_input = "PROVIDER_NUMBER",
       onetoone=True,
       visual_matchweight=False,
       visual_waterfall=False,
      match_summary=False,
      ):
    blocking_rule_for_training = f'l.{blocking_rule_for_training_input} = r.{blocking_rule_for_training_input}'
    test1 = healmatcher(
        df_a=testa, df_b=testb, 
        col_a=col_a, col_b=col_b, 
        match_prob_threshold=match_prob_threshold, 
        iteration=iteration_input 
    )
    df_a = test1.trackid_gen(df_a)
    df_b = test1.trackid_gen(df_b)
    data_cds1 = test1.data_check(df_a, col_a )
    data_medicaid1 = test1.data_check(df_b, col_b)
    
    print("Model 1 begins..")
    linker = test1.model_setup(df_a=data_cds1, df_b=data_medicaid1,
                               col_a=col_a,col_b=col_b, 
                               variable = ['dob','ln','ssn','sex'],
                              model_status=1,
                              match_prob_threshold=match_prob_threshold, 
                              iteration=iteration, 
                              blocking_rule_prov = blocking_rule_prov
    )
    model1=test1.model_training(linker = linker, blocking_rule_prov = blocking_rule_prov,
        blocking_rule_for_training = blocking_rule_for_training, match_prob_threshold = match_prob_threshold
    )
    try:
        test1.model_visual( df_predictions=model1[0], linker=model1[1], visual_matchweight=True, visual_waterfall=True)
    except:
        print("Visualization error")
        pass
    
    print("Model 2 begins..")
    if model2 == True:
        un_a=test1.find_unmatch(data_cds1, model1[0].as_pandas_dataframe())
        un_b=test1.find_unmatch(data_medicaid1, model1[0].as_pandas_dataframe())
        if un_a.shape[0]>0 and un_b.shape[0]>0:
            linker2 = test1.model_setup( df_a=un_a,df_b=un_b, col_a=col_a,col_b=col_b, 
                                   variable = ['dob','ln','ssn','sex'],
                                    model_status=2,
                                  match_prob_threshold=match_prob_threshold, 
                                  iteration=iteration, 
                                  blocking_rule_prov = blocking_rule_prov
            )
            model2=test1.model_training(linker = linker2, blocking_rule_prov = blocking_rule_prov,
                blocking_rule_for_training = blocking_rule_for_training, match_prob_threshold = match_prob_threshold
            )
            #model2 = model_out[0].as_pandas_dataframe()
            model1 = pd.concat([model1[0].as_pandas_dataframe(), 
                                model2[0].as_pandas_dataframe()
                               ]).drop_duplicates()
        else:
            print("No unmatched data")
            model1 = model1[0].as_pandas_dataframe()
            pass
    # one to one
    if onetoone == True:
        print("One to one cleaning begins..")
        df1 = model1[model1.groupby('unique_id_l')['match_probability'].rank(method='first', ascending=False) <= 1]
        df1 = df1[df1.groupby('unique_id_r')['match_probability'].rank(method='first', ascending=False) <= 1]
    gc.collect()
    if match_summary == True:
        print("-"*10,"Summary","-"*10)
        print("Matched trackid counts: ",df1['unique_id_r'].nunique())
        print("Matched rate (cds): ", round(df1['unique_id_r'].nunique()/data_cds1['trackid'].nunique(),2 )*100,"%" )
        print("Matched rate (Medicaid): ", round(df1['unique_id_r'].nunique()/data_medicaid1['trackid'].nunique(),2 )*100,'%' )
    print("Matching complete")
    return df1
'''
def processing(data, cols):
    data['DOB_y'] = pd.to_datetime(data['dob']).dt.year.astype(str)
    data['DOB_m'] = pd.to_datetime(data['dob']).dt.month.astype(str)
    data['DOB_d'] = pd.to_datetime(data['dob']).dt.day.astype(str)
    data['DOB_m_count'] = data['DOB_m'].apply(lambda x: len(x))
    data['DOB_d_count'] = data['DOB_d'].apply(lambda x: len(x))
    data['DOB_m'] = np.where(data['DOB_m_count']==1, str(0)+data['DOB_m'], data['DOB_m'] )
    data['DOB_d'] = np.where(data['DOB_d_count']==1, str(0)+data['DOB_d'], data['DOB_d'] )
    data['unique_id'] = data['trackid']#.copy()
    #data['unique_id'] = data[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
    return data.drop(columns=['DOB_y','DOB_m','DOB_d','DOB_m_count','DOB_d_count'])
d
'''

class healmatcher:
    def __init__(self, df_a, df_b, col_a, col_b,
                 blocking_rule = blocking_rule_prov,
                 blocking_rule_for_training = "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER",
                 match_prob_threshold=0.001, iteration=20):
        self.df_a = df_a
        self.df_b = df_b
        self.col_a = col_a
        self.col_b = col_b
        self.match_prob_threshold = match_prob_threshold
        self.iteration = iteration
        self.blocking_rule_for_training = blocking_rule_for_training
        self.blocking_rule = blocking_rule
    
    def trackid_gen(self, check3):
        check3 = check3.astype({'sex':str,'ssn':str })
        check3['DOB_y'] = pd.to_datetime(check3['dob']).dt.year.astype(str)
        check3['DOB_m'] = pd.to_datetime(check3['dob']).dt.month.astype(str)
        check3['DOB_d'] = pd.to_datetime(check3['dob']).dt.day.astype(str)
        check3['DOB_m_count'] = check3['DOB_m'].apply(lambda x: len(x))
        check3['DOB_d_count'] = check3['DOB_d'].apply(lambda x: len(x))
        check3['DOB_m'] = np.where(check3['DOB_m_count']==1, str(0)+check3['DOB_m'], check3['DOB_m'] )
        check3['DOB_d'] = np.where(check3['DOB_d_count']==1, str(0)+check3['DOB_d'], check3['DOB_d'] )
        check3["trackid"] = check3["sex"].str.cat(check3[["DOB_y", "DOB_m",'DOB_d','ssn','ln']].astype(str), sep="")
        check3['unique_id'] = check3['trackid']
        check3 = check3.drop(columns=['DOB_y','DOB_m','DOB_d','DOB_m_count','DOB_d_count'])
        return check3    
    
    def data_check(self, df_a, col_a):
        if len(sorted(set(col_a).intersection(df_a.columns))) != len(col_a):
            raise ValueError("Missing columns in data")
        df_a['sex'] = df_a['sex'].astype(str)  
        if df_a['dob'].dtype == 'object':
            df_a['dob'] = pd.to_datetime(df_a['dob'])
        return df_a
    
    def model_setup(self, df_a, df_b, col_a, col_b, variable, model_status, match_prob_threshold, iteration, blocking_rule_prov):
        if model_status == 1:
            settings = {
                "link_type": "link_only",
                "blocking_rules_to_generate_predictions":  blocking_rule_prov,
                "comparisons": [            
                    ctl.date_comparison( variable[0],
                        levenshtein_thresholds=[0.9],
                        datediff_thresholds=[30, 12, 1],
                        datediff_metrics=["day", "month", "year"],
                    ),
                    ctl.name_comparison(variable[1],
                            levenshtein_thresholds=[1],
                            jaro_winkler_thresholds=[],
                            jaccard_thresholds=[1]
                    ),
                    cl.levenshtein_at_thresholds(variable[2], term_frequency_adjustments=True),
                    cl.levenshtein_at_thresholds(variable[3], term_frequency_adjustments=True)
                ],
                "retain_matching_columns": True,
                "retain_intermediate_calculation_columns": True,
                "max_iterations": iteration,
                "em_convergence": match_prob_threshold  
            }
        elif model_status ==2:
            settings = {
                "link_type": "link_only",
                "blocking_rules_to_generate_predictions":  blocking_rule_prov,
                "comparisons": [            
                    ctl.date_comparison(variable[0],
                        levenshtein_thresholds=[0.9],
                        datediff_thresholds=[30, 12, 1],
                        datediff_metrics=["day", "month", "year"],
                    ),
                    ctl.name_comparison(variable[1],
                            levenshtein_thresholds=[1],
                            jaro_winkler_thresholds=[],
                            jaccard_thresholds=[1]
                    ),
                    cl.levenshtein_at_thresholds(variable[3], term_frequency_adjustments=True)
                ],
                "retain_matching_columns": True,
                "retain_intermediate_calculation_columns": True,
                "max_iterations": iteration,
                "em_convergence": match_prob_threshold  
            }            
        linker = DuckDBLinker([df_a,df_b],
                              settings, 
                              input_table_aliases=["CDS", "MEDICAID"] )
        return linker
    
    def model_training(self, linker, blocking_rule_prov, blocking_rule_for_training, match_prob_threshold):
        linker.estimate_probability_two_random_records_match( blocking_rule_prov, recall=0.6 )
        linker.estimate_u_using_random_sampling(max_pairs=1e7)    
        gc.collect()
        linker.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)
        gc.collect()
        df_predictions = linker.predict(threshold_match_probability= match_prob_threshold)
        return [df_predictions,linker]#.as_pandas_dataframe()

    def model_visual(self, df_predictions, linker, visual_matchweight=True, visual_waterfall=True):
        if visual_matchweight==True:
            display(linker.match_weights_chart())
        if visual_waterfall == True:
            records_to_view = df_predictions.as_record_dict(limit=5)
            display(linker.waterfall_chart(records_to_view, filter_nulls=False) ) 
    
    def find_unmatch(self, df_a, df1):
        unmatched = df_a[~df_a['trackid'].isin(df1['unique_id_r'].unique())]
        return unmatched
    
if __name__ == "__main__":
    print("healmatcher loaded!")