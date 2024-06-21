import pandas as pd
import numpy as np
from datetime import datetime
import gc
import pyarrow.parquet as pq
import time
#!pip uninstall -y splink
#!pip install splink
import splink
from splink.duckdb.duckdb_linker import DuckDBLinker
import splink.duckdb.duckdb_comparison_library as cl
import splink.duckdb.comparison_template_library as ctl
from IPython.display import display
from . import blocking_rule_prov


class healmatcher:
    def __init__(self, 
                 df_a, 
                 df_b, 
                 col_a, 
                 col_b,
                 blocking_rule = blocking_rule_prov,
                 blocking_rule_for_training = "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER",
                 match_prob_threshold=0.001, 
                 iteration=20,
                 data_name = ['data_left','data_right'],
                 pair_num = 1e6
                 ):
        self.df_a = df_a
        self.df_b = df_b
        self.col_a = col_a
        self.col_b = col_b
        self.match_prob_threshold = match_prob_threshold
        self.iteration = iteration
        self.blocking_rule_for_training = blocking_rule_for_training
        self.blocking_rule = blocking_rule
        self.data_name = data_name
        self.pair_num = pair_num
    
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
    
    def model_setup(self, 
                    df_a, df_b, 
                    col_a, col_b, 
                    variable, 
                    model_status, 
                    match_prob_threshold, 
                    iteration, 
                    blocking_rule_prov,
                    data_name):
        if model_status == 1:
            settings = {
                "link_type": "link_only",
                "blocking_rules_to_generate_predictions":  blocking_rule_prov,
                "comparisons": [            
                    #  Warning:   comparison template library (ctl) format can be changed in the future 
                    ctl.date_comparison(variable[variable.index('dob')],
                        levenshtein_thresholds=[0.9],
                        datediff_thresholds=[30,12, 1],
                        datediff_metrics=["day","month", "year"],
                        cast_strings_to_date=True
                    ),                    
                    #ctl.date_comparison(
                    #    "dob", #variable[variable.index('dob')], 
                    #    cast_strings_to_date=True
                    #    ),
                    ctl.name_comparison(
                        variable[variable.index('ln')],
                        levenshtein_thresholds=[1],
                        jaro_winkler_thresholds=[],
                        jaccard_thresholds=[1]
                    ),
                    cl.levenshtein_at_thresholds(variable[variable.index('ssn')], 
                                                 term_frequency_adjustments=True),
                    cl.levenshtein_at_thresholds(variable[variable.index('sex')], 
                                                 term_frequency_adjustments=True)
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
                    #  Warning:   comparison template library (ctl) format can be changed in the future 
                    ctl.date_comparison(variable[variable.index('dob')],
                        levenshtein_thresholds=[0.9],
                        datediff_thresholds=[30,12, 1],
                        datediff_metrics=["day","month", "year"],
                        cast_strings_to_date=True
                    ),                     
                    #ctl.date_comparison(
                    #        "dob", #variable[variable.index('dob')], 
                    #        cast_strings_to_date=True),
                    ctl.name_comparison(
                            variable[variable.index('ln')],
                            levenshtein_thresholds=[1],
                            jaro_winkler_thresholds=[],
                            jaccard_thresholds=[1]
                    ),
                    cl.levenshtein_at_thresholds(
                        variable[variable.index('sex')], 
                        term_frequency_adjustments=True
                    )
                ],
                "retain_matching_columns": True,
                "retain_intermediate_calculation_columns": True,
                "max_iterations": iteration,
                "em_convergence": match_prob_threshold  
            }            
        linker = DuckDBLinker([df_a.astype({"dob":str}),
                               df_b.astype({"dob":str})
                               ],
                              settings, 
                              input_table_aliases= data_name )
        return linker
    
    def model_training(self, 
                       linker, 
                       blocking_rule_prov, 
                       blocking_rule_for_training, 
                       match_prob_threshold,
                       pair_num = 1e6
                       ):
        linker.estimate_probability_two_random_records_match( blocking_rule_prov, recall=0.6 )
        gc.collect()
        linker.estimate_u_using_random_sampling(max_pairs=pair_num)    
        gc.collect()
        linker.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)
        gc.collect()
        df_predictions = linker.predict(threshold_match_probability= match_prob_threshold)
        return [df_predictions,linker]

    def model_visual(self, 
                     df_predictions, 
                     linker, 
                     visual_matchweight=True, 
                     visual_waterfall=True):
        if visual_matchweight==True:
            display(linker.match_weights_chart())
        if visual_waterfall == True:
            records_to_view = df_predictions.as_record_dict(limit=5)
            display(linker.waterfall_chart(records_to_view, filter_nulls=False) ) 
    
    def find_unmatch(self, df_a, df1):
        unmatched = df_a[~df_a['trackid'].isin(df1['unique_id_r'].unique())]
        return unmatched
    
    
def hm(df_a, 
       df_b, 
       col_a=None, 
       col_b=None,
       match_prob_threshold=0.001, 
       iteration=20,
       blocking_rule_prov = blocking_rule_prov,
       iteration_input = 20,
       model2 = False,
       blocking_rule_for_training_input=None,
       onetoone=True,
       visual_matchweight=False,
       visual_waterfall=False,
      match_summary=False,
      data_name = ['dfa','dfb'],
      use_save_model = None,
      save_model_path = None,
      export_model = None,
      export_model_path = None,
      pair_num_input = 1e6
      ):
    if df_a.empty :
        raise ValueError("Left dataframe is empty")
    elif df_b.empty:
        raise ValueError("Right dataframe is empty")
    if not blocking_rule_for_training_input in df_a.columns:
        raise ValueError("Missing blocking columns in Left dataframe!")
    elif not blocking_rule_for_training_input in df_b.columns:
        raise ValueError("Missing blocking columns in Right dataframe!")
    else:
        blocking_rule_for_training = f'l.{blocking_rule_for_training_input} = r.{blocking_rule_for_training_input}'
    
    test1 = healmatcher(
        df_a=df_a, 
        df_b=df_b, 
        col_a=col_a, 
        col_b=col_b, 
        match_prob_threshold=match_prob_threshold, 
        iteration=iteration_input ,
        blocking_rule = blocking_rule_prov,
        blocking_rule_for_training = blocking_rule_for_training,
        data_name = data_name,
        pair_num = pair_num_input
    )
    df_a = test1.trackid_gen(df_a)
    df_b = test1.trackid_gen(df_b)
    data_cds1 = test1.data_check(df_a, col_a)
    data_medicaid1 = test1.data_check(df_b, col_b)
    
    print("Model 1 begins..")
    linker = test1.model_setup(df_a=data_cds1, 
                               df_b=data_medicaid1,
                               col_a=col_a,
                               col_b=col_b, 
                               variable = ['dob','ln','ssn','sex'],
                              model_status=1,
                              match_prob_threshold=match_prob_threshold, 
                              iteration=iteration, 
                              blocking_rule_prov = blocking_rule_prov,
                              data_name = data_name
    )
    
    if use_save_model is True:
        try:
            linker.load_model(save_model_path)
            model1=test1.model_training(linker = linker, 
                                        blocking_rule_prov = blocking_rule_prov,
                                        blocking_rule_for_training = blocking_rule_for_training, 
                                        match_prob_threshold = match_prob_threshold,
                                        pair_num = pair_num_input
            )    
        except Exception as e:
            print("Loading saved model error->", e)
            print("Model running begins..")
            model1=test1.model_training(linker = linker, 
                                    blocking_rule_prov = blocking_rule_prov,
                                    blocking_rule_for_training = blocking_rule_for_training, 
                                    match_prob_threshold = match_prob_threshold,
                                    pair_num = pair_num_input
            )
            pass   
    else:
        model1=test1.model_training(linker = linker, 
                                    blocking_rule_prov = blocking_rule_prov,
                                    blocking_rule_for_training = blocking_rule_for_training, 
                                    match_prob_threshold = match_prob_threshold,
                                    pair_num = pair_num_input
        )
    
    if export_model is True:
        try:
            setting = model1[1].save_model_to_json(export_model_path, overwrite=True)
        except Exception as e:
            print("Saving model error->", e)
            pass
            
    try:
        test1.model_visual(df_predictions=model1[0], 
                           linker=model1[1], 
                           visual_matchweight=visual_matchweight, 
                           visual_waterfall=visual_waterfall)
    except Exception as e:
        print("Visualization error->", e)
        pass
    
    print("Model 2 begins..")
    if model2 == True:
        un_a=test1.find_unmatch(data_cds1, model1[0].as_pandas_dataframe())
        un_b=test1.find_unmatch(data_medicaid1, model1[0].as_pandas_dataframe())
        if un_a.shape[0]>0 and un_b.shape[0]>0:
            linker2 = test1.model_setup(df_a=un_a,
                                        df_b=un_b, 
                                        col_a=col_a,
                                        col_b=col_b, 
                                   variable = ['dob','ln','ssn','sex'],
                                    model_status=2,
                                  match_prob_threshold=match_prob_threshold, 
                                  iteration=iteration, 
                                  blocking_rule_prov = blocking_rule_prov,
                                  data_name = data_name
            )
            model2=test1.model_training(linker = linker2, 
                                        blocking_rule_prov = blocking_rule_prov,
                                        blocking_rule_for_training = blocking_rule_for_training, 
                                        match_prob_threshold = match_prob_threshold,
                                        pair_num = pair_num_input
            )
            #model2 = model_out[0].as_pandas_dataframe()
            model1 = pd.concat([model1[0].as_pandas_dataframe(), 
                                model2[0].as_pandas_dataframe()
                               ]).drop_duplicates()
        else:
            print("No unmatched data. Skipping model 2.")
            model1 = model1[0].as_pandas_dataframe()
            pass
    elif model2 == False:
        model1 = model1[0].as_pandas_dataframe()
    # one to one
    if onetoone == True:
        print("One to one cleaning begins..")
        df1 = model1[model1.groupby('unique_id_l')['match_probability'].rank(method='first', ascending=False) <= 1]
        df1 = df1[df1.groupby('unique_id_r')['match_probability'].rank(method='first', ascending=False) <= 1]
    gc.collect()
    if match_summary == True:
        print("-"*10,"Summary","-"*10)
        print("Matched trackid counts: ", df1['unique_id_r'].nunique())
        print("Matched rate (cds): ", round(df1['unique_id_r'].nunique()/data_cds1['trackid'].nunique(),2 )*100,"%" )
        print("Matched rate (Medicaid): ", round(df1['unique_id_r'].nunique()/data_medicaid1['trackid'].nunique(),2 )*100,'%' )
    print("Matching complete")
    return df1

def medicaid_clean_second(dat):
    return dat.assign(
        indate = dat.groupby('TRACKID')['srv_dt'].transform('min'),
        outdate = dat.groupby('TRACKID')['srv_end_dt'].transform('max')
    )

def hm2(
        df_l,
        df_r,
        matched,
        prob_threshold = [0.8, 0.5],
        apply_srvdate_rule = False
    ):
    join_r = pd.merge(
        pd.merge(
            df_l.drop_duplicates(),
            matched[['match_probability','unique_id_l','unique_id_r']],
            left_on = ['trackid'],
            right_on = ['unique_id_l'],
            how = 'inner'
        ).drop_duplicates().drop(columns=['indate','outdate']).drop_duplicates(),
        df_r,
        left_on=['PROVIDER_NUMBER','unique_id_r'],
        right_on=['PROVIDER_NUMBER','TRACKID'],
        how = 'left'
    )
    step3 = medicaid_clean_second(join_r)
    step3 = step3.assign(
        group=np.where(
            step3['match_probability']>=prob_threshold[0],"high",
            np.where(
                (
                    (
                        step3['match_probability']>=prob_threshold[1]
                    ) &
                    (
                        step3['match_probability']<prob_threshold[0]
                    )
                ),
                1, 0
            )
        ),
        check_srvdate = np.where(
            (
                step3['srv_dt']>=step3['indate']
            ) &
            (
                step3['srv_dt']<=step3['outdate']
            ),
            1, 0
        )
    )
    if apply_srvdate_rule==True:
        step3=step3[
            (
                step3['check_srvdate']==1
            ) &
            (
                step3['group'].isin(['high','medium'])
            )
        ]
    return step3

def pretrained_model(use_saved_model="No",
                     bring_your_model="No",
                     save_model_output="No",
                     type_of_data = "OTP",
                     year_of_data = 19,
                     bring_your_model_path = None
                     ):
    using_model, using_model_path,export_model,export_model_path = False,False,False,False
    if use_saved_model=="Yes":
        print("Using original pre-trained model")
        using_model=True
        if type_of_data=="OTP":
            using_model_path='./otp/otp_'+year_of_data+"_model.json"
        elif type_of_data=="OP":
            using_model_path='./op/op_'+year_of_data+"_model.json"
    if bring_your_model=="Yes":
        bring_your_model=True
        print("Using your own model")
        using_model_path=bring_your_model_path.copy()
    if save_model_output=="Yes":
        print("Export trained model")
        if type_of_data=="OTP":
            export_model_path='./otp/otp_'+year_of_data+"_model.json"
        elif type_of_data=="OP":
            using_model_path='./op/op_'+year_of_data+"_model.json"
    return[using_model, using_model_path, export_model, export_model_path]
        
    
def hm_viz(data, 
           data_raw, 
           group='PROVIDER_NUMBER',
           count='trackid', 
           count2='new_tcn', 
           oasas_crosswalk = False,
           oasas_crosswalk_file = None,
           year_of_data = None,
           show_table = True, 
           show_table_save=False,
           show_visual=True,
           show_visual_save=False
    ):
    if not group in data_raw.columns:
        raise ValueError("Missing group column in dataframe!")
    df= pd.merge(
            pd.merge(
                pd.DataFrame(
                    data_raw[data_raw[count].isin(data[count])].groupby(group)[count].nunique()
                ).sort_values(by=count,ascending=False).reset_index().rename(columns={count:count+"_match"}),
                pd.DataFrame(
                    data_raw[~data_raw[count].isin(data[count])].groupby(group)[count].nunique()
                ).sort_values(by=count,ascending=False).reset_index().rename(columns={count:count+"_unmatch"}),
                on = group, how='left'
            ),
            pd.merge(
                pd.DataFrame(
                    data_raw[data_raw[count2].isin(data[count2])].groupby(group)[count2].nunique()
                ).sort_values(by=count2,ascending=False).reset_index().rename(columns={count2:count2+"_match"}),
                pd.DataFrame(
                    data_raw[~data_raw[count2].isin(data[count2])].groupby(group)[count2].nunique()
                ).sort_values(by=count2,ascending=False).reset_index().rename(columns={count2:count2+"_unmatch"}),
                on = group, how='left'
            ),
            on=[group], how = 'left'
        )
    df[count+"_rate"]= round(df[count+"_match"]/(df[count+"_match"]+df[count+"_unmatch"])*100,1)
    df[count2+"_rate"]= round(df[count2+"_match"]/(df[count2+"_match"]+df[count2+"_unmatch"])*100,1)
    
    if oasas_crosswalk == True:
        df = pd.merge(
            df, oasas_crosswalk_file[['PROVIDER_NUMBER','OASAS_PROVIDER_NAME']].drop_duplicates(),
            on = ['PROVIDER_NUMBER'], how = 'left'
        ).drop_duplicates()
    if show_table==True:
        from IPython.display import display
        display(df)
        if show_table_save ==True:
            show_table.to_excel("match_ratio_table.xlsx")
    if show_visual == True:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[group],
            y=df[count+"_rate"],
            customdata=df[df.columns] ,
            hovertemplate = "%{y} % <br><b>Match count:</b> %{customdata[1]}" \
                             "<br><b>Unmatche count:</b> %{customdata[2]} "\
                             "<br><b>Group:</b> %{customdata[0]}" \
                             "<br><b>Provider Name:</b> %{customdata[7]}" ,
            mode='lines',
            name='<b>'+count+'</b>')
        )
        fig.add_trace(go.Scatter(x=df[group],
                             y=df[count2+"_rate"],
                             customdata=df ,
                             hovertemplate = "%{y} % <br><b>Match count:</b> %{customdata[3]}" \
                                             "<br><b>Unmatche count:</b> %{customdata[4]} "\
                                             "<br><b>Group:</b> %{customdata[0]}",
                            mode='lines+markers',
                            name='<b>'+count2+'</b>')
        )
        fig.update_layout(title='Match ratio by '+ group+' (20'+ year_of_data+")",
                           xaxis_title=group,
                           yaxis_title='Ratio (match)',
                         plot_bgcolor='white')
        fig.update_layout(hovermode='x unified')
        fig.show()
        if show_visual_save==True:
            fig.write_html("match_ratio_visual.html")
            
if __name__ == "__main__":
    print("healmatcher loaded!")