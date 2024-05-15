# test
import pandas as pd
import numpy as np

blocking_rule_prov = [
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.sex=r.sex and l.ssn=r.ssn",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.ssn=r.ssn",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.sex=r.sex",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ssn=r.ssn",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ssn=r.ssn",
    "l.dob = r.dob and l.sex=r.sex and l.ln=r.ln and l.ssn=r.ssn",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ln=r.ln and l.ssn=r.ssn",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ssn=r.ssn",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln",
    "l.dob = r.dob and l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex",
    "l.dob = r.dob and l.ssn=r.ssn and l.sex=r.sex",
    "l.dob = r.dob and l.ssn=r.ssn and l.ln=r.ln",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.ssn=r.ssn",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.ln=r.ln and l.sex=r.sex",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ssn=r.ssn",
    "l.PROVIDER_NUMBER=r.PROVIDER_NUMBER and l.sex=r.sex and l.ln=r.ln",
    "l.dob = r.dob and l.ssn=r.ssn",
    "l.dob = r.dob"
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