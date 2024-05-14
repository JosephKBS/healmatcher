# healmatcher
- `healmatcher` is a simple but fast probabilistic matching package developed by NYULH HEAL Lab.
- Currently, the model supports 4 variables (sex, date of birth, last 4 digits of ssn, and first 2 letters of last name) to run the linkage process


## How to use (example)

`pip install healmatcher`


```python
# Install package
!pip install healmatcher

# Load package
from hm import hm

# create example dataset
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
    'PROVID

# Run matching
hm(
    df_a = testa,
    df_b = testb,
    col_a=['sex','dob','ssn','ln'],
    col_b=['sex','dob','ssn','ln'],
    match_prob_threshold = 0.001,
    iteration = 20,
    model2 = True,
    blocking_rule_for_input = 'PROVIDER_NUMBER',
    onetoone = True,
    match_summary = True
)
```

# Webpage
[healmatcher](https://pypi.org/project/healmatcher/)
