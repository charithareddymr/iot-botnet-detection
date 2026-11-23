# Bot-IoT data summary

## Shape

```
train: 2,934,817 rows x 19 columns
test:  733,705 rows x 19 columns
```

## Columns and types

```
pkSeqID                int64
proto                    str
saddr                    str
sport                    str
daddr                    str
dport                    str
seq                    int64
stddev               float64
N_IN_Conn_P_SrcIP      int64
min                  float64
state_number           int64
mean                 float64
N_IN_Conn_P_DstIP      int64
drate                float64
srate                float64
max                  float64
attack                 int64
category                 str
subcategory              str
```

## Missing values (train)

```
Series([], )
```

## attack label (train): 0 = normal, 1 = attack

```
           rows  percent
attack                  
1       2934447  99.9874
0           370   0.0126
```

## attack label (test): 0 = normal, 1 = attack

```
          rows  percent
attack                 
1       733598  99.9854
0          107   0.0146
```

## category (train)

```
                   rows  percent
category                        
DDoS            1541315  52.5183
DoS             1320148  44.9823
Reconnaissance    72919   2.4846
Normal              370   0.0126
Theft                65   0.0022
```

## subcategory (train)

```
                      rows  percent
subcategory                        
UDP                1584650  53.9948
TCP                1274843  43.4386
Service_Scan         58626   1.9976
OS_Fingerprint       14293   0.4870
HTTP                  1970   0.0671
Normal                 370   0.0126
Keylogging              59   0.0020
Data_Exfiltration        6   0.0002
```

## The accuracy trap

```
A 'model' that always answers ATTACK scores 99.9854% accuracy on the test set,
while catching 0 of 107 normal flows.
So accuracy is meaningless here: we must report per-class recall,
balanced accuracy, and false-positive rate instead.
```

## Single-feature leak test (balanced accuracy using ONE column alone; 0.5 = random)

```
pkSeqID                1.000   <-- suspicious
N_IN_Conn_P_DstIP      0.990   <-- suspicious
dport                  0.973   <-- suspicious
N_IN_Conn_P_SrcIP      0.965   <-- suspicious
daddr                  0.962   <-- suspicious
seq                    0.913
srate                  0.907
stddev                 0.860
max                    0.856
state_number           0.855
mean                   0.846
min                    0.784
saddr                  0.764
proto                  0.686
sport                  0.646
drate                  0.576
```
