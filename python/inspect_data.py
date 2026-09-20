import argparse,json,os,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--data',default='data/Obfuscated-MalMem2022.csv');a=p.parse_args(); d=pd.read_csv(a.data)
os.makedirs('results/intermediate',exist_ok=True)
info={'path':a.data,'samples':len(d),'columns':d.columns.tolist(),'shape':d.shape,'dtypes':d.dtypes.astype(str).to_dict(),'missing':d.isna().sum().to_dict(),'duplicates':int(d.duplicated().sum()),'class_counts':d.iloc[:,-1].value_counts().to_dict()}
print(json.dumps(info,indent=2));json.dump(info,open('results/intermediate/raw_data_inspection.json','w'),indent=2)
