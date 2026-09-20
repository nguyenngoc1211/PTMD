"""Consistent comparisons on matched log-loss, correlations and figures."""
from common import CONFIG, setup, inference, adjust_bh
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def save(fig,name):
    fig.tight_layout()
    fig.savefig(f'results/figures/{name}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    setup(); sns.set_theme(style='whitegrid',font_scale=.85)
    cfi=pd.read_csv('results/tables/conditional_feature_importance.csv')
    mfi=pd.read_csv('results/tables/marginal_feature_importance.csv')
    m=np.load('results/intermediate/mfi_logloss_differences.npz')
    c=np.load('results/intermediate/cfi_differences.npz')
    if not np.array_equal(m['features'],c['features']): raise ValueError('Feature order mismatch')
    matched=[]
    for j,feature in enumerate(m['features']):
        valid=np.isfinite(c['differences'][:,j])
        stat=inference(m['differences'][valid,j],CONFIG['alpha'])
        matched.append(dict(feature=feature,marginal_logloss=stat['importance'],marginal_logloss_se=stat['se']))
    merged=cfi.merge(pd.DataFrame(matched),on='feature',validate='one_to_one').merge(mfi,on='feature',validate='one_to_one')
    merged['marginal_logloss_rank']=merged.marginal_logloss.rank(ascending=False,method='min').astype('Int64')
    merged['rank_change']=merged.conditional_rank-merged.marginal_logloss_rank
    merged['forensic_group']=merged.feature.str.split('.').str[0]
    merged['evidence']=np.where(merged.q_value.lt(CONFIG['alpha']) & merged.ci_lo.gt(0),
                                'Positive CFI after BH adjustment','Not established')
    merged.to_csv('results/tables/marginal_vs_conditional.csv',index=False)
    selected=list(dict.fromkeys(merged.nsmallest(10,'marginal_logloss_rank').feature.tolist()+
                                merged.nsmallest(10,'conditional_rank').feature.tolist()))
    saved=joblib.load('results/models/xgboost_model.joblib')
    X=pd.DataFrame(saved['X_test'],columns=saved['features'])
    # Correlations are descriptive on the complete held-out test set.
    corr=X[selected].corr(method='spearman')
    corr.to_csv('results/tables/correlation_top15.csv')
    all_corr=X.corr(method='spearman')
    pairs=[dict(feature_a=a,feature_b=b,spearman=float(all_corr.loc[a,b]))
           for i,a in enumerate(all_corr) for b in all_corr.columns[i+1:]
           if abs(all_corr.loc[a,b])>=.8]
    pd.DataFrame(pairs,columns=['feature_a','feature_b','spearman']).sort_values('spearman',key=abs,ascending=False).to_csv('results/tables/high_corr_pairs.csv',index=False)
    top=merged.nsmallest(10,'conditional_rank')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(top.feature,top.conditional_importance,xerr=top.conditional_importance_se,color='tomato')
    ax.invert_yaxis(); ax.set(xlabel='Log-loss increase (error bars: row-level SE)',title='Top-10 conditional importance')
    save(fig,'04_conditional_importance_top10')
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(merged.marginal_logloss_rank,merged.conditional_rank,alpha=.65)
    ax.plot([1,len(merged)],[1,len(merged)],'r--',lw=1)
    for _,row in merged.nsmallest(5,'conditional_rank').iterrows():
        ax.annotate(row.feature,(row.marginal_logloss_rank,row.conditional_rank),fontsize=7)
    ax.set(xlabel='Marginal log-loss rank',ylabel='Conditional log-loss rank',title='Matched-metric importance ranks')
    save(fig,'05_rank_mfi_vs_cfi')
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(merged.marginal_logloss,merged.conditional_importance,alpha=.65)
    ax.set_xscale('symlog',linthresh=1e-5); ax.set_yscale('symlog',linthresh=1e-5)
    ax.set(xlabel='Marginal log-loss increase',ylabel='Conditional log-loss increase',title='MFI vs CFI (matched rows per feature)')
    save(fig,'05_mfi_vs_cfi_scatter')
    top=merged.nsmallest(10,'marginal_logloss_rank')
    fig,ax=plt.subplots(figsize=(11,6))
    width=.36; pos=np.arange(len(top))
    ax.barh(pos-width/2,top.marginal_logloss,width,xerr=top.marginal_logloss_se,label='Marginal')
    ax.barh(pos+width/2,top.conditional_importance,width,xerr=top.conditional_importance_se,label='Conditional')
    ax.set_yticks(pos,top.feature); ax.invert_yaxis(); ax.legend()
    ax.set(xlabel='Log-loss increase (SE)',title='Matched-metric marginal and conditional importance')
    save(fig,'05_marginal_vs_conditional')
    fig,ax=plt.subplots(figsize=(11,6))
    for _,row in top.iterrows():
        ax.plot([0,1],[row.marginal_logloss_rank,row.conditional_rank],'o-',alpha=.65)
        ax.text(-.05,row.marginal_logloss_rank,row.feature,ha='right',va='center',fontsize=7)
    ax.set_xticks([0,1],['MFI log-loss','CFI log-loss']); ax.invert_yaxis()
    ax.set(xlim=(-1.4,1.2),ylabel='Rank (1 = highest)',title='Rank changes; ties can overlap')
    save(fig,'06_rank_change')
    fig,ax=plt.subplots(figsize=(12,10))
    sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool),k=1),vmin=-1,vmax=1,center=0,
                cmap='vlag',annot=True,fmt='.2f',annot_kws={'size':6},ax=ax)
    ax.set_title('Spearman correlations: union of top-10 MFI and CFI')
    save(fig,'07_top_feature_correlation_heatmap')
    cases=pd.read_csv('results/tables/conditioning_case_studies.csv')
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    labels=['All other features','Same forensic group','Other forensic groups']
    for ax,(feature,frame) in zip(axes.flat,cases.groupby('feature',sort=False)):
        frame=frame.set_index('conditioning_set').loc[labels]
        ax.bar(range(3),frame.importance,yerr=frame.se,color=['#4878a8','#d87754','#60a26e'])
        ax.set_xticks(range(3),['All','Same group','Other groups'],rotation=20)
        ax.set_title(feature,fontsize=9); ax.set_ylabel('Log-loss increase (SE)')
    axes.flat[-1].axis('off')
    save(fig,'08_conditioning_sets')
    summary=merged.groupby('forensic_group').agg(n_features=('feature','size'),
            max_cfi=('conditional_importance','max'),mean_cfi=('conditional_importance','mean'),
            max_mfi_logloss=('marginal_logloss','max'),
            n_significant_bh=('q_value',lambda p:int((p<CONFIG['alpha']).sum())))
    summary.sort_values('max_cfi',ascending=False).to_csv('results/tables/forensic_group_summary.csv')
    print('Comparison tables and 10 figures updated')


if __name__ == '__main__': main()
