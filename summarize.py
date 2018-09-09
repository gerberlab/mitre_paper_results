""" Tabulate results of MITRE calculations.

"""
import pandas as pd
import numpy as np
from StringIO import StringIO
import re

def process_kfold(path, comparison):
    report_path = path + '_mitre_cv_report.txt'
    point_lines = []
    ensemble_lines = []
    with open(report_path) as f:
        raw = f.read()

    point, ensemble = raw.split('Ensemble summary:')
    point = '\n'.join(point.split('\n')[1:-2])
    point = pd.read_csv(StringIO(re.sub(' +',',',point.strip())))
    ensemble = pd.read_csv(StringIO(re.sub(' +',',',ensemble.strip())))

    point = point.loc['median/sum','F1']
    ensemble = ensemble.loc['median/sum','F1']

    rf = None
    l1 = None

    if comparison:
        # We credit the RF/L1 methods with the better of their
        # scores as applied to data at two different points in
        # preprocessing
        step3_path = path + '_benchmark_step3_comparison.txt'
        step4_path = path + '_benchmark_step4_comparison.txt'
        rfs = []
        l1s = []
        for p in (step3_path, step4_path):
            with open(p) as f:
                raw = f.read()
            s = raw.split('Random forest:')[1]
            rf_table, l1_table = s.split('L1-regularized logistic regression:')
            l1_table = pd.read_csv(StringIO(re.sub(' +',',',l1_table.strip())))
            rf_table = pd.read_csv(StringIO(re.sub(' +',',',rf_table.strip())))
            rfs.append(rf_table.loc['median/sum','F1'])
            l1s.append(l1_table.loc['median/sum','F1'])
        rf = np.max(rfs)
        l1 = np.max(l1s)
    
    
    return ensemble, point, l1, rf


def process_leave_one_out(path, comparison):
    report_path = path + '_mitre_leave_one_out_report.txt'
    with open(report_path) as f:
        raw = f.read()
    report = pd.read_csv(StringIO(re.sub(' +',',',raw.strip())))
    point = report.loc['point','F1']
    ensemble = report.loc['ensemble','F1']

    rf = None
    l1 = None

    if comparison:
        # We credit the RF/L1 methods with the better of their
        # scores as applied to data at two different points in
        # preprocessing
        step3_path = path + '_benchmark_step3__leave_one_out_comparison.txt'
        step4_path = path + '_benchmark_step4__leave_one_out_comparison.txt'
        rfs = []
        l1s = []
        for p in (step3_path, step4_path):
            with open(p) as f:
                raw = f.read()
            raw = raw.replace('Random forest','rf')
            raw = raw.replace('L1-regularized logistic regression','l1')
            report = pd.read_csv(StringIO(re.sub(' +',',',raw.strip())))
            rfs.append(report.loc['rf','F1'])
            l1s.append(report.loc['l1','F1'])
        rf = np.max(rfs)
        l1 = np.max(l1s)
    
    return ensemble, point, l1, rf

def check_convergence(path):
    t = pd.read_csv(path + '_convergence.csv', index_col=0)
    worst_statistic = np.max(t.iloc[-1,:].values)
    print path
    print t.index[-1]
    return worst_statistic

def process_sensitivity(path):
    report_path = path + '_mitre_cv_report.txt'
    point_lines = []
    ensemble_lines = []
    with open(report_path) as f:
        raw = f.read()

    point, ensemble = raw.split('Ensemble summary:')
    point = '\n'.join(point.split('\n')[1:-2])
    point = pd.read_csv(StringIO(re.sub(' +',',',point.strip())))
    ensemble = pd.read_csv(StringIO(re.sub(' +',',',ensemble.strip())))

    folds = ['fold_%d' %i for i in xrange(5)]
    point = point.loc[folds,'F1']
    ensemble = ensemble.loc[folds,'F1']

    return (np.percentile(point,[25.,50.,75.]),
            np.percentile(ensemble,[25.,50.,75.]))

def get_sizes(path,comparison):
    detectors = None
    s3 = None
    s4 = None
    # Didn't request detector/feature numbers for sensitivity calculations
    if 'sensitivity' in path:
        return detectors, s3, s4
    pool_path = path + '.detector.pool.size'
    with open(pool_path) as f:
        detectors = int(f.read().strip())
    if comparison:
        s3_path = path + '_benchmark_step3_comparison.data_matrix_size'
        with open(s3_path) as f:
            l = f.read().strip()
            t = l.split(')')[0]
            i = t.split(', ')[1]
            s3 = int(i)
    if comparison:
        s4_path = path + '_benchmark_step4_comparison.data_matrix_size'
        with open(s4_path) as f:
            l = f.read().strip()
            t = l.split(')')[0]
            i = t.split(', ')[1]
            s4 = int(i)
    return s3, s4, detectors

if __name__ == '__main__':
    calculations = pd.read_csv('calculations.csv',index_col=0)
    results = pd.DataFrame(index=calculations.index,columns=['group','mitre_ensemble_f1','mitre_point_f1','rf_f1','l1_f1','comparator_features_step3','comparator_features_step4','detector_pool_size','worst_convergence_statistic','convergence_ok'])
    results['group'] = calculations['group']
    
    # First extract performance measurements
    for k in calculations.index:
        if not calculations.loc[k,'is_benchmark']:
            continue

        cv_type = calculations.loc[k,'cv_type']
        has_comparison = (calculations.loc[k,'has_comparison'] == True)
        path = calculations.loc[k,'paths']

        if cv_type == 'k_fold':
            scores = process_kfold(path, has_comparison)


        elif cv_type == 'leave_one_out':
            scores = process_leave_one_out(path, has_comparison)


        results.loc[k,['mitre_ensemble_f1','mitre_point_f1','l1_f1','rf_f1']] = scores
        results.loc[k,['comparator_features_step3','comparator_features_step4','detector_pool_size']] = get_sizes(path, has_comparison)

    # Now check convergence where appropriate
    for k in calculations.index:
        if not calculations.loc[k,'is_reference']:
            continue
        worst = check_convergence(calculations.loc[k,'paths'])
        results.loc[k,'worst_convergence_statistic'] = worst
        results.loc[k,'convergence_ok'] = worst < 1.1

    # Split these for easier interactive work
    reference = results.loc[calculations.is_reference,['group','worst_convergence_statistic','convergence_ok']]
    benchmark = results.loc[calculations.is_benchmark].drop(['worst_convergence_statistic','convergence_ok'],axis=1)
    reference.to_csv('reference_calculation_convergence_results.csv')
    benchmark.to_csv('performance_results.csv')

    sensitivity = calculations[calculations.group=='sensitivity'].copy()
    sensitivity.loc['knat_benchmark',:] = calculations.loc['knat_benchmark',:]
    sensitivity_results = pd.DataFrame(index=sensitivity.index,
                                       columns=['point_q1','point_median','point_q3','ensemble_q1','ensemble_median','ensemble_q3'])
    for k,path in zip(sensitivity.index, sensitivity.paths.values):
        p,e = process_sensitivity(path)
        sensitivity_results.loc[k,['point_q1','point_median','point_q3']] = p
        sensitivity_results.loc[k,['ensemble_q1','ensemble_median','ensemble_q3']] = e
    sensitivity_results.to_csv('sensitivity.csv')
         
