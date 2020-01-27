import numpy as np

def avg_report(reports_lst, k_fold):
    
    f_0 = 0
    p_0  = 0
    r_0  = 0
    
    f_1 = 0
    p_1  = 0
    r_1  = 0
    
    acc = 0
    
    avg_p = 0
    avg_r  = 0
    avg_f  = 0
    
    wavg_p = 0
    wavg_r  = 0
    wavg_f  = 0
    
    
    avg_scores = reports_lst[0].copy()
    for report in reports_lst:
        

        f_0 =  f_0 +  report['0']['f1-score']/k_fold
        p_0  = p_0 + report['0']['precision']/k_fold
        r_0  = r_0 + report['0']['recall']/k_fold

        
        f_1 =  f_1 +  report['1']['f1-score']/k_fold
        p_1  = p_1 + report['1']['precision']/k_fold
        r_1  = r_1 + report['1']['recall']/k_fold
        
        acc = acc + report['accuracy']/k_fold
        
        avg_p =  avg_p +  report['macro avg']['precision']/k_fold
        avg_r  = avg_r +  report['macro avg']['recall']/k_fold
        avg_f  = avg_f +  report['macro avg']['f1-score']/k_fold
        
        wavg_p =  wavg_p +  report['weighted avg']['precision']/k_fold
        wavg_r  = wavg_r +  report['weighted avg']['recall']/k_fold
        wavg_f  = wavg_f +  report['weighted avg']['f1-score']/k_fold
        
        
    avg_scores['0']['f1-score'] = f_0
    avg_scores['0']['precision']= p_0
    avg_scores['0']['recall']= r_0
    
    avg_scores['1']['f1-score'] = f_1
    avg_scores['1']['precision']= p_1
    avg_scores['1']['recall']   = r_1 
    
    avg_scores['accuracy'] = acc   
    avg_scores['macro avg']['precision'] = avg_p
    avg_scores['macro avg']['recall'] = avg_r
    avg_scores['macro avg']['f1-score'] = avg_f
    
    avg_scores['weighted avg']['precision'] = wavg_p
    avg_scores['weighted avg']['recall'] = wavg_r
    avg_scores['weighted avg']['f1-score'] = wavg_f
    
    
    return avg_scores



def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
        