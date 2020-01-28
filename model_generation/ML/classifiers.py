from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC



clfs_all = {'SVM':  {'clf': SVC(class_weight="balanced"),
                 'parms':  {'C':                    [0.1,0.5, 1,5, 10,50, 100,500, 1000],  
                            'gamma':                [1,0.5, 0.1,0.05, 0.01, 0.005, 0.001,0.0005, 0.0001], 
                            'kernel':               ['linear','rbf']}  ,
                },
                
        'RF':   {'clf': RandomForestClassifier(n_estimators=500, class_weight="balanced"),
                 'parms': {"max_depth":            [15,13,11,9,5,3,2,None],
                           "max_features":         [20,25,30,35,40,50,60,70,80,90,100],
                           "min_samples_split":    [2,3,5,7,10,15,20,25,30,35,40],
                           "bootstrap":            [True, False],
                           "criterion":            ["gini", "entropy"]},
                },
        
        'AB':  {'clf': AdaBoostClassifier(),
                 'parms': {'n_estimators':[100,150,200,250,300],
                           'learning_rate':[1.0,1.5,2.0,2.5,3,3.5,4.0]},
                           'algorithm' : ['SAMME', 'SAMME.R']
                },
        
         
        }


clfs = {'SVM':  {'clf': SVC(class_weight="balanced"),
                 'parms':  {'C':                    [0.1,0.5, 1,5, 10,50, 100,500, 1000],  
                            'gamma':                [1,0.5, 0.1,0.05, 0.01, 0.005, 0.001,0.0005, 0.0001], 
                            'kernel':               ['linear','rbf']}  ,
                }
         
        }








