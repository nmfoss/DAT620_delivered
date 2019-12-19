import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from collections import Counter

#from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


'''
# transformer for non-text based features
class NumericalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols, scaler, params):
        self.cols = cols
        self.scaler = scaler
        self.params = params
        self.iScaler = None

    def fit(self, df, y=None):
        if self.params['scale']:
            temp = self.params['scale']
            del self.params['scale']
            self.iScaler = self.scaler(**self.params)
            self.params['scale'] = temp
            self.iScaler.fit(df[self.cols].values)
        return self

    def transform(self, df):
        if len(self.cols) > 0:
            if self.iScaler is not None:
                return self.iScaler.transform(df[self.cols].values)
            else:
                return df[self.cols].values
        else:
            return self
        
            
        
        if len(self.cols) > 0:
            return df[self.cols].notnull()
        else:
            return df.notnull()

#transformer to text based features
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col, vec, params):
        self.col = col
        self.vec = vec
        self.params = params
        self.iVec = None

    def transform(self, df, y=None):
        
        return self.iVec.transform(df[self.col].fillna("").values)
        

    def fit(self, df, y=None):        
        self.iVec = self.vec(**self.params)
        self.iVec.fit(df[self.col].fillna("").values)
        return self
'''

class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, binary_clf, binary_clf_params, bucket_clf, bucket_clf_params, judge):
        self.binary_clf = binary_clf
        self.binary_clf_params = binary_clf_params
        self.bucket_clf = bucket_clf
        self.bucket_clf_params = bucket_clf_params
        self.judge = judge
        
    def get_buckets(self, counted):
        counted_sum = sum(counted.values())
        buckets = []
        index = 0
        for c, v in counted.most_common():
            if index == 0:
                sum_left = counted_sum
                sum_bucket = 0
                new_bucket = []

            new_bucket.append(c)
            sum_bucket += v
            sum_left -= v

            if sum_bucket >= sum_left:
                buckets.append(new_bucket)
                new_bucket = []
                sum_bucket = 0
            index += 1
            if index == len(counted) and len(new_bucket) > 0:
                buckets.append(new_bucket)
        return buckets
    
    def get_vs_all(self, buckets):
        bucket_vs_all = []
        for l_index in range(len(buckets)):
            vs_all = []
            if l_index+1 < len(buckets):
                for r_index in range(l_index+1, len(buckets)):
                    vs_all += buckets[r_index]
                bucket_vs_all.append((buckets[l_index], vs_all))
        return bucket_vs_all
 
    def fit(self, X, y):

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.y_list_ = y.tolist()
        self.y_counts_ = Counter(self.y_list_)
        
        self.buckets_ = self.get_buckets(self.y_counts_)
        self.vs_all_ = self.get_vs_all(self.buckets_)

        self.len_b_ = len(self.vs_all_)

        self.vs_all_indicies_ = []
        self.buckets_indicies_ = []
        for bucket, rest in self.vs_all_:
            bucket_indices = []
            rest_indices = []
            #for index, value in self.y_.iteritems():
            
            for index in range(len(self.y_list_)):
                if self.y_list_[index] in bucket:
                    bucket_indices.append(index)
                if self.y_list_[index] in rest:
                    rest_indices.append(index)
                    
            self.vs_all_indicies_.append(rest_indices)
            self.buckets_indicies_.append(bucket_indices)

        self.binary_clfs_ = [self.binary_clf(**self.binary_clf_params) for _ in range(self.len_b_)]
        
        for index, clf in enumerate(self.binary_clfs_):

            fit_y = []
            fit_x_index = []
            for i in range(len(self.y_list_)):
                if i in self.buckets_indicies_[index]:
                    fit_y.append(0)
                    fit_x_index.append(i)
                if i in self.vs_all_indicies_[index]:
                    fit_y.append(1)
                    fit_x_index.append(i)
                    
            fit_x = self.X_[fit_x_index]
            
            clf.fit(fit_x, fit_y)

        self.bucket_clfs_ = [self.bucket_clf(**self.bucket_clf_params) for _ in range(self.len_b_)]
        
        for index, clf in enumerate(self.bucket_clfs_):
            
            bucket_y = [y for i, y in enumerate(self.y_list_) if i in self.buckets_indicies_[index]]
            bucket_x = self.X_[self.buckets_indicies_[index]]
            
            clf.fit(bucket_x, bucket_y)
            
        '''
        judge_train = []

        for index in range(self.len_b_):
            
            judge_train.append(self.binary_clfs_[index].predict(self.X_))
            judge_train.append(self.bucket_clfs_[index].predict(self.X_))
             
        self.judge_train_t_ = np.array(judge_train).T

        self.judge.fit(self.judge_train_t_, self.y_list_)
        '''
        return self

    def predict(self, X):
        
        '''
        votes = []
        
        for index in range(self.len_b_):
            votes.append(self.binary_clfs_[index].predict(X))
            votes.append(self.bucket_clfs_[index].predict(X))

        votes_t = np.array(votes).T 
        
        print(list(votes_t))
        
        #return votes_t
        preds = self.judge.predict(votes_t)
        
        '''
        
        len_x = X.shape[0]
        
        preds = np.ones(len_x) * 12
        
        predicted = []
        for index in range(self.len_b_):
            
            split = self.binary_clfs_[index].predict(X)

            in_bucket = [i for i in range(len_x) if split[i] == 0 and i not in predicted]
            preds[in_bucket] = self.bucket_clfs_[index].predict(X[in_bucket])
            
            predicted += in_bucket


        return preds




class BucketClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hashfunc, classifiers, judge, num_buckets):
        self.num_classifiers = len(classifiers)
        self.hashfunc = hashfunc
        self.classifiers = classifiers
        self.judge = judge
        self.num_buckets = num_buckets
 
    def fit(self, X, y):

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        self.buckets = [[] for _ in range(self.num_buckets)]

        bucketed = []

        for n in range(self.num_classifiers):
            bucketed.append([self.hashfunc.hash(str(yi+n))%self.num_buckets for yi in self.y_])
            for yu in self.classes_:
                b = self.hashfunc.hash(str(yu+n))%self.num_buckets
                self.buckets[b].append(yu)

        self.bucketed_ = bucketed
        
        self.bucketed_t_ = np.array(bucketed).T

        for ci in range(self.num_classifiers):
            self.classifiers[ci].fit(self.X_, self.bucketed_[ci])

            
        self.judge.fit(self.bucketed_t_, self.y_)

        return self

    def predict(self, X):

        bucket_preds = []
        
        for ci in range(self.num_classifiers):
            bucket_preds.append(self.classifiers[ci].predict(X))
        
        bucket_preds_t = np.array(bucket_preds).T
        
        '''
        preds = []

        for p in range(X.shape[0]):
            candidates = {}
            for b in bucket_preds_t[p]:
                for yu in self.buckets[b]:
                    if yu not in candidates:
                        candidates[yu] = 1
                    else:
                        candidates[yu] += 1
            sorted_candidates = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
            preds.append(sorted_candidates[0])
            
            
        return [x for x,_ in preds]
        '''
        
        preds = self.judge.predict(bucket_preds_t)

        return preds












class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df[self.cols]


class FastTextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, df, y=None):
        return self

    def transform(self, df):

        if self.col != False:

            data = df[self.col].tolist()
            M = np.zeros((len(data), 100))
            for index in range(len(data)):
                M[index] = data[index]

            return M

        else:

            print(self.col)

            return df



'''
class NumericalSelector2(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df[self.cols].values

class TextSelector2(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def transform(self, df):
        return df[self.col]

    def fit(self, df, y=None):
        return self

class ScalerMixin(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, params):
        self.scaler = scaler
        self.params = params
        self.iScaler = None

    def transform(self, x):
        if self.iScaler is not None and x.shape[1] > 0:
            return self.iScaler.transform(x)
        else:
            return x

    def fit(self, x, y=None):
        if self.params['scale'] and x.shape[1] > 0:
            temp = self.params['scale']
            del self.params['scale']
            self.iScaler = self.scaler(**self.params)
            self.params['scale'] = temp
            self.iScaler.fit(x)
        return self
'''

def worker(
    train_X, train_y, validation_X, validation_y,
    vectorizers, selectors, scalers, classifiers, samplers,
    corpus, feats, fastt
    ):
    
    results = pd.DataFrame([], columns=[
        'Features',
        'Vectorizer',
        'V.params',
        'Selector',
        'Sel.params',
        'Scaler',
        'Sca.params',
        'Classifier',
        'C.params',
        'Sampler',
        'S.params',
        'Accuracy',
        'Precision',
        'Recall',
        'Fscore'])
    
    preds_validation = {
        'validation_y': validation_y,
        'preds': []
    }
    index = 0

    for f_title, ftr in feats:
        for v_title, vec, v_params in vectorizers:
            for v_p in v_params: 
                for c_title, clf, c_params in classifiers:
                    for c_p in c_params:
                        for s_title, smpl, s_params in samplers:
                            for s_p in s_params:
                                for sel_title, sel, sel_params in selectors:
                                    for sel_p in sel_params:
                                        for sca_title, sca, sca_params in scalers:
                                            for sca_p in sca_params:

                                                title = str(index)+'_'+ f_title + "/" +v_title + "/" + c_title + "/" + s_title + "/" + sel_title + "/" + sca_title

                                                print(title)
                                                print(v_p)
                                                print(c_p)
                                                print(s_p)
                                                print(sel_p)
                                                print(sca_p)
                                                
                                                #testy = smpl(**s_p)
                                                #continue

                                                build_union = []

                                                if fastt != False:
                                                    build_union.append(
                                                        ('embedded', Pipeline([
                                                            ('fasttext', FastTextSelector(fastt)),
                                                            #('scaling', MinMaxScaler()),
                                                        ]))
                                                    )

                                                if corpus != False:
                                                    build_union.append(
                                                        ('text', Pipeline([
                                                            ('article', ColumnSelector(corpus)),
                                                            ('vectorizer', vec(**v_p))
                                                        ]))
                                                    )

                                                if ftr != False:
                                                    build_union.append(
                                                        ('numerical', Pipeline([
                                                            ('meta', ColumnSelector(ftr)),
                                                            ('scaler', sca(**sca_p))
                                                        ]))
                                                    )


                                                model = Pipeline([
                                                    ('features', FeatureUnion(build_union)),
                                                    ('selector', sel(**sel_p)),
                                                    ('sampler', smpl(**s_p)),
                                                    ('classifier', clf(**c_p))
                                                ])

                                                '''
                                                model = Pipeline([
                                                    ('features', FeatureUnion([
                                                        ('embedded', Pipeline([
                                                            ('fasttext', fast_selector(**fast_col)),
                                                        ])),
                                                        ('text', Pipeline([
                                                            ('article', ColumnSelector(corpus)),
                                                            #('article', TextSelector2('Lemma_stripped')),
                                                            ('vectorizer', vec(**v_p))
                                                        ])),
                                                        ('numerical', Pipeline([
                                                            ('meta', ColumnSelector(ftr)),
                                                            #('meta', NumericalSelector2(ftr)),
                                                            ('scaler', sca(**sca_p))
                                                            #('scaler', ScalerMixin(sca, sca_p))
                                                        ])),
                                                    ])),
                                                    ('selector', sel(**sel_p)),
                                                    ('sampler', smpl(**s_p)),
                                                    #('scaler', sca(**sca_p)),
                                                    ('classifier', clf(**c_p))
                                                ])
                                                '''

                                                '''
                                                model = Pipeline([
                                                    ('features', FeatureUnion([
                                                        #('article', TextSelector2('Lemma_stripped')),
                                                        #('vec', vec(**v_p)),
                                                        ('article', TextSelector('Lemma_stripped', vec, v_p)),
                                                        ('meta', NumericalSelector(ftr, sca, sca_p)),
                                                    ])),
                                                    #('sampler', smpl(**s_p)),
                                                    ('selector', sel(**sel_p)),
                                                    #('scaler', sca(**sca_p)),
                                                    ('classifier', clf(**c_p))
                                                ])'''

                                                model.fit(train_X, train_y)
                                                preds = model.predict(validation_X)

                                                acc = accuracy_score(validation_y, preds)
                                                prec, reca, fsco, _ = precision_recall_fscore_support(validation_y, preds, average='macro')

                                                print(acc, prec, reca, fsco)

                                                #ax = plot_cf(validation_y, preds, title = title)

                                                results = results.append(pd.Series({
                                                    'Features': f_title,
                                                    'Vectorizer': v_title,
                                                    'V.params': v_p,
                                                    'Selector': sel_title,
                                                    'Sel.params': sel_p,
                                                    'Scaler': sca_title,
                                                    'Sca.params': sca_p,
                                                    'Classifier': c_title,
                                                    'C.params': c_p,
                                                    'Sampler': s_title,
                                                    'S.params': s_p,
                                                    'Accuracy': acc,
                                                    'Precision': prec,
                                                    'Recall': reca,
                                                    'Fscore': fsco
                                                }, name=title))

                                                
                                                preds_validation['preds'].append(preds)

                                                index += 1
        
    return results, preds_validation

def plot_cf(y, preds, title = False):

    """
    This function prints and plots the confusion matrix.
    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """

    if not title:
        title = 'Normalized confusion matrix'

    cm = confusion_matrix(y, preds)

    classes = unique_labels(y, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(18.5, 18.5)
    return fig

def print_stats(y, preds, title=False):
    print('Accuracy = {}'.format(accuracy_score(y, preds)))
    print('Classification report:')
    print(classification_report(y, preds))
    plot_cf(y, preds, title=title)