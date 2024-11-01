import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import pdb

from collections import Counter
from collections.abc import Mapping

# For clustered conformal
from .clustering_utils import embed_all_classes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#========================================
#  Misc.
#========================================

def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n

#========================================
#   Data preparation
#========================================

# Used for creating randomly sampled calibration dataset
def random_split(X, y, avg_num_per_class, seed=0):
    '''
    Randomly splits X and y into X1, y1, X2, y2. X1 and y1 will have of
    avg_num_per_class * num_classes examples. The remaining examples
    will be in X2 and y2.
    
    Inputs:
    - X: numpy array. axis 0 must be the same length as y
    - y: 1-D numpy array of class labels (classes should be 0-indexed)
    - avg_num_per_class: Average number of examples per class to include 
    in the data split. 
    - seed: (int) random seed for reproducibility
    
    Output: X1, y1, X2, y2
    '''

    np.random.seed(seed)
    
    num_classes = np.max(y) + 1
    
    num_samples = avg_num_per_class * num_classes
    
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False) # Numeric index
    idx2 = ~np.isin(np.arange(len(y)), idx1) # Boolean index
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]
    
    return X1, y1, X2, y2

# Used for creating balanced or stratified calibration dataset
def split_X_and_y_Orin(X, y, n_k, num_classes, seed=0, split='balanced'):
    '''
    Randomly generate two subsets of features X and corresponding labels y such that the
    first subset contains n_k instances of each class k and the second subset contains all
    other instances 
    
    Inputs:
        X: n x d array (e.g., matrix of softmax vectors)
        y: n x 1 array
        n_k: positive int or n x 1 array
        num_classes: total number of classes, corresponding to max(y)
        seed: random seed
        
    Output:
        X1, y1
        X2, y2
    '''
    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
        
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape)

    i = 0
    for k in range(num_classes):

        # Randomly select n instances of class k
        idx = np.argwhere(y==k).flatten()
        selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))

        X1[i:i+n_k[k]] = X[selected_idx]
        y1[i:i+n_k[k]] = k
        i += n_k[k]
        
        all_selected_indices[selected_idx] = 1
        
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    
    return X1, y1, X2, y2

def split_X_and_y(X, y, n_k, num_classes, seed=0, split='balanced'):
    '''
    Randomly generate two subsets of features X and corresponding labels y such that the
    first subset contains n_k instances of each class k and the second subset contains all
    other instances 
    
    Inputs:
        X: n x d array (e.g., matrix of softmax vectors)
        y: n x 1 array
        n_k: positive int or n x 1 array
        num_classes: total number of classes, corresponding to max(y)
        seed: random seed
        
    Output:
        X1, y1
        X2, y2
    '''
    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)

    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
        
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape, dtype=np.int32)
    all_selected_indices_sum = np.sum(all_selected_indices)
    #print("Number of selected indices:", all_selected_indices_sum)
    unique_selected_indices = np.unique(np.where(all_selected_indices == 1)[0])
    #print(f"Total unique selected indices: {len(unique_selected_indices)}")

    i = 0
    for k in range(num_classes):

        # Find instances of class k
        idx = np.argwhere(y==k).flatten()

        # Check if there are instances of class k
        if len(idx) == 0:
            print(f"No instances found for class {k}. Skipping...")
            continue
        
        # Determine the size for the current class
        size_for_k = min(n_k[k], len(idx))

        # Randomly select n instances of class k
        selected_idx = np.random.choice(idx, replace=False, size=size_for_k)
        #print(f"For class {k}, selected indices: {selected_idx}")


        X1[i:i+size_for_k] = X[selected_idx]
        y1[i:i+size_for_k] = k
        i += size_for_k

        #all_selected_indices[selected_idx] = True
        # Randomly select n instances of class k
    #print("Number of True values in all_selected_indices:", np.sum(all_selected_indices))
        
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    
    return X1, y1, X2, y2

def split_X_and_y_new(X, y, n_k, num_classes, seed=0, split='balanced'):
    

    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)

    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
        
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(len(y), dtype=np.bool_)

    i = 0
    for k in range(num_classes):

        # Find instances of class k
        idx = np.argwhere(y == k).flatten()

        # Check if there are instances of class k
        if len(idx) == 0:
            print(f"No instances found for class {k}. Skipping...")
            continue
        
        # Determine the size for the current class
        size_for_k = min(n_k[k], len(idx))

        # Randomly select n instances of class k
        selected_idx = np.random.choice(idx, replace=False, size=size_for_k)

        X1[i:i+size_for_k] = X[selected_idx]
        y1[i:i+size_for_k] = k
        i += size_for_k

        all_selected_indices[selected_idx] = True
        
        
    X2 = X[~all_selected_indices]
    y2 = y[~all_selected_indices]

    # Ensure that there's no overlap between X1 and X2
    assert not np.any(np.isin(X1, X2))
    
    return X1, y1, X2, y2


# Used for creating balanced or stratified calibration dataset
def split_X_and_y_EMNIST(X, y, n_k, num_classes, seed=0, split='balanced'):
    '''
    Randomly generate two subsets of features X and corresponding labels y such that the
    first subset contains n_k instances of each class k and the second subset contains all
    other instances 
    
    Inputs:
        X: n x d array (e.g., matrix of softmax vectors)
        y: n x 1 array
        n_k: positive int or n x 1 array
        num_classes: total number of classes, corresponding to max(y)
        seed: random seed
        
    Output:
        X1, y1
        X2, y2
    '''
    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
        
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape)

    i = 0
    for k in range(num_classes):

        # Find instances of class k
        idx = np.argwhere(y==k).flatten()

        # Check if there are instances of class k
        if len(idx) == 0:
            print(f"No instances found for class {k}. Skipping...")
            continue
        
        # Determine the size for the current class
        size_for_k = min(n_k[k], len(idx))

        # Randomly select n instances of class k
        selected_idx = np.random.choice(idx, replace=False, size=size_for_k)

        X1[i:i+size_for_k] = X[selected_idx]
        y1[i:i+size_for_k] = k
        i += size_for_k

        all_selected_indices[selected_idx] = 1
        # Randomly select n instances of class k
        
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    
    return X1, y1, X2, y2

def get_true_class_conformal_score(scores_all, labels):
    '''
    Extracts conformal scores that corresponds to the true class labels
    
    Inputs:
        scores_all: n x num_classes array 
        labels: length-n array of true class labels
        
    Output: length-n array of scores corresponding to true label for each entry
    '''
    return scores_all[np.arange(len(labels)), labels]

#========================================
#   General conformal utils
#========================================

def get_conformal_quantile(scores, alpha, default_qhat=np.inf, exact_coverage=False):
    '''
    Compute finite-sample-adjusted 1-alpha quantile of scores
    
    Inputs:
        - scores: num_instances-length array of conformal scores for true class. A higher score 
            indicates more uncertainty
        - alpha: float between 0 and 1 specifying coverage level
        - default_qhat: the value that will be returned if there are insufficient samples to compute
        the quantile. Should be np.inf if you want a coverage guarantee.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    
    '''
    if exact_coverage:
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        return exact_cov_params
    
    else:
        n = len(scores)

        if n == 0:
            # print(f'Using default q_hat of {default_qhat} because n={n}')
            return default_qhat

        val = np.ceil((n+1)*(1-alpha))/n
        if val > 1:
            # print(f'Using default q_hat of {default_qhat} because n={n} and quantile={val}')
            qhat = default_qhat
        else:
            qhat = np.quantile(scores, val, method='inverted_cdf')

        return qhat

#========================================
#   Standard conformal prediction
#========================================

def compute_qhat(scores_all, true_labels, alpha, exact_coverage=False, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores, or num_instances-length array of 
        conformal scores for true class. A higher score indicates more uncertainty
        - true_labels: num_instances length array of ground truth labels
        - alpha: float between 0 and 1 specifying coverage level
        - plot_scores: If True, plot histogram of true class scores 
    '''
    # If necessary, select scores that correspond to correct label
    if len(scores_all.shape) == 2:
        scores = np.squeeze(np.take_along_axis(scores_all, np.expand_dims(true_labels, axis=1), axis=1))
    else:
        scores = scores_all
    
    q_hat = get_conformal_quantile(scores, alpha, exact_coverage=exact_coverage)
    
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hat

# Create prediction sets
def create_prediction_sets(scores_all, q_hat, exact_coverage=False):
    '''
    Create standard conformal prediction sets
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores
        - q_hat: conformal quantile, as returned from compute_qhat()
    '''
    if exact_coverage:
        assert isinstance(q_hat, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_a, q_b, gamma = q_hat['q_a'], q_hat['q_b'], q_hat['gamma']
        set_preds = construct_exact_coverage_standard_sets(q_a, q_b, gamma, scores_all)
        
    else:   
        assert(not hasattr(q_hat, '__iter__')), "q_hat should be a single number and not a list or array"
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            set_preds.append(np.where(scores_all[i,:] <= q_hat)[0])
        
    return set_preds

# Standard conformal pipeline
def standard_conformal(cal_scores_all, cal_labels, val_scores_all, val_labels, alpha, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for standard conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    '''
    standard_qhat = compute_qhat(cal_scores_all, cal_labels, alpha, exact_coverage=exact_coverage)
    standard_preds = create_prediction_sets(val_scores_all, standard_qhat, exact_coverage=exact_coverage)
    class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics = compute_all_metrics(val_labels, standard_preds, alpha)
    
    return standard_qhat, class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics

#========================================
#   Classwise conformal prediction
#========================================

# Unused in paper (only necessary if we apply regularization)
def reconformalize(qhats, scores, labels, alpha, adjustment_min=-1, adjustment_max=1):
    '''
    Adjust qhats by additive factor so that marginal coverage of 1-alpha is achieved
    '''
    print('Applying additive adjustment to qhats')
    # ===== Perform binary search =====
    # Convergence criteria: Either (1) marginal coverage is within tol of desired or (2)
    # quantile_min and quantile_max differ by less than .001, so there is no need to try 
    # to get a more precise estimate
    tol = 0.0005

    marginal_coverage = 0
    while np.abs(marginal_coverage - (1-alpha)) > tol:

        adjustment_guess = (adjustment_min +  adjustment_max) / 2
        print(f"\nCurrent adjustment: {adjustment_guess:.6f}")

        curr_qhats = qhats + adjustment_guess 

        preds = create_classwise_prediction_sets(scores, curr_qhats)
        marginal_coverage = compute_coverage(labels, preds)
        print(f"Marginal coverage: {marginal_coverage:.4f}")

        if marginal_coverage > 1 - alpha:
            adjustment_max = adjustment_guess
        else:
            adjustment_min = adjustment_guess
        print(f"Search range: [{adjustment_min}, {adjustment_max}]")

        if adjustment_max - adjustment_min < .00001:
            adjustment_guess = adjustment_max # Conservative estimate, which ensures coverage
            print("Adequate precision reached; stopping early.")
            break
            
    print('Final adjustment:', adjustment_guess)
    qhats += adjustment_guess
    
    return qhats 

def compute_class_specific_qhats(cal_scores_all, cal_true_labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all: 
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed). 
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the 
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat. 
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class 
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1 
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for 
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhat = compute_qhat(cal_scores_all, alpha, default_qhat=default_qhat)
        q_a, q_b, gamma = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, 
                                                                       num_classes, alpha, 
                                                                       default_qhat=default_qhat, null_params=null_qhat)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha=alpha)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class
            idx = (cal_true_labels == k)
            scores = cal_scores_all[idx]

            class_cts[k] = scores.shape[0]

            q_hats[k] = get_conformal_quantile(scores, alpha, default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))


        # Optionally apply shrinkage 
        if regularize:
            N = num_classes
            n_k = np.maximum(class_cts, 1) # So that classes that never appear do not cause division by 0 issues. 
            shrinkage_factor = .03 * n_k # smaller = less shrinkage
            shrinkage_factor = np.minimum(shrinkage_factor, 1)
            print('SHRINKAGE FACTOR:', shrinkage_factor)  
            print(np.min(shrinkage_factor), np.max(shrinkage_factor))
            q_hats = default_qhat + shrinkage_factor * (q_hats - default_qhat)

            # Correct qhats via additive factor to achieve marginal coverage
            q_hats = reconformalize(q_hats, reserved_scores, reserved_labels, alpha)


        return q_hats

# Create classwise prediction sets
def create_classwise_prediction_sets(scores_all, q_hats, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            set_preds.append(np.where(scores_all[i,:] <= q_hats)[0])

    return set_preds

# Classwise conformal pipeline
def classwise_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha, class_gap, 
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
    See compute_class_specific_qhats() docstring for more details about expected inputs.
    '''
    num = len(totalcal_labels)/num_classes
    cc_alpha = alpha - (class_gap/np.sqrt(num))
    
    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=cc_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    classwise_preds = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
    #print(classwise_preds)

    class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics
                

#========================================
#   Clustered conformal prediction
#========================================

def compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_true_labels, alpha, 
                                   null_qhat='standard', exact_coverage=False):
    '''
    Computes cluster-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cluster_assignments: num_classes length array where entry i is the index of the cluster that class i belongs to.
          Clusters should be 0-indexed. Rare classes can be assigned to cluster -1 and they will automatically be given
          qhat_k = default_qhat. 
        - cal_scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i.
         Alternatively, a num_instances-length array of conformal scores for true class
        - cal_true_labels: num_instances length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - null_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to null_qhat.
        If null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
        - exact_coverage: If True, return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
         
    Output:
        num_classes length array where entry i is the quantile corresponding to the cluster that class i belongs to. 
        All classes in the same cluster have the same quantile.
        
        OR (if exact_coverage=True), dict containing clustered conformal parameters needed to achieve exact coverage
    '''
    # If we want the null_qhat to be the standard qhat, we should compute this using the original class labels
    if null_qhat == 'standard' and not exact_coverage:
        null_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha)
            
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
        
    # Edge case: all cluster_assignments are -1. 
    if np.all(cluster_assignments==-1):
        if exact_coverage:
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha) # Assign standard conformal params to null cluster
            q_as = null_qa * np.ones(cluster_assignments.shape)
            q_bs = null_qb * np.ones(cluster_assignments.shape)
            gammas = null_gamma * np.ones(cluster_assignments.shape)
            return q_as, q_bs, gammas
        else:
            return null_qhat * np.ones(cluster_assignments.shape)
    
    # Map true class labels to clusters
    cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])
    
    # Compute cluster qhats
    if exact_coverage:
        if null_qhat == 'standard':
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha) # Assign standard conforml params to null cluster
            null_params = {'q_a': null_qa, 'q_b': null_qb, 'gamma': null_gamma}
        clustq_as, clustq_bs, clustgammas = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_clusters,
                                                                          num_classes=np.max(cluster_assignments)+1, 
                                                                          alpha=alpha, 
                                                                          default_qhat=np.inf, null_params=null_params)
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        q_as = np.array([clustq_as[cluster_assignments[k]] for k in range(num_classes)])
        q_bs = np.array([clustq_bs[cluster_assignments[k]] for k in range(num_classes)])
        gammas = np.array([clustgammas[cluster_assignments[k]] for k in range(num_classes)])

        return q_as, q_bs, gammas
   
    else:
            
        cluster_qhats = compute_class_specific_qhats(cal_scores_all, cal_true_clusters, 
                                                     alpha=alpha, num_classes=np.max(cluster_assignments)+1,
                                                     default_qhat=np.inf,
                                                     null_qhat=null_qhat)                            
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        class_qhats = np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])

        return class_qhats

    # Note: To create prediction sets, just pass class_qhats into create_classwise_prediction_sets()
    
    
def clustered_conformal(totalcal_scores_all, totalcal_labels,
                        alpha, cluster_gap, 
                        val_scores_all, val_labels, num_classes,
                        frac_clustering='auto', num_clusters='auto',
                        split='random',
                        exact_coverage=False, seed=0):
    '''
    Use totalcal_scores and totalcal_labels to compute conformal quantiles for each
    class using the clustered conformal procedure. Optionally evaluates 
    performance on val_scores and val_labels
    
    Clustered conformal procedure:
        1. Filter out classes where np.ceil((num_samples+1)*(1-alpha)) / num_samples < 1. 
           Assign those classes the standard qhat
        2. To select n_clustering (or frac_clustering) and num_clusters, apply heuristic 
           (which takes as input num_samples and num_classes) to num_samples of rarest remaining class
        3. Run clustering with chosen hyperparameters. Estimate a qhat for each cluster.
    
    Inputs:
         - totalcal_scores: num_instances x num_classes array where 
           cal_scores_all[i,j] = score of class j for instance i
         - totalcal_labels: num_instances-length array of true class labels (0-indexed classes)
         - alpha: number between 0 and 1 that determines coverage level.
         Coverage level will be 1-alpha.
         - n_clustering: Number of points per class to use for clustering step. The remaining
         points are used for the conformal calibration step.
         - num_clusters: Number of clusters to group classes into
         - val_scores: num_val_instances x num_classes array, or None. If not None, 
         the class coverage gap and average set sizes will be computed on val_scores
         and val_labels.
         - val_labels: num_val_instances-length array of true class labels, or None. 
         If not None, the class coverage gap and average set sizes will be computed 
         on val_scores and val_labels.
         - split: How to split data between clustering step and calibration step. Options are
         'balanced' (sample n_clustering per class), 'proportional' (sample proportional to
         distribution such that rarest class has n_clustering example), 'doubledip' (don't
         split and use all data for both steps, or 'random' (each example is assigned to
         clustering step with some fixed probability) 
         - exact_coverage: If True, randomize the prediction sets to achieve exact 1-alpha coverage. Otherwise, 
         the sets will be unrandomized but slightly conservative
         - seed: Random seed. Affects split between clustering set and proper calibraton set
         
    Outputs:
        - qhats: num_classes-length array where qhats[i] = conformal quantial estimate for class i
        - [Optionally, if val_scores and val_labels are not None] 
            - val_preds: clustered conformal predictions on val_scores
            - val_class_coverage_gap: Class coverage gap, compute on val_scores and val_labels
            - val_set_size_metrics: Dict containing set size metrics, compute on val_scores and val_labels
            
    '''
    np.random.seed(seed) 

    num = len(totalcal_labels)/num_classes
    clc_alpha = alpha - (cluster_gap/np.sqrt(num))
    
    def get_rare_classes(labels, alpha, num_classes):
        thresh = get_quantile_threshold(alpha)
        classes, cts = np.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]
        
        # Also included any classes that are so rare that we have 0 labels for it
        zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
        rare_classes = np.concatenate((rare_classes, zero_ct_classes))
        
        return rare_classes
        
    def remap_classes(labels, rare_classes):
        '''
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed

        Outputs:
            - remaining_idx: Boolean array the same length as labels. Entry i is True
            iff labels[i] is not in rare_classes 
            - remapped_labels: Array that only contains the entries of labels that are 
            not in rare_classes (in order) 
            - remapping: Dict mapping old class index to new class index

        '''
        remaining_idx = ~np.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1
        return remaining_idx, remapped_labels, remapping
    
    # Data preperation: Get conformal scores for true classes
    num_classes = totalcal_scores_all.shape[1]
    totalcal_scores = get_true_class_conformal_score(totalcal_scores_all, totalcal_labels)
    # totalcal_scores = totalcal_scores_all
    
    # 1) Apply heuristic to choose hyperparameters if not prespecified
    if frac_clustering == 'auto' and num_clusters == 'auto':
        cts_dict = Counter(totalcal_labels)
        cts = [cts_dict.get(k, 0) for k in range(num_classes)]
        n_min = min(cts)
        n_thresh = get_quantile_threshold(clc_alpha) 
        n_min = max(n_min, n_thresh) # Classes with fewer than n_thresh examples will be excluded from clustering
        num_remaining_classes = np.sum(np.array(list(cts)) >= n_min)

        n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)
        print(f'n_clustering={n_clustering}, num_clusters={num_clusters}')
        # Convert n_clustering to fraction relative to n_min
        frac_clustering = n_clustering / n_min

        
    # 2a) Split data
    if split == 'proportional':
        n_k = [int(frac_clustering*cts[k]) for k in range(num_classes)]
        scores1, labels1, scores2, labels2 = split_X_and_y_Orin(totalcal_scores, 
                                                           totalcal_labels, 
                                                           n_k, 
                                                           num_classes=num_classes, 
                                                           seed=0)
#                                                            split=split, # Balanced or stratified sampling 
    elif split == 'doubledip':
        scores1, labels1 = totalcal_scores, totalcal_labels
        scores2, labels2 = totalcal_scores, totalcal_labels
    elif split == 'random':
        # Each point is assigned to clustering set w.p. frac_clustering 
        idx1 = np.random.uniform(size=(len(totalcal_labels),)) < frac_clustering 
        scores1 = totalcal_scores[idx1]
        labels1 = totalcal_labels[idx1]
        scores2 = totalcal_scores[~idx1]
        labels2 = totalcal_labels[~idx1]
        
    else:
        raise Exception('Invalid split. Options are balanced, proportional, doubledip, and random')

    # 2b)  Identify "rare" classes = classes that have fewer than 1/alpha - 1 examples 
    # in the clustering set 
    rare_classes = get_rare_classes(labels1, clc_alpha, num_classes)
    print(f'{len(rare_classes)} of {num_classes} classes are rare in the clustering set'
          ' and will be assigned to the null cluster')
    
    # 3) Run clustering
    if num_classes - len(rare_classes) > num_clusters and num_clusters > 1:  
        # Filter out rare classes and re-index
        remaining_idx, filtered_labels, class_remapping = remap_classes(labels1, rare_classes)
        filtered_scores = scores1[remaining_idx]
        
        # Compute embedding for each class and get class counts
        embeddings, class_cts = embed_all_classes(filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)
    
        kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10).fit(embeddings, sample_weight=np.sqrt(class_cts))
        nonrare_class_cluster_assignments = kmeans.labels_  

        # Print cluster sizes
        print(f'Cluster sizes:', [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()])

        # Remap cluster assignments to original classes. Any class not included in kmeans clustering is a rare 
        # class, so we will assign it to cluster "-1" = num_clusters by Python indexing
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        for cls, remapped_cls in class_remapping.items():
            cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
    else: 
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        print('Skipped clustering because the number of clusters requested was <= 1')
        
    # 4) Compute qhats for each cluster
    cal_scores_all = scores2
    cal_labels = labels2
    if exact_coverage: 
        q_as, q_bs, gammas = compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_labels, alpha = clc_alpha, 
                                   null_qhat='standard', exact_coverage=True)
        
    else: 
        qhats = compute_cluster_specific_qhats(cluster_assignments, 
                   cal_scores_all, cal_labels, 
                   alpha=clc_alpha, 
                   null_qhat='standard')
        

    # 5) [Optionally] Apply to val set. Evaluate class coverage gap and set size 
    if (val_scores_all is not None) and (val_labels is not None):
        if exact_coverage:
            preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, val_scores_all)
            qhats = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas} # Alias for function return
        else:
            preds = create_classwise_prediction_sets(val_scores_all, qhats)
        class_coverage_metrics, class_cov_metrics, set_size_metrics, tail_metrics = compute_all_metrics(val_labels, preds, alpha,
                                                                  cluster_assignments=cluster_assignments)

        # Add # of classes excluded from clustering to class_cov_metrics
        # class_cov_metrics['num_unclustered_classes'] = len(rare_classes)
        
        return qhats, class_coverage_metrics, class_cov_metrics, set_size_metrics, tail_metrics
    else:
        return qhats

    
def get_clustering_parameters(num_classes, n_totalcal):
    '''
    Returns a guess of good values for num_clusters and n_clustering based solely 
    on the number of classes and the number of examples per class. 
    
    This relies on two heuristics:
    1) We want at least 150 points per cluster on average
    2) We need more samples as we try to distinguish between more distributions. 
    To distinguish between 2 distribution, want at least 4 samples per class. 
    To distinguish between 5 distributions, want at least 10 samples per class. 
    
    Output: n_clustering, num_clusters
    
    '''
    # Alias for convenience
    K = num_classes
    N = n_totalcal
    
    n_clustering = int(N*K/(75+K))
    num_clusters = int(np.floor(n_clustering / 2))
    
    return n_clustering, num_clusters

#========================================
#   Conformal variant: exact coverage via randomization 
#========================================

def get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf):
    '''
    Compute the quantities necessary for performing conformal prediction with exact coverage
    
    Inputs:
        score: length-n array of conformal scores
        alpha: float between 0 and 1 denoting the desired coverage level
        default_qhat = value used when n is too small for computing the relevant quantiles
    Outputs:
        q_a: The smallest score that is larger than (n+1) * (1-alpha) of the other scores (= normal conformal qhat)
        q_b: The smallest score that is larger than (n+1) * (1-alpha) - 1 of the other scores
        gamma: value between 0 and 1 such that gamma * q_a + (1-gamma)*g_b = 1-alpha
    '''

    n = len(scores)
    
    if n == 0:
        return np.inf, np.inf, 1
    
    val_a = np.ceil((n+1)*(1-alpha)) / n
    if val_a > 1:
        q_a = default_qhat
    else:
        q_a = np.quantile(scores, val_a, method='inverted_cdf')
        
    val_b = (np.ceil((n+1)*(1-alpha)-1)) / n
    if val_b > 1:
        q_b = default_qhat
        
    else:
        q_b = np.quantile(scores, val_b, method='inverted_cdf') 
        
    if val_a > 1 and val_b > 1:
        gamma = 1 # Any value would work, since q_a and q_b are both equal to default_qhat
    else:
        overcov = np.ceil((n+1)*(1-alpha))/(n+1) - (1-alpha) # How much coverage using q_a will exceed 1 - alpha
        undercov = (1-alpha) - (np.ceil((n+1)*(1-alpha))-1)/(n+1)  #  How much coverage using q_b will undershoot 1 - alpha
        gamma = undercov / (undercov + overcov)

    return q_a, q_b, gamma

def construct_exact_coverage_standard_sets(q_a, q_b, gamma, scores_all):
    '''
    Construct randomized standard conformal sets
    
    Inputs:
        - q_a, q_b, gamma: as output by get_exact_coverage_conformal_params()
        - scores_all: num_instances x num_classes array
    '''
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        if np.random.rand() < gamma: # Bernoulli random var
            q_hat = q_a
        else:
            q_hat = q_b
        set_preds.append(np.where(scores_all[i,:] <= q_hat)[0]) 

    return set_preds

def compute_exact_coverage_class_specific_params(scores_all, labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_params=None):
    '''
    Compute the quantities necessary for performing classwise conformal prediction with exact coverage
    
    Inputs:
        - scores_all: (num_instances, num_classes) array where scores_all[i,j] = score of class j for instance i
        - labels: (num_instances,) array of true class labels
        - num_classes: Number of classes
        - alpha: number between 0 and 1 specifying desired coverage level
        - default_qhat: Quantile that will be used when there is insufficient data to compute a quantile
        - null_params: Dict of {'q_a': ..., 'q_b': ...., 'gamma', ...} to be assigned to
               class -1 (will be appended to end of param lists). Not needed if -1 does not appear in labels
    '''
    
    q_as = np.zeros((num_classes,))   
    q_bs = np.zeros((num_classes,)) 
    gammas = np.zeros((num_classes,)) 
    for k in range(num_classes):
        # Only select data for which k is true class
        
        idx = (labels == k)
        if len(scores_all.shape) == 2:
            scores = scores_all[idx, k]
        else:
            scores = scores_all[idx]
        
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=default_qhat)
        q_as[k] = q_a
        q_bs[k] = q_b
        gammas[k] = gamma
        
    if -1 in labels:
        q_as = np.concatenate((q_as, [null_params['q_a']]))
        q_bs = np.concatenate((q_bs, [null_params['q_b']]))
        gamma = np.concatenate((gammas, [null_params['gamma']]))
        
    return q_as, q_bs, gammas
  
def construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
        q_hats = np.where(Bs, q_as, q_bs)
        #q_hat = max(q_hats)
        set_preds.append(np.where(scores_all[i,:] <= q_hats)[0]) 

    return set_preds

#========================================
#   Adaptive Prediction Sets (APS)
#========================================

def get_APS_scores(softmax_scores, labels, randomize=True, seed=0):
    '''
    Compute conformity score defined in Romano et al, 2020
    (Including randomization, unless randomize is set to False)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
    
    Output: 
        length-n array of APS scores
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
    
def get_APS_scores_all(softmax_scores, randomize=True, seed=0):
    '''
    Similar to get_APS_scores(), except the APS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

#========================================
#   Regularized Adaptive Prediction Sets (RAPS)
#========================================

def get_RAPS_scores(softmax_scores, labels, lmbda=.01, kreg=5, randomize=True, seed=0):
    '''
    Essentially the same as get_APS_scores() except with regularization.
    See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
        lmbda, kreg: regularization parameters
    Output: 
        length-n array of APS scores
    
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    
    # Regularization
    y_rank = pi.argsort(1)[range(labels_calib.shape[0]), labels_calib] + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=y_rank.shape))
    scores += reg
    
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
        
def get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True, seed=0):
    '''
    Similar to get_RAPS_scores(), except the RAPS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    
    # Regularization (pretend each class is true label)
    y_rank = pi.argsort(1) + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=scores.shape))
 
    scores += reg
        
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

    
#========================================
#   Evaluation
#========================================

# Helper function for computing accuracy (marginal coverage) of confidence sets
def compute_coverage(true_labels, set_preds):
    true_labels = np.array(true_labels) # Convert to numpy to avoid weird pytorch tensor issues
    num_correct = 0
    for true_label, preds in zip(true_labels, set_preds):
        if true_label in preds:
            num_correct += 1
    set_pred_acc = num_correct / len(true_labels)
    
    return set_pred_acc

# Helper function for computing class-conditional coverage of confidence sets
def compute_class_specific_coverage(true_labels, set_preds):
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    class_idx = np.zeros((num_classes,))
    class_size = np.zeros((num_classes,))
    #class_idx = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        selected_preds = [set_preds[i] for i in idx]
        #num_correct = np.sum([1 if np.any(pred_set == k) else 0 for pred_set in selected_preds])
        num_correct = np.sum([1 if k in pred_set else 0 for pred_set in selected_preds])
        #print(num_correct)
        class_specific_cov[k] = num_correct / len(selected_preds)
        class_idx[k] = k
        curr_set_sizes = [len(x) for x in selected_preds]
        class_size[k] = np.mean(curr_set_sizes)

    #classes = np.arange(num_classes)

    return class_specific_cov, class_idx, class_size

# Helper function for computing average set size
def compute_avg_set_size(list_of_arrays):
    return np.mean([len(arr) for arr in list_of_arrays])

def compute_all_metrics(val_labels, preds, alpha, cluster_assignments=None):
    class_cond_cov, class_idx, class_size = compute_class_specific_coverage(val_labels, preds)
        
    # Average class coverage gap
    avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))

    # Average gap for classes that are over-covered
    #overcov_idx = (class_cond_cov > (1-alpha))
    #overcov_gap = np.mean(class_cond_cov[overcov_idx] - (1-alpha))

    # Average gap for classes that are under-covered
    #thresh = 0.05
    #undercov_idx = (class_cond_cov < (1-alpha))
    # Create a boolean mask of classes with Class Conditional Coverage less than 1 - alpha
    mask = class_cond_cov < (1 - alpha)

    # Sum up the mask to get the count of such classes
    num_classes_below = np.sum(mask)

    undercov_ratio = num_classes_below/len(class_cond_cov)
    
    # Fraction of classes that are at least 10% under-covered
    #thresh = .1
    #very_undercovered = np.mean(class_cond_cov < (1-alpha-thresh))
    
    # Max gap
    max_gap = np.max(np.abs(class_cond_cov - (1-alpha)))

    # Marginal coverage
    marginal_cov = compute_coverage(val_labels, preds)
    curr_set_sizes = [len(x) for x in preds]

    #avg_set_size = np.mean(curr_set_sizes)

    num_classes = len(class_size)
    one_quarter = num_classes // 4
    remainder = num_classes % 4

    # Adjust if the number of classes is not divisible by 4
    num_classes = len(class_size)
    one_quarter = num_classes // 4
    remainder = num_classes % 4

    # Compute indices for quartiles
    one_quarter_adjusted = one_quarter
    two_quarters_adjusted = 2 * one_quarter
    three_quarters_adjusted = 3 * one_quarter + remainder

    # Compute average set sizes for the three categories
    majority_avg_size = np.mean(class_size[0:one_quarter_adjusted])  # First quarter
    medium_avg_size = np.mean(class_size[one_quarter_adjusted:three_quarters_adjusted])  # Middle half
    minority_avg_size = np.mean(class_size[three_quarters_adjusted:num_classes])  # Last quarter

    # Add them to metrics 
    tail_metrics = {
        'majority_avg_set_size': majority_avg_size,
        'medium_avg_set_size': medium_avg_size,
        'minority_avg_set_size': minority_avg_size
    }

    class_coverage_metrics = {'Class Conditional Coverage': class_cond_cov,
                              'Prediction Set Size': class_size,
                              'Condition on Class': class_idx
                             }
    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'undercov ratio': undercov_ratio, 
                         #'overcov_gap': overcov_gap, 
                         #'average set size': class_size,
                         'marginal_cov': marginal_cov,
                         'average set size': np.mean(class_size)
                         #'cluster_assignments': cluster_assignments # Also save class cluster assignments
                        }

    set_size_metrics = {'mean': np.mean(curr_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])}
    #curr_set_sizes_metrics = {'each class set size': curr_set_sizes}
    
    print('Marginal Coverage:', marginal_cov)
    # print('Class Conditional Coverage:', class_cond_cov)
    print('Under Covered Ratio:', undercov_ratio)
    print('Average Set Size:', np.mean(class_size))
    # print('Prediction Set Size:', class_size)
    
    return class_coverage_metrics, class_cov_metrics, set_size_metrics, tail_metrics

#========================================================
#   Truncated class-wise conformal prediction (uncorrect)
#========================================================

def uncorrect_truncated_compute_class_specific_qhats(cal_scores_all, cal_true_labels, totalcal_ranks, suit_indices, alphas, num_classes,
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all:
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed).
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat.
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhats = uncorrect_truncated_compute_qhat(cal_scores_all, totalcal_ranks, suit_indices, alphas, num_classes, default_qhat=default_qhat)
        q_as, q_bs, gammas = uncorrect_truncated_compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, totalcal_ranks,
                                                                       suit_indices, alphas, num_classes, 
                                                                       default_qhat=default_qhats, null_params=null_qhat)
        exact_cov_params = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhats = uncorrect_truncated_compute_qhat(cal_scores_all, cal_true_labels, totalcal_ranks, suit_indices, alphas, num_classes)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class
            idx = (cal_true_labels == k) & (totalcal_ranks < suit_indices[k])
            scores = cal_scores_all[idx]

            class_cts[k] = scores.shape[0]

            q_hats[k] = get_conformal_quantile(scores, alphas[k], default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))

        return q_hats

# Create classwise prediction sets
def uncorrect_truncated_create_classwise_prediction_sets(scores_all, q_hats, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = uncorrect_truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all, k)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            #q_hat = max(q_hats)
            set_preds.append(np.where(scores_all[i,:] <= q_hats)[0])

    return set_preds


# # Classwise conformal pipeline
# def truncated_classwise_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
#                          num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
#     '''
#     Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
#     If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
#     unrandomized conservative sets. 
#     Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
#     See compute_class_specific_qhats() docstring for more details about expected inputs.
#     '''
    
#     classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels,  
#                                                    alpha=alpha, 
#                                                    num_classes=num_classes,
#                                                    default_qhat=default_qhat, regularize=regularize,
#                                                    exact_coverage=exact_coverage)
#     classwise_preds = truncated_create_classwise_prediction_sets(val_scores_all, k, classwise_qhats, exact_coverage=exact_coverage)

#     class_coverage_metrics, coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
#     return classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics

def uncorrect_truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, val_scores_all, val_labels, alpha, truncated_gap,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    err_matrix = 1 - accuracy_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num))
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    #print(suit_k)
    #print(k_max)
    mix_paras = np.linspace(0, 1, rank_all-k_max)
    best_para = 1

    for mix_para in mix_paras:
        test_indices = [math.ceil((1 - mix_para) * suit_k[i] + num_classes * mix_para) for i in range(num_classes)]
        #print(test_indices)
        test_err = [err_matrix[test_indices[i]-1, i] for i in range(num_classes)]
        #print(test_err)
        test_alpha = [tc_alpha - err for err in test_err]
        #print(test_alpha)
    # for k in range(k_max-1, num_classes):
    #     suit_err = err_matrix[k,:]
    #     test_alpha = np.array(alpha - suit_err)
        #print(test_alpha)
        classwise_qhats = uncorrect_truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, totalcal_ranks, 
                                                   suit_indices = test_indices, alphas=test_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
        #print(classwise_qhats)
        classwise_preds = uncorrect_truncated_create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
        curr_set_sizes = [len(x) for x in classwise_preds]
        avg_size = np.mean(curr_set_sizes)
        if avg_size < smallest_ps:
            smallest_ps = avg_size
            best_para = mix_para
    
    print(f'best_para = {best_para}')
    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(num_classes)]
    #print(best_indices)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(num_classes)]
    tilde_alpha = [tc_alpha - err for err in best_err]
    #print(tilde_alpha)
    classwise_qhats = uncorrect_truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, totalcal_ranks,
                                                   suit_indices = best_indices, alphas=tilde_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    classwise_preds = uncorrect_truncated_create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
    
    class_coverage_metrics, coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, classwise_preds, class_coverage_metrics, coverage_metrics, set_size_metrics


def uncorrect_truncated_compute_qhat(scores_all, true_labels, totalcal_ranks, suit_indices, alphas, num_classes, exact_coverage=False, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores, or num_instances-length array of 
        conformal scores for true class. A higher score indicates more uncertainty
        - true_labels: num_instances length array of ground truth labels
        - alpha: float between 0 and 1 specifying coverage level
        - plot_scores: If True, plot histogram of true class scores 
    '''
    # If necessary, select scores that correspond to correct label
    if len(scores_all.shape) == 2:
        scores = np.squeeze(np.take_along_axis(scores_all, np.expand_dims(true_labels, axis=1), axis=1))
    else:
        scores = scores_all

    q_hats = np.zeros((num_classes))   
    for k in range(num_classes):
        idx = (cal_true_labels == k) & (totalcal_ranks < suit_indices[k])
        rank_scores = scores_all[idx]
        q_hats[k] = get_conformal_quantile(rank_scores, alphas[k], exact_coverage=exact_coverage)
        #q_hats.append(q_hat)
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hats

# def truncated_get_exact_coverage_conformal_params(scores, alphas, default_qhat=np.inf):
#     '''
#     Compute the quantities necessary for performing conformal prediction with exact coverage for multiple alphas
    
#     Inputs:
#         score: length-n array of conformal scores
#         alphas: array of floats between 0 and 1 denoting the desired coverage levels
#         default_qhat = value used when n is too small for computing the relevant quantiles
#     Outputs:
#         q_a_vals: An array of the smallest scores that is larger than (n+1) * (1-alpha) of the other scores (= normal conformal qhat) for each alpha
#         q_b_vals: An array of the smallest scores that is larger than (n+1) * (1-alpha) - 1 of the other scores for each alpha
#         gammas: An array of values between 0 and 1 such that gamma * q_a + (1-gamma)*g_b = 1-alpha for each alpha
#     '''

#     n = len(scores)
#     q_a_vals = []
#     q_b_vals = []
#     gammas = []

#     for alpha in alphas:

#         if n == 0:
#             q_a_vals.append(np.inf)
#             q_b_vals.append(np.inf)
#             gammas.append(1)
#             continue

#         val_a = np.ceil((n+1)*(1-alpha)) / n
#         if val_a > 1:
#             q_a = default_qhat
#         else:
#             q_a = np.quantile(scores, val_a, method='inverted_cdf')

#         val_b = (np.ceil((n+1)*(1-alpha)-1)) / n
#         if val_b > 1:
#             q_b = default_qhat
#         else:
#             q_b = np.quantile(scores, val_b, method='inverted_cdf') 

#         if val_a > 1 and val_b > 1:
#             gamma = 1 # Any value would work, since q_a and q_b are both equal to default_qhat
#         else:
#             overcov = np.ceil((n+1)*(1-alpha))/(n+1) - (1-alpha) # How much coverage using q_a will exceed 1 - alpha
#             undercov = (1-alpha) - (np.ceil((n+1)*(1-alpha))-1)/(n+1)  #  How much coverage using q_b will undershoot 1 - alpha
#             gamma = undercov / (undercov + overcov)

#         q_a_vals.append(q_a)
#         q_b_vals.append(q_b)
#         gammas.append(gamma)

#     return q_a_vals, q_b_vals, gammas

def uncorrect_truncated_compute_exact_coverage_class_specific_params(scores_all, labels, totalcal_ranks, suit_indices, num_classes, alphas, 
                                 default_qhat=np.inf, null_params=None):
    '''
    Compute the quantities necessary for performing classwise conformal prediction with exact coverage
    
    Inputs:
        - scores_all: (num_instances, num_classes) array where scores_all[i,j] = score of class j for instance i
        - labels: (num_instances,) array of true class labels
        - num_classes: Number of classes
        - alpha: number between 0 and 1 specifying desired coverage level
        - default_qhat: Quantile that will be used when there is insufficient data to compute a quantile
        - null_params: Dict of {'q_a': ..., 'q_b': ...., 'gamma', ...} to be assigned to
               class -1 (will be appended to end of param lists). Not needed if -1 does not appear in labels
    '''
    
    q_as = np.zeros((num_classes,))   
    q_bs = np.zeros((num_classes,)) 
    gammas = np.zeros((num_classes,)) 
    for k in range(num_classes):
        # Only select data for which k is true class
        
        idx = (labels == k) & (totalcal_ranks < suit_indices[k])
        if len(scores_all.shape) == 2:
            scores = scores_all[idx, k]
        else:
            scores = scores_all[idx]
        
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha[k], default_qhat=default_qhat[k])
        q_as[k] = q_a
        q_bs[k] = q_b
        gammas[k] = gamma
        
    if -1 in labels:
        q_as = np.concatenate((q_as, [null_params['q_a']]))
        q_bs = np.concatenate((q_bs, [null_params['q_b']]))
        gamma = np.concatenate((gammas, [null_params['gamma']]))
        
    return q_as, q_bs, gammas
# def truncated_construct_exact_coverage_standard_sets(q_a, q_b, gamma, scores_all):
#     '''
#     Construct randomized standard conformal sets
    
#     Inputs:
#         - q_a, q_b, gamma: as output by get_exact_coverage_conformal_params()
#         - scores_all: num_instances x num_classes array
#     '''
#     # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
#     scores_all = np.array(scores_all)
#     set_preds = []
#     num_samples = len(scores_all)
#     for i in range(num_samples):
#         if np.random.rand() < gamma: # Bernoulli random var
#             q_hat = q_a
#         else:
#             q_hat = q_b
#     #set_preds.append(np.where(scores_all[i,:] <= q_hat)[0]) & (np.arange(scores_all.shape[1]) <= k))[0] 
#     valid_indices = np.where((scores_all[i, :] <= q_hat) & (np.arange(scores_all.shape[1]) <= k))[0]
#     set_preds.append(valid_indices)

      #return set_preds
def uncorrect_truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
        q_hats = np.where(Bs, q_as, q_bs)
        #q_hat = max(q_hat)
        set_preds.append(np.where(scores_all[i,:] <= q_hats)[0])
    return set_preds

# def truncated_get_conformal_quantile(scores, alphas, default_qhat=np.inf, exact_coverage=False):
#     '''
#     Compute finite-sample-adjusted 1-alpha quantile of scores
    
#     Inputs:
#         - scores: num_instances-length array of conformal scores for true class. A higher score 
#             indicates more uncertainty
#         - alphas: array of floats between 0 and 1 specifying coverage levels
#         - default_qhat: the value that will be returned if there are insufficient samples to compute
#         the quantile. Should be np.inf if you want a coverage guarantee.
#         - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
#         such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
#         coverage
    
#     '''

#     n = len(scores)

#     if n == 0:
#         print(f'Using default q_hat of {default_qhat} because n={n}')
#         return [default_qhat] * len(alphas)

#     qhats = []
    
#     for alpha in alphas:
#         if exact_coverage:
#             q_a, q_b, gamma = truncated_get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf)
#             exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
#             qhats.append(exact_cov_params)
#         else:
#             val = np.ceil((n+1)*(1-alpha))/n
#             if val > 1:
#                 print(f'Using default q_hat of {default_qhat} because n={n}')
#                 qhat = default_qhat
#             else:
#                 qhat = np.quantile(scores, val, method='inverted_cdf')
#             qhats.append(qhat)

#     return qhats

def truncated_random_split(X, y, r, avg_num_per_class, seed=0):
    '''
    Randomly splits X and y into X1, y1, X2, y2. X1 and y1 will have of
    avg_num_per_class * num_classes examples. The remaining examples
    will be in X2 and y2.
    
    Inputs:
    - X: numpy array. axis 0 must be the same length as y
    - y: 1-D numpy array of class labels (classes should be 0-indexed)
    - avg_num_per_class: Average number of examples per class to include 
    in the data split. 
    - seed: (int) random seed for reproducibility
    
    Output: X1, y1, X2, y2
    '''

    np.random.seed(seed)
    
    num_classes = np.max(y) + 1
    
    num_samples = avg_num_per_class * num_classes
    
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False) # Numeric index
    idx2 = ~np.isin(np.arange(len(y)), idx1) # Boolean index
    X1, y1, r1 = X[idx1], y[idx1], r[idx1]
    X2, y2, r2 = X[idx2], y[idx2], r[idx2]
    
    return X1, y1, r1, X2, y2, r2

# Used for creating balanced or stratified calibration dataset
def truncated_split_X_and_y(X, y, r, n_k, num_classes, seed=0, split='balanced'):
    '''
    Randomly generate two subsets of features X and corresponding labels y such that the
    first subset contains n_k instances of each class k and the second subset contains all
    other instances 
    
    Inputs:
        X: n x d array (e.g., matrix of softmax vectors)
        y: n x 1 array
        n_k: positive int or n x 1 array
        num_classes: total number of classes, corresponding to max(y)
        seed: random seed
        
    Output:
        X1, y1
        X2, y2
    '''
    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
        
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    r1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape)

    i = 0
    for k in range(num_classes):

        # Randomly select n instances of class k
        idx = np.argwhere(y==k).flatten()
        selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))

        X1[i:i+n_k[k]] = X[selected_idx]
        y1[i:i+n_k[k]] = k
        r1[i:i+n_k[k]] = r[selected_idx]
        i += n_k[k]
        
        all_selected_indices[selected_idx] = 1
        
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    r2 = r[all_selected_indices == 0]
    
    return X1, y1, r1, X2, y2, r2

#============================================================
# Partically Truncated class-wise conformal prediction method
#============================================================

def Uncorrect_partially_truncated_compute_class_specific_qhats(cal_scores_all, cal_true_labels, totalcal_ranks, suit_indices, alphas, label_index, num_classes, 
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all:
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed).
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat.
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhats = truncated_compute_qhat(cal_scores_all, totalcal_ranks, suit_indices, alphas, num_classes, default_qhat=default_qhat)
        q_as, q_bs, gammas = truncated_compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, totalcal_ranks,
                                                                       suit_indices, alphas, num_classes, 
                                                                       default_qhat=default_qhats, null_params=null_qhat)
        exact_cov_params = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhats = truncated_compute_qhat(cal_scores_all, cal_true_labels, totalcal_ranks, suit_indices, alphas, num_classes)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class
            class_label = label_index[k]
            idx = (cal_true_labels == class_label) & (totalcal_ranks < suit_indices[k])
            scores = cal_scores_all[idx]

            class_cts[k] = scores.shape[0]

            q_hats[k] = get_conformal_quantile(scores, alphas[k], default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))

        return q_hats

def Uncorrect_partially_truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, val_scores_all, val_labels, alpha, class_gap, truncated_gap, topk, 
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    num = len(totalcal_labels)/num_classes
    cc_alpha = alpha - (class_gap/np.sqrt(num))
    
    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=cc_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    # Assuming classwise_qhats is a numpy array or can be converted into one
    if not isinstance(classwise_qhats, np.ndarray):
        classwise_qhats = np.array(classwise_qhats)
        
    # Get indices of (1-top-k) minimum values
    # topk_indices = classwise_qhats.argsort()[-topk:][::-1]
    bottomk = num_classes - topk
    sorted_indices = classwise_qhats.argsort()
    bottomk_indices = sorted_indices[:bottomk]
    bottomk_indices = np.sort(bottomk_indices)
    print(f'The uncertain class index: {bottomk_indices}')
    # Get the top-k maximum values
    bottomk_values = classwise_qhats[bottomk_indices]
    print(f'The uncertain class quantile: {bottomk_values}')

    idx = np.zeros(len(totalcal_labels), dtype=bool)
    for k in bottomk_indices:
        idx |= (totalcal_labels == k)

    selected_cal_scores = totalcal_scores_all[idx]
    selected_cal_labels = totalcal_labels[idx]
    selected_cal_ranks = totalcal_ranks[idx]
    selected_val_scores = val_scores_all[:, bottomk_indices]
    selected_val_labels = val_labels[idx]

    selected_acc_matrix = accuracy_matrix[:, bottomk_indices]
    #print(selected_acc_matrix)
    selected_num_class = len(bottomk_indices)

    err_matrix = 1 - selected_acc_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    # num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num))
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    #suit_k = [np.where(err_matrix[:, col] < cc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < cc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    #print(suit_k)
    #print(k_max)
    mix_paras = np.linspace(0, 1, rank_all - k_max)
    #print(mix_paras)
    best_para = 1

    for mix_para in mix_paras:
        test_indices = [math.ceil((1 - mix_para) * suit_k[i] + num_classes * mix_para) for i in range(selected_num_class)]
        #print(test_indices)
        test_err = [err_matrix[test_indices[i]-1, i] for i in range(selected_num_class)]
        #print(test_err)
        test_alpha = [tc_alpha - err for err in test_err]
        #print(test_alpha)
    # for k in range(k_max-1, num_classes):
    #     suit_err = err_matrix[k,:]
    #     test_alpha = np.array(alpha - suit_err)
        #print(test_alpha)
        test_qhats = partially_truncated_compute_class_specific_qhats(selected_cal_scores, selected_cal_labels, selected_cal_ranks, 
                                                   suit_indices = test_indices, alphas=test_alpha, 
                                                   label_index = bottomk_indices, num_classes=selected_num_class,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
        #print(classwise_qhats)
        test_preds = partially_truncated_create_classwise_prediction_sets(selected_val_scores, test_qhats, exact_coverage=exact_coverage)
        curr_set_sizes = [len(x) for x in test_preds]
        avg_size = np.mean(curr_set_sizes)
        if avg_size < smallest_ps:
            smallest_ps = avg_size
            best_para = mix_para
    
    print(f'best_para = {best_para}')
    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(selected_num_class)]
    #print(best_indices)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(selected_num_class)]
    tilde_alpha = [tc_alpha - err for err in best_err]
    #print(tilde_alpha)
    bottomk_qhats = partially_truncated_compute_class_specific_qhats(selected_cal_scores, selected_cal_labels, selected_cal_ranks,
                                                   suit_indices = best_indices, alphas=tilde_alpha, 
                                                   label_index = bottomk_indices, num_classes=selected_num_class,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)

    # topk_qhats, _, _, _, _ = truncated_classwise_conformal(selected_acc_matrix, selected_cal_scores, selected_cal_labels, selected_cal_ranks, selected_val_scores, selected_val_labels, alpha, truncated_gap,
                         # selected_num_class, default_qhat=default_qhat, regularize=regularize, exact_coverage=exact_coverage)
    print(f'The truncated class quantile: {bottomk_qhats}')

    classwise_qhats[bottomk_indices] = bottomk_qhats
    
    classwise_preds = truncated_create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
    
    class_coverage_metrics, coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, classwise_preds, class_coverage_metrics, coverage_metrics, set_size_metrics


def Uncorrect_partially_truncated_create_classwise_prediction_sets(scores_all, q_hats, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all, k)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            #q_hat = max(q_hats)
            # relevant_scores = scores_all[i, topk_indices]
            # set_preds.append(np.where(relevant_scores <= q_hats)[0])
            set_preds.append(np.where(scores_all[i,:] <= q_hats)[0])

    return set_preds

#============================================
#   Truncated class-wise conformal prediction
#============================================

# # Create classwise prediction sets
# def truncated_create_classwise_prediction_sets(scores_all, q_hats, suit_indices, rank, exact_coverage=False):
#     '''
#     Inputs:
#         - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
#         - q_hats: as output by compute_class_specific_quantiles
#         - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
#     '''
#     if exact_coverage:
#         assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
#         'you must pass in q_hats computed with exact_coverage=True')
        
#         q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
#         set_preds = truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all, suit_indices)
    
#     else:
#         scores_all = np.array(scores_all)
#         set_preds = []
#         num_samples = len(scores_all)
#         for i in range(num_samples):

#             # Step 1: Get the ranks of the scores in descending order
#             sorted_indices = np.argsort(scores_all[i, :])[::-1]
#             ranks = np.empty_like(sorted_indices)
#             ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

#             # Step 2 & 3: Select indices based on the rank, suit_indices comparison and q_hats condition
#             selected_indices = [j for j, rank in enumerate(ranks) if rank <= suit_indices[j] and scores_all[i, j] <= q_hats[j]]
        
#             set_preds.append(selected_indices)
#             # set_preds.append(np.where(scores_all[i,:] <= q_hats)[0])

#     return set_preds

def truncated_create_classwise_prediction_sets(scores_all, q_hats, suit_indices, rank, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - rank: precomputed ranks of the scores for each instance in the format output by compute_val_class_rank
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all, suit_indices, rank)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):

            # Use the computed ranks
            ranks = rank[i]

            # Select indices based on the rank, suit_indices comparison and q_hats condition
            selected_indices = [j for j, rank_value in enumerate(ranks) if rank_value <= suit_indices[j] and scores_all[i, j] <= q_hats[j]]
        
            set_preds.append(selected_indices)

    return set_preds


def truncated_compute_class_specific_qhats(cal_scores_all, cal_true_labels, alphas, num_classes,
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all:
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed).
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat.
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhats = truncated_compute_qhat(cal_scores_all, alphas, num_classes, default_qhat=default_qhat)
        q_as, q_bs, gammas = truncated_compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, 
                                                                       alphas, num_classes, 
                                                                       default_qhat=default_qhats, null_params=null_qhat)
        exact_cov_params = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhats = truncated_compute_qhat(cal_scores_all, cal_true_labels, alphas, num_classes)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class
            idx = (cal_true_labels == k) 
            scores = cal_scores_all[idx]

            class_cts[k] = scores.shape[0]

            q_hats[k] = get_conformal_quantile(scores, alphas[k], default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))

        return q_hats

def truncated_compute_qhat(scores_all, true_labels, alphas, num_classes, exact_coverage=False, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores, or num_instances-length array of 
        conformal scores for true class. A higher score indicates more uncertainty
        - true_labels: num_instances length array of ground truth labels
        - alpha: float between 0 and 1 specifying coverage level
        - plot_scores: If True, plot histogram of true class scores 
    '''
    # If necessary, select scores that correspond to correct label
    if len(scores_all.shape) == 2:
        scores = np.squeeze(np.take_along_axis(scores_all, np.expand_dims(true_labels, axis=1), axis=1))
    else:
        scores = scores_all

    q_hats = np.zeros((num_classes))   
    for k in range(num_classes):
        idx = (cal_true_labels == k) 
        rank_scores = scores_all[idx]
        q_hats[k] = get_conformal_quantile(rank_scores, alphas[k], exact_coverage=exact_coverage)
        #q_hats.append(q_hat)
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hats

def truncated_compute_exact_coverage_class_specific_params(scores_all, labels, num_classes, alphas, 
                                 default_qhat=np.inf, null_params=None):
    '''
    Compute the quantities necessary for performing classwise conformal prediction with exact coverage
    
    Inputs:
        - scores_all: (num_instances, num_classes) array where scores_all[i,j] = score of class j for instance i
        - labels: (num_instances,) array of true class labels
        - num_classes: Number of classes
        - alpha: number between 0 and 1 specifying desired coverage level
        - default_qhat: Quantile that will be used when there is insufficient data to compute a quantile
        - null_params: Dict of {'q_a': ..., 'q_b': ...., 'gamma', ...} to be assigned to
               class -1 (will be appended to end of param lists). Not needed if -1 does not appear in labels
    '''
    
    q_as = np.zeros((num_classes,))   
    q_bs = np.zeros((num_classes,)) 
    gammas = np.zeros((num_classes,)) 
    for k in range(num_classes):
        # Only select data for which k is true class
        
        idx = (labels == k) 
        if len(scores_all.shape) == 2:
            scores = scores_all[idx, k]
        else:
            scores = scores_all[idx]
        
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha[k], default_qhat=default_qhat)
        q_as[k] = q_a
        q_bs[k] = q_b
        gammas[k] = gamma
        
    if -1 in labels:
        q_as = np.concatenate((q_as, [null_params['q_a']]))
        q_bs = np.concatenate((q_bs, [null_params['q_b']]))
        gamma = np.concatenate((gammas, [null_params['gamma']]))
        
    return q_as, q_bs, gammas

# # Classwise conformal pipeline
# def truncated_classwise_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
#                          num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
#     '''
#     Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
#     If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
#     unrandomized conservative sets. 
#     Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
#     See compute_class_specific_qhats() docstring for more details about expected inputs.
#     '''
    
#     classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels,  
#                                                    alpha=alpha, 
#                                                    num_classes=num_classes,
#                                                    default_qhat=default_qhat, regularize=regularize,
#                                                    exact_coverage=exact_coverage)
#     classwise_preds = truncated_create_classwise_prediction_sets(val_scores_all, k, classwise_qhats, exact_coverage=exact_coverage)

#     class_coverage_metrics, coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
#     return classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics

def truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, val_scores_all, val_labels, val_rank, alpha, truncated_gap,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    err_matrix = 1 - accuracy_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num)) 
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    #print(suit_k)
    #print(k_max)
    mix_paras = np.linspace(0, 1, rank_all-k_max)
    #print(mix_paras)
    best_para = 1

    for mix_para in mix_paras:
        test_indices = [math.ceil((1 - mix_para) * suit_k[i] + num_classes * mix_para) for i in range(num_classes)]
        #print(test_indices)
        test_err = [err_matrix[test_indices[i]-1, i] for i in range(num_classes)]
        #print(test_err)
        test_alpha = [tc_alpha - err for err in test_err]
        #print(test_alpha)
    # for k in range(k_max-1, num_classes):
    #     suit_err = err_matrix[k,:]
    #     test_alpha = np.array(alpha - suit_err)
        #print(test_alpha)
        classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=test_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
        #print(classwise_qhats)
        classwise_preds = truncated2_create_classwise_prediction_sets(val_scores_all, classwise_qhats, suit_indices = test_indices, rank = val_rank, exact_coverage=exact_coverage)
        curr_set_sizes = [len(x) for x in classwise_preds]
        avg_size = np.mean(curr_set_sizes)
        #print(avg_size)
        if avg_size < smallest_ps:
            smallest_ps = avg_size
            best_para = mix_para
    
    print(f'best_para = {best_para}')
    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(num_classes)]
    #print(best_indices)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(num_classes)]
    tilde_alpha = [tc_alpha - err for err in best_err]
    #print(tilde_alpha)
    classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=tilde_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    classwise_preds = truncated2_create_classwise_prediction_sets(val_scores_all, classwise_qhats, suit_indices = best_indices, rank = val_rank, exact_coverage=exact_coverage)
    
    class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics


def truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all, suit_indices, rank):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
        q_hats = np.where(Bs, q_as, q_bs)
        #q_hat = max(q_hat)
            
        # Use the computed ranks
        ranks = rank[i]

        # Select indices based on the rank, suit_indices comparison and q_hats condition
        selected_indices = [j for j, rank_value in enumerate(ranks) if rank_value <= suit_indices[j] and scores_all[i, j] <= q_hats[j]]
        
        set_preds.append(selected_indices)

    return set_preds

def truncated2_create_classwise_prediction_sets(scores_all, q_hats, suit_indices, rank, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - rank: precomputed ranks of the scores for each instance
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):

            # Get the classes based on q_hats condition
            prediction_classes = np.where(scores_all[i,:] <= q_hats)[0]

            # Fetch the precomputed ranks for this sample
            ranks_for_i = rank[i]

            # Extract the ranks corresponding to the prediction_classes
            prediction_ranks = [ranks_for_i[class_idx] for class_idx in prediction_classes]

            # Filter the prediction_classes based on rank condition
            final_prediction_classes = [class_idx for idx, class_idx in enumerate(prediction_classes) 
                                        if prediction_ranks[idx] <= suit_indices[class_idx]]

            length_gap = len(prediction_classes) - len(final_prediction_classes)
            # if i % 100 == 0:
            #     print(f"For index {i}, Prediction Ranks are:", prediction_ranks)
            #     print(f"For index {i}, Length Gap between original and filtered predictions:", length_gap)

            set_preds.append(final_prediction_classes)

    return set_preds


def compute_val_score_rank(scores):

    scores_all = np.array(scores)
    score_rank = []
    num_samples = len(scores_all)
    for i in range(num_samples):

        sorted_indices = np.argsort(scores_all[i, :])[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        score_rank.append(ranks)

    return score_rank

def truncated_classwise_conformal_test(accuracy_matrix, totalcal_scores_all, totalcal_labels, true_val_ranks, val_scores_all, val_labels, val_rank, alpha, class_gap, truncated_gap,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    err_matrix = 1 - accuracy_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num)) 
    cc_alpha = alpha - (class_gap/np.sqrt(num)) 
    # print(tc_alpha)
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    # prob_class = [1 - err_matrix[suit_k[col], col] for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    # print(prob_class)
    #print(k_max)
    mix_paras = np.linspace(0, 1, int(rank_all-k_max))
    #print(mix_paras)
    best_para = 1

    for mix_para in mix_paras:
        test_indices = [math.ceil((1 - mix_para) * suit_k[i] + num_classes * mix_para) for i in range(num_classes)]
        test_err = [err_matrix[test_indices[i]-1, i] for i in range(num_classes)]
        test_alpha = [tc_alpha - err for err in test_err]
        classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=test_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
        #print(classwise_qhats)
        classwise_preds = truncated2_create_classwise_prediction_sets(val_scores_all, classwise_qhats, suit_indices = test_indices, rank = val_rank, exact_coverage=exact_coverage)
        curr_set_sizes = [len(x) for x in classwise_preds]
        avg_size = np.mean(curr_set_sizes)
        #print(avg_size)
        if avg_size < smallest_ps:
            smallest_ps = avg_size
            best_para = mix_para
    
    print(f'best_para = {best_para}')
    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(num_classes)]
    # print(best_indices)
    #prob_class = [1 - err_matrix[best_indices[col], col] for col in range(err_matrix.shape[1])]
    #print(prob_class)
    # rank_prob = compute_rank_prob(num_classes, ranks = val_rank, suit_indices = best_indices)
    # print(rank_prob)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(num_classes)]
    tilde_alpha = [tc_alpha - err for err in best_err]
    #print(tilde_alpha)
    classwise_qhats_K = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=tilde_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    # classwise_preds_C = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
    
    # CCP_tilde_prob = compute_prob(classwise_preds_C, num_classes, val_scores_all)

    classwise_preds_K = truncated_create_classwise_prediction_sets_test(val_scores_all, val_labels, num_classes, classwise_qhats, suit_indices = best_indices, rank = val_rank, exact_coverage=exact_coverage)
    
    KCCP_prob = compute_prob(classwise_preds_K, num_classes, val_scores_all)

    # class_prob = [i / j if j != 0 else 0 for i, j in zip(KCCP_prob, CCP_tilde_prob)]
    # print(f'Class_ratio  = {class_prob}')

    # beta = [ p / q if q != 0 else 0 for p, q in zip(rank_prob, class_prob)]
    # print(f'Beta  = {beta}')s
    
    classwise_qhats_C = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=cc_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    classwise_preds_C = create_classwise_prediction_sets(val_scores_all, classwise_qhats_C, exact_coverage=exact_coverage)

    CCP_prob = compute_prob(classwise_preds_C, num_classes, val_scores_all)    
    sigma = [ p / q if q != 0 else 0 for p, q in zip(KCCP_prob, CCP_prob)]
    # print(f'Sigma  = {sigma}')

    results = {'Class Ratio': sigma}

    return results


def truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all, suit_indices, rank):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
        q_hats = np.where(Bs, q_as, q_bs)
        #q_hat = max(q_hat)
            
        # Use the computed ranks
        ranks = rank[i]

        # Select indices based on the rank, suit_indices comparison and q_hats condition
        selected_indices = [j for j, rank_value in enumerate(ranks) if rank_value <= suit_indices[j] and scores_all[i, j] <= q_hats[j]]
        
        set_preds.append(selected_indices)

    return set_preds

def truncated_create_classwise_prediction_sets_test(scores_all, val_labels, num_classes, q_hats, suit_indices, rank, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - rank: precomputed ranks of the scores for each instance
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = truncated_construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        # prob_list = []
        num_samples = len(scores_all)
        for i in range(num_samples):

            # Get the classes based on q_hats condition
            prediction_classes = np.where(scores_all[i,:] <= q_hats)[0]

            # Fetch the precomputed ranks for this sample.
            ranks_for_i = rank[i]

            # Extract the ranks corresponding to the prediction_classes
            prediction_ranks = [ranks_for_i[class_idx] for class_idx in prediction_classes]

            # Filter the prediction_classes based on rank condition
            final_prediction_classes = [class_idx for idx, class_idx in enumerate(prediction_classes) 
                                        if prediction_ranks[idx] <= suit_indices[class_idx]]

            # length_gap = len(prediction_classes) - len(final_prediction_classes)
            # prob = len(final_prediction_classes) / len(prediction_classes)
            # prob_list.append(prob)
            # if i % 100 == 0:
            #     print(f"For index {i}, Prediction Ranks are:", prediction_ranks)
            #     print(f"For index {i}, Length Gap between original and filtered predictions:", length_gap)

            set_preds.append(final_prediction_classes)
        
        # prob_array = np.array(prob_list)
        # class_prob = np.zeros((num_classes,)) 
        # for k in range(num_classes):

        #     # Only select data for which k is true class
        #     idx = (val_labels == k)
        #     probs = prob_array[idx]

        #     class_prob[k] = np.mean(probs)

        # print(class_prob)
    return set_preds


def compute_prob(set_preds, num_classes, scores_all):

    class_prob_cov = np.zeros((num_classes,))
    num_samples = len(scores_all)
    
    for k in range(num_classes):
        
        class_num = np.sum([1 if k in pred_set else 0 for pred_set in set_preds])
        #print(num_correct)
        class_prob_cov[k] = class_num / num_samples
    
    class_prob = np.array(class_prob_cov)
    return class_prob

def compute_rank_prob(num_classes, ranks, suit_indices):

    class_rank_prob_cov = np.zeros((num_classes,))
    num_samples = len(ranks)
    print(num_samples)
    ranks = np.array(ranks)

    for k in range(num_classes):
        
        class_rank_num = np.sum([1 if ranks[i, k] <= suit_indices[k] else 0 for i in range(num_samples)])
        #print(num_correct)
        class_rank_prob_cov[k] = class_rank_num / num_samples
    
    class_rank_prob = np.array(class_rank_prob_cov)
    return class_rank_prob


def truncated_classwise_conformal_test2(accuracy_matrix, totalcal_scores_all, totalcal_labels, true_cal_ranks, cal_rank, alpha, class_gap, truncated_gap,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    err_matrix = 1 - accuracy_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num)) 
    cc_alpha = alpha - (class_gap/np.sqrt(num)) 
    # print(tc_alpha)
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    # prob_class = [1 - err_matrix[suit_k[col], col] for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    # print(prob_class)
    #print(k_max)
    mix_paras = np.linspace(0, 1, int(rank_all-k_max))
    #print(mix_paras)
    best_para = 1

    for mix_para in mix_paras:
        test_indices = [math.ceil((1 - mix_para) * suit_k[i] + num_classes * mix_para) for i in range(num_classes)]
        test_err = [err_matrix[test_indices[i]-1, i] for i in range(num_classes)]
        test_alpha = [tc_alpha - err for err in test_err]
        classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=test_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
        #print(classwise_qhats)
        classwise_preds = truncated2_create_classwise_prediction_sets(totalcal_scores_all, classwise_qhats, suit_indices = test_indices, rank = cal_rank, exact_coverage=exact_coverage)
        curr_set_sizes = [len(x) for x in classwise_preds]
        avg_size = np.mean(curr_set_sizes)
        #print(avg_size)
        if avg_size < smallest_ps:
            smallest_ps = avg_size
            best_para = mix_para
    
    print(f'best_para = {best_para}')
    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(num_classes)]
    # print(best_indices)
    #prob_class = [1 - err_matrix[best_indices[col], col] for col in range(err_matrix.shape[1])]
    #print(prob_class)
    # rank_prob = compute_rank_prob(num_classes, ranks = val_rank, suit_indices = best_indices)
    # print(rank_prob)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(num_classes)]
    tilde_alpha = [tc_alpha - err for err in best_err]
    #print(tilde_alpha)
    classwise_qhats_K = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=tilde_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    # classwise_preds_C = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
    
    # CCP_tilde_prob = compute_prob(classwise_preds_C, num_classes, val_scores_all)

    classwise_preds_K = truncated_create_classwise_prediction_sets_test(totalcal_scores_all, totalcal_labels, num_classes, classwise_qhats, suit_indices = best_indices, rank = cal_rank, exact_coverage=exact_coverage)
    
    KCCP_prob = compute_prob(classwise_preds_K, num_classes, totalcal_scores_all)

    # class_prob = [i / j if j != 0 else 0 for i, j in zip(KCCP_prob, CCP_tilde_prob)]
    # print(f'Class_ratio  = {class_prob}')

    # beta = [ p / q if q != 0 else 0 for p, q in zip(rank_prob, class_prob)]
    # print(f'Beta  = {beta}')s
    
    classwise_qhats_C = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=cc_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    classwise_preds_C = create_classwise_prediction_sets(totalcal_scores_all, classwise_qhats_C, exact_coverage=exact_coverage)

    CCP_prob = compute_prob(classwise_preds_C, num_classes, totalcal_scores_all)    
    sigma = [ p / q if q != 0 else 0 for p, q in zip(KCCP_prob, CCP_prob)]
    # print(f'Sigma  = {sigma}')

    results = {'Class Ratio': sigma}

    return results

def truncated_classwise_conformal_qc(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, val_scores_all, val_labels, val_rank, alpha, truncated_gap,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    err_matrix = 1 - accuracy_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num)) 
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    #print(suit_k)
    #print(k_max)
    mix_paras = np.linspace(0, 1, rank_all-k_max)
    #print(mix_paras)
    best_para = 0

    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(num_classes)]
    #print(best_indices)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(num_classes)]
    tilde_alpha = [tc_alpha - 0 for err in best_err]
    #print(tilde_alpha)
    classwise_qhats = truncated_compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alphas=tilde_alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    classwise_preds = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)
    # classwise_preds = truncated2_create_classwise_prediction_sets(val_scores_all, classwise_qhats, suit_indices = best_indices, rank = val_rank, exact_coverage=exact_coverage)
    
    class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics


def truncated_classwise_conformal_rc(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, val_scores_all, val_labels, val_rank, alpha, truncated_gap,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    err_matrix = 1 - accuracy_matrix
    err_matrix = np.array(err_matrix)
    #print(err_matrix)
    #print(err_matrix.shape)
    smallest_ps = num_classes
    num = len(totalcal_labels)/err_matrix.shape[1]
    rank_all = err_matrix.shape[0]
    tc_alpha = alpha - (truncated_gap/np.sqrt(num)) 
    suit_k = [np.where(err_matrix[:, col] < tc_alpha)[0][0] + 1 if len(np.where(err_matrix[:, col] < tc_alpha)[0]) > 0 else num_classes for col in range(err_matrix.shape[1])]
    k_max = max(suit_k)
    #print(suit_k)
    #print(k_max)
    mix_paras = np.linspace(0, 1, rank_all-k_max)
    #print(mix_paras)
    best_para = 0

    best_indices = [math.ceil((1 - best_para) * suit_k[i] + num_classes * best_para) for i in range(num_classes)]
    #print(best_indices)
    best_err = [err_matrix[best_indices[i]-1, i] for i in range(num_classes)]
    tilde_alpha = [tc_alpha - err for err in best_err]
    #print(tilde_alpha)

    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha = 0, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    #print(classwise_qhats)
    classwise_preds = truncated2_create_classwise_prediction_sets(val_scores_all, classwise_qhats, suit_indices = best_indices, rank = val_rank, exact_coverage=exact_coverage)
    
    class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, class_coverage_metrics, coverage_metrics, set_size_metrics, tail_metrics


