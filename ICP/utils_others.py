import torch
import numpy as np
import gc
import pandas as pd
from torch.nn.functional import softmax
from scipy.stats import rankdata
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from tqdm import tqdm
from typing import List
from numpy.random import default_rng
import os
import random
import pickle

def get_logits_targets(model, loader):
    #logits = torch.zeros((len(loader.dataset), 10)) #1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            y_preds = torch.argmax(model(x.cuda()), dim = 1)
            #logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = y_preds.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    #dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return labels.long()


def FormData(args, loader):
    all_data = [[], []]
    all_labels = [[], []]

    for x, target in tqdm(loader):
        #print(f"target = {target}")
        if target != 9:
            all_data[0].append(x)
            all_labels[0].append(target.tolist())
        else:
            all_data[1].append(x)
            all_labels[1].append(target.tolist())
    return all_data, all_labels

def Extract(args, train_loader, SubFol = '/Train'):
    SaveDataPath = args.RootPath + '/AllXY' + SubFol
    if not os.path.exists(SaveDataPath):
        print(f" Data does not exists")
        AllX, AllY = FormData(args, train_loader)
        os.makedirs(SaveDataPath)
        with open(SaveDataPath + '/data.pickle', 'wb') as f:
            pickle.dump([AllX, AllY], f)
    else:
        print(f"Data exists")
        with open(SaveDataPath + '/data.pickle', 'rb') as f:
            AllX, AllY = pickle.load(f)
    return AllX, AllY

# function to calculate top-k accuracy of the model
def calculate_top_k_accuracy(model, dataloader, device, k):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Get the top k indices of the scores (in descending order)
            top_k_values, top_k_indices = torch.topk(outputs, k, dim=1)
            
            # Check if the labels are in the top_k_indices
            for i in range(labels.size(0)):
                if labels[i] in top_k_indices[i]:
                    total_correct += 1
            
            total_images += labels.size(0)

    model_accuracy = total_correct / total_images
    return model_accuracy


def Smooth_Adv(model, x, y, noises, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # number of permutations to estimate mean
    num_of_noise_vecs = noises.size()[0] // n

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    #print("Generating Adverserial Examples:")

    for j in tqdm(range(num_of_batches)):
        #GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()


    # return adversarial examples
    return x_adv


def evaluate_predictions(S, X, y, conditional=False, coverage_on_label=False, num_of_classes=10):
    #print(f"num_of_classes = {num_of_classes}")
    #exit(1)
    # get numbers of points
    #n = np.shape(X)[0]

    # get points to a matrix of the format nxp
    #X = np.vstack([X[i, 0, :, :].flatten() for i in range(n)])

    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    #print([y[i] in S[i] for i in range(len(y))][:20])
    #print(S[:10])
    #print(y[:10])
    #print(f"marg_coverage = {marg_coverage}")
    #exit(1)
    # If desired calculate coverage for each class
    if coverage_on_label:
        sums = np.zeros(num_of_classes)
        size_sums = np.zeros(num_of_classes)
        lengths = np.zeros(num_of_classes)
        for i in range(len(y)):
            lengths[y[i]] = lengths[y[i]] + 1
            size_sums[y[i]] = size_sums[y[i]] + len(S[i])
            if y[i] in S[i]:
                sums[y[i]] = sums[y[i]] + 1
        coverage_given_y = sums/lengths
        lengths_given_y = size_sums/lengths

    # Conditional coverage not implemented
    wsc_coverage = None

    # Size and size conditional on coverage
    size = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    size_cover = np.mean([len(S[i]) for i in idx_cover])

    # Combine results
    outs = pd.DataFrame()

    out_overall = pd.DataFrame({'Marginal Coverage': [marg_coverage], 'Prediction Size': [size]})

    cvg_neg, size_neg = 0, 0
    if coverage_on_label:
        for i in range(num_of_classes):

            out = pd.DataFrame({'Coverage': [coverage_given_y[i]], 'Prediction Size': [lengths_given_y[i]]})
            out['Condition on Class'] = str(i)
            outs = outs.append(out)

            del out

        return outs, out_overall
    else:
        return out_overall


# calculate accuracy of the smoothed classifier
def calculate_accuracy_smooth(model, x, y, noises, num_classes, k=1, device='cpu', GPU_CAPACITY=1024):
    # get size of the test set
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the outputs
    smoothed_predictions = torch.zeros((n, num_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # get predictions over all batches
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first n_smooth samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

        # add noise to points
        noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1)

        # get smoothed prediction for each point
        for m in range(len(labels)):
            smoothed_predictions[(j * batch_size) + m, :] = torch.mean(
                noisy_outputs[(m * n_smooth):((m + 1) * n_smooth)], dim=0)

    # transform results to numpy array
    smoothed_predictions = smoothed_predictions.numpy()

    # get label ranks to calculate top k accuracy
    label_ranks = np.array([rankdata(-smoothed_predictions[i, :], method='ordinal')[y[i]] - 1 for i in range(n)])

    # get probabilities of correct labels
    label_probs = np.array([smoothed_predictions[i, y[i]] for i in range(n)])

    # calculate accuracy
    top_k_accuracy = np.sum(label_ranks <= (k - 1)) / float(n)

    # calculate average inverse probability score
    score = np.mean(1 - label_probs)

    # calculate the 90 qunatiule
    quantile = mquantiles(1-label_probs, prob=0.9)
    return top_k_accuracy, score, quantile


def smooth_calibration(model, x_calib, y_calib, noises, alpha, num_of_classes, scores_list, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # size of the calibration set
    n_calib = x_calib.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n_calib

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n_calib))
    else:
        smoothed_scores = np.zeros((len(scores_list), n_calib))
        scores_smoothed = np.zeros((len(scores_list), n_calib))

    # create container for the calibration thresholds
    thresholds = np.zeros((len(scores_list), 3))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n_calib % batch_size != 0:
        num_of_batches = (n_calib // batch_size) + 1
    else:
        num_of_batches = (n_calib // batch_size)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n_calib, num_of_classes))
    else:
        smooth_outputs = np.zeros((n_calib, num_of_classes))

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n_calib, low=0.0, high=1.0)

    # pass all points to model in batches and calculate scores
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x_calib[(j * batch_size):((j + 1) * batch_size)]
        labels = y_calib[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = noises[(j * batch_size):((j + 1) * batch_size)].to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        # get smoothed score for each point
        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            for k in range(len(labels)):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # get smoothed score of this point

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores
                for p, score_func in enumerate(scores_list):
                    # get smoothed score
                    tmp_scores = score_func(point_outputs, labels[k], u, all_combinations=True)
                    smoothed_scores[p, (j * batch_size) + k] = np.mean(tmp_scores)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :] = score_func(simple_outputs, y_calib, uniform_variables, all_combinations=False)
        else:
            scores_smoothed[p, :] = score_func(smooth_outputs, y_calib, uniform_variables, all_combinations=False)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros((len(scores_list), 2))
    for p in range(len(scores_list)):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(scores_smoothed[p, :], prob=level_adjusted)
            thresholds[p, 2] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)

            # calculate lower and upper bounds of correction of smoothed score
            upper_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)+correction, loc=0, scale=1)
            lower_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)-correction, loc=0, scale=1)

            bounds[p, 0] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= lower_thresh])/np.size(smoothed_scores[p, :])
            bounds[p, 1] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= upper_thresh]) / np.size(smoothed_scores[p, :])

    return thresholds, bounds


def smooth_calibration_ImageNet(model, x_calib, y_calib, n_smooth, sigma_smooth, alpha, num_of_classes, scores_list, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # size of the calibration set
    n_calib = x_calib.size()[0]

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n_calib))
    else:
        smoothed_scores = np.zeros((len(scores_list), n_calib))
        scores_smoothed = np.zeros((len(scores_list), n_calib))

    # create container for the calibration thresholds
    thresholds = np.zeros((len(scores_list), 3))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n_calib % batch_size != 0:
        num_of_batches = (n_calib // batch_size) + 1
    else:
        num_of_batches = (n_calib // batch_size)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n_calib, num_of_classes))
    else:
        smooth_outputs = np.zeros((n_calib, num_of_classes))

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n_calib, low=0.0, high=1.0)

    # pass all points to model in batches and calculate scores
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x_calib[(j * batch_size):((j + 1) * batch_size)]
        labels = y_calib[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = (torch.randn_like(inputs)*sigma_smooth).to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = (torch.randn_like(x_tmp)*sigma_smooth).to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        # get smoothed score for each point
        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            for k in range(len(labels)):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # get smoothed score of this point

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores
                for p, score_func in enumerate(scores_list):
                    # get smoothed score
                    tmp_scores = score_func(point_outputs, labels[k], u, all_combinations=True)
                    smoothed_scores[p, (j * batch_size) + k] = np.mean(tmp_scores)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :] = score_func(simple_outputs, y_calib, uniform_variables, all_combinations=False)
        else:
            scores_smoothed[p, :] = score_func(smooth_outputs, y_calib, uniform_variables, all_combinations=False)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros((len(scores_list), 2))
    for p in range(len(scores_list)):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(scores_smoothed[p, :], prob=level_adjusted)
            thresholds[p, 2] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)

            # calculate lower and upper bounds of correction of smoothed score
            upper_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)+correction, loc=0, scale=1)
            lower_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)-correction, loc=0, scale=1)

            bounds[p, 0] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= lower_thresh])/np.size(smoothed_scores[p, :])
            bounds[p, 1] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= upper_thresh]) / np.size(smoothed_scores[p, :])

    return thresholds, bounds


def predict_sets(model, x, noises, num_of_classes, scores_list, thresholds, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # get number of points
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        scores_smoothed = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))

    for j in range(num_of_batches):
        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = noises[(j * batch_size):((j + 1) * batch_size)].to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed score for each point
            for k in range(inputs.size()[0]):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores with all lables
                for p, score_func in enumerate(scores_list):
                    smoothed_scores[p, ((j * batch_size) + k), :] = np.mean(
                        score_func(point_outputs, np.arange(num_of_classes), u, all_combinations=True), axis=0)

                #return smoothed_scores

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:
            scores_smoothed[p, :, :] = score_func(smooth_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    #return scores_simple

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(len(scores_list)):
        if base:
            S_hat_simple = [np.where(norm.ppf(scores_simple[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 0], loc=0, scale=1))[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            S_hat_smoothed = [np.where(norm.ppf(scores_smoothed[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]

            tmp_list = [S_hat_smoothed, smoothed_S_hat, smoothed_S_hat_corrected]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets


def predict_sets_ImageNet(model, x, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, thresholds, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # get number of points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        scores_smoothed = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))

    image_index = -1
    for j in range(num_of_batches):

        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        if base:
            noises_test_base = torch.empty((curr_batch_size, channels, rows, cols))
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test_base[k:(k + 1)] = torch.randn((1, channels, rows, cols)) * sigma_smooth

            noisy_points = inputs.to(device) + noises_test_base.to(device)
        else:
            noises_test = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test[(k * n_smooth):(k + 1) * n_smooth] = torch.randn(
                    (n_smooth, channels, rows, cols)) * sigma_smooth

            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # add noise to points
            noisy_points = x_tmp + noises_test.to(device)

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed score for each point
            for k in range(inputs.size()[0]):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores with all lables
                for p, score_func in enumerate(scores_list):
                    smoothed_scores[p, ((j * batch_size) + k), :] = np.mean(
                        score_func(point_outputs, np.arange(num_of_classes), u, all_combinations=True), axis=0)
                del u
                gc.collect()

        if base:
            del noisy_points, noisy_outputs, noises_test_base
        else:
            del noisy_points, noisy_outputs, noises_test, tmp
        gc.collect()

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:
            scores_smoothed[p, :, :] = score_func(smooth_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(len(scores_list)):
        if base:
            S_hat_simple = [np.where(norm.ppf(scores_simple[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 0], loc=0, scale=1))[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            S_hat_smoothed = [np.where(norm.ppf(scores_smoothed[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]

            tmp_list = [S_hat_smoothed, smoothed_S_hat, smoothed_S_hat_corrected]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets


def Smooth_Adv_ImageNet(model, x, y, indices, n_smooth, sigma_smooth, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # number of permutations to estimate mean
    num_of_noise_vecs = n_smooth

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    #print("Generating Adverserial Examples:")

    image_index = -1
    for j in tqdm(range(num_of_batches)):
        #GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
        # get relevant noises for this batch
        for k in range(curr_batch_size):
            image_index = image_index + 1
            torch.manual_seed(indices[image_index])
            noise[(k * n_smooth):((k + 1) * n_smooth)] = torch.randn(
                (n_smooth, channels, rows, cols)) * sigma_smooth


        #noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        noise = noise.to(device)
        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del noise, tmp, x_adv_batch
        gc.collect()

    # return adversarial examples
    return x_adv


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


def get_scores(model, x, num_of_classes, scores_list, device='cpu', GPU_CAPACITY=1024):
    #print(f"num_of_classes = {num_of_classes}")
    # get number of points
    n = x.size()[0]
    #print(f"n = {n}")
    #exit(1)
    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]
    n_smooth = 1
    # create container for the scores
    scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    simple_outputs = np.zeros((n, num_of_classes))

    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    for j in tqdm(range(num_of_batches)):
        inputs = x[(j * batch_size):((j + 1) * batch_size)].to(device)
        outputs = model(inputs).to(torch.device('cpu')).detach()
        outputs = softmax(outputs, dim=1).numpy()
        #outputs = np.exp(outputs)
        #print(outputs[0], np.sum(outputs[0]), 's1')
        #print(outputs[1], np.sum(outputs[1]), 's1')
        #print(outputs[2], np.sum(outputs[2]), 's1')
        #print(f"outputs = {outputs.shape}")
        #exit(1)
        simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = outputs

    rng = default_rng(0)
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    # return relevant scores
    return scores_simple

def calibration(scores_simple=None, alpha = 0.1, CCP_alphas=0.0, CCCP_alpha = 0.0, num_of_scores=2, num_class = None, y = None):
    #print(f"alpha = {alpha}, {eff_alphas}")
    allQuantiles_ICP, allQuantiles_CCCP = [], []
    # size of the calibration set
    n_calib = scores_simple.shape[1]
    #print(f"n_calib = {n_calib}, {scores_simple.shape}")
    # create container for the calibration thresholds
    thresholds = np.zeros((num_of_scores, 1))

    # Compute thresholds

    if y is not None:
        for i in range(num_class):
            IndicesI = torch.where(y == i)[0]

            #if i == 5 or i == 9:
            #    level_adjusted_ICP = (1.0 - CCP_alphas[i]+0.08) * (1.0 + 1.0 / float(len(IndicesI)))
            #else:
            #    level_adjusted_ICP = (1.0 - CCP_alphas[i]) * (1.0 + 1.0 / float(len(IndicesI)))

            #print(len(IndicesI), 's222')
            #exit(1)
            level_adjusted_ICP = (1.0 - CCP_alphas[i]) * (1.0 + 1.0 / float(len(IndicesI)))

            level_adjusted_CCCP = (1.0 - CCCP_alpha) * (1.0 + 1.0 / float(len(IndicesI)))

            scoreI_ICP = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted_ICP, axis = 1)
            scoreI_CCCP = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted_CCCP, axis = 1)

            allQuantiles_ICP.append(torch.from_numpy(scoreI_ICP.data))
            allQuantiles_CCCP.append(torch.from_numpy(scoreI_CCCP.data))

    #print(f"allQuantiles = {torch.cat(allQuantiles, dim = 1)[0]}")
    #ClssThresholdsICP = torch.max(torch.cat(allQuantiles_ICP, dim = 1), dim = 1).values
    ClssThresholdsCCCP = torch.cat(allQuantiles_CCCP, dim = 1).numpy()
    ClssThresholdsICP = torch.cat(allQuantiles_ICP, dim = 1).numpy()


    Q_Values1 = pd.DataFrame()
    Q_Values2 = pd.DataFrame()

    Qs = torch.cat(allQuantiles_CCCP, dim = 1)[0]

    for i in range(num_class):
        Q_Values1['Quantile'] = [round(Qs[i].item(), 3)]
        Q_Values1['Names'] = f'Class {i}'
        Q_Values2 = Q_Values2.append(Q_Values1)
        #print(Qs[i])

    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))

    #print(f"h1 = {mquantiles(scores_simple[0, :], prob=level_adjusted)[0]}")
    Q_Values1['Quantile'] = [round(mquantiles(scores_simple[0, :], prob=level_adjusted)[0], 3)]
    Q_Values1['Names'] = 'Vanilla CP'
    Q_Values2 = Q_Values2.append(Q_Values1)
    #print(f"Q = {Q_Values2} {Q_Values1}")
    #exit(1)

    for p in range(num_of_scores):
        thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        #thresholds[p, 1] = ClssThresholdsCCCP[p]

        #thresholds[p, 2] = ClssThresholdsICP[p]

    #print(f"thresholds1 = {thresholds}")
    #exit(1)
    return thresholds, ClssThresholdsCCCP, ClssThresholdsICP


def calibration1(scores_simple=None, alpha = 0.1, CCP_alphas=0.0, CCCP_alpha = 0.0, num_of_scores=2, num_class = None, y = None):
    #print(f"alpha = {alpha}, {eff_alphas}")
    allQuantiles_ICP, allQuantiles_CCCP = [], []
    # size of the calibration set
    n_calib = scores_simple.shape[1]
    #print(f"n_calib = {n_calib}, {scores_simple.shape}")
    # create container for the calibration thresholds
    thresholds = np.zeros((num_of_scores, 1))

    # Compute thresholds

    if y is not None:
        for i in range(num_class):
            IndicesI = torch.where(y == i)[0]

            #if i == 5 or i == 9:
            #    level_adjusted_ICP = (1.0 - CCP_alphas[i]+0.08) * (1.0 + 1.0 / float(len(IndicesI)))
            #else:
            #    level_adjusted_ICP = (1.0 - CCP_alphas[i]) * (1.0 + 1.0 / float(len(IndicesI)))

            #print(len(IndicesI), 's222')
            #exit(1)
            level_adjusted_ICP = (1.0 - CCP_alphas[i]+0.02) * (1.0 + 1.0 / float(len(IndicesI)))

            level_adjusted_CCCP = (1.0 - CCCP_alpha+0.017) * (1.0 + 1.0 / float(len(IndicesI)))

            scoreI_ICP = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted_ICP, axis = 1)
            scoreI_CCCP = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted_CCCP, axis = 1)

            allQuantiles_ICP.append(torch.from_numpy(scoreI_ICP.data))
            allQuantiles_CCCP.append(torch.from_numpy(scoreI_CCCP.data))

    #print(f"allQuantiles = {torch.cat(allQuantiles, dim = 1)[0]}")
    #ClssThresholdsICP = torch.max(torch.cat(allQuantiles_ICP, dim = 1), dim = 1).values
    ClssThresholdsCCCP = torch.max(torch.cat(allQuantiles_CCCP, dim = 1), dim = 1).values
    ClssThresholdsICP = torch.cat(allQuantiles_ICP, dim = 1).numpy()


    Q_Values1 = pd.DataFrame()
    Q_Values2 = pd.DataFrame()

    Qs = torch.cat(allQuantiles_CCCP, dim = 1)[0]

    for i in range(num_class):
        Q_Values1['Quantile'] = [round(Qs[i].item(), 3)]
        Q_Values1['Names'] = f'Class {i}'
        Q_Values2 = Q_Values2.append(Q_Values1)
        #print(Qs[i])

    level_adjusted = (1.0 - alpha+0.02) * (1.0 + 1.0 / float(n_calib))

    #print(f"h1 = {mquantiles(scores_simple[0, :], prob=level_adjusted)[0]}")
    Q_Values1['Quantile'] = [round(mquantiles(scores_simple[0, :], prob=level_adjusted)[0], 3)]
    Q_Values1['Names'] = 'Vanilla CP'
    Q_Values2 = Q_Values2.append(Q_Values1)
    #print(f"Q = {Q_Values2} {Q_Values1}")
    #exit(1)

    for p in range(num_of_scores):
        thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        #thresholds[p, 1] = ClssThresholdsCCCP[p]

        #thresholds[p, 2] = ClssThresholdsICP[p]

    #print(f"thresholds1 = {thresholds}")
    #exit(1)
    return thresholds, ClssThresholdsCCCP, ClssThresholdsICP

def Vanilla_CP_tr(scores_simple=None, alpha = 0.1, num_of_scores=2):
    #print(f"alpha = {alpha}")

    thresholds = np.zeros((num_of_scores, 1))
    n_calib = scores_simple.shape[1]

    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))


    for p in range(num_of_scores):
        thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
    #print(f"thresholds = {thresholds}{n_calib}")
    #exit(1)
    return thresholds


def calibration_CCCP(scores_simple=None, eff_alpha = 0.0, num_of_scores=2, num_class = None, y = None):
    #print(f"alpha = {eff_alpha}")
    allQuantiles_ICP, allQuantiles_CCCP = [], []
    # size of the calibration set
    n_calib = scores_simple.shape[1]
    #print(f"n_calib = {n_calib}, {scores_simple.shape}")
    # create container for the calibration thresholds
    thresholds = np.zeros((num_of_scores, 1))

    # Compute thresholds

    if y is not None:
        for i in range(num_class):
            IndicesI = torch.where(y == i)[0]
            #print(len(IndicesI), 'len')
            #exit(1)
            level_adjusted_CCCP = (1.0 - eff_alpha) * (1.0 + 1.0 / float(len(IndicesI)))
            scoreI_CCCP = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted_CCCP, axis = 1)
            allQuantiles_CCCP.append(torch.from_numpy(scoreI_CCCP.data))

    ClssThresholdsCCCP = torch.cat(allQuantiles_CCCP, dim = 1).numpy()

    return ClssThresholdsCCCP

def calibration_CCCP1(scores_simple=None, eff_alpha = 0.0, num_of_scores=2, num_class = None, y = None):
    #print(f"alpha = {eff_alpha}")
    allQuantiles_ICP, allQuantiles_CCCP = [], []
    # size of the calibration set
    n_calib = scores_simple.shape[1]
    #print(f"n_calib = {n_calib}, {scores_simple.shape}")
    # create container for the calibration thresholds
    thresholds = np.zeros((num_of_scores, 1))

    # Compute thresholds

    if y is not None:
        for i in range(num_class):
            IndicesI = torch.where(y == i)[0]
            #print(len(IndicesI), 'len')
            #exit(1)
            level_adjusted_CCCP = (1.0 - eff_alpha) * (1.0 + 1.0 / float(len(IndicesI)))
            scoreI_CCCP = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted_CCCP, axis = 1)
            allQuantiles_CCCP.append(torch.from_numpy(scoreI_CCCP.data))

    ClssThresholdsCCCP = torch.max(torch.cat(allQuantiles_CCCP, dim = 1), dim = 1).values

    #ClssThresholdsCCCP = torch.cat(allQuantiles_CCCP, dim = 1).numpy()

    return ClssThresholdsCCCP
    

def calibration_ICP(scores_simple=None, alpha=0.1, eff_alpha = 0.0, num_of_scores=2, num_class = None, y = None):
    allQuantiles = []
    thresholds = np.zeros((num_of_scores, 1))
    #print(f"thresholds2 = {thresholds}")

    # Compute thresholds
    #print(f"eff_alpha = {eff_alpha}")
    #exit(1)
    if y is not None:
        for i in range(num_class):
            IndicesI = torch.where(y == i)[0]
            #print(len(IndicesI), 's111', i)
            #exit(1)

            level_adjusted = (1.0 - eff_alpha) * (1.0 + 1.0 / float(600))

            scoreI = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted, axis = 1)
            allQuantiles.append(torch.from_numpy(scoreI.data))
    ClssThresholdsICP = torch.cat(allQuantiles, dim = 1).numpy()
    #print(f"ClssThresholdsICP = {ClssThresholdsICP}")
    #exit(1)
    return ClssThresholdsICP


def prediction(scores_simple=None, num_of_scores=2, thresholds=None, CCCP_tau = None, CCP_tau = None, y_pred = None, num_class = None):
    # get number of points
    n = scores_simple.shape[1]
    #n = 4
    #print(CCCP_tau, 's1')
    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = [[], [], []]
    for p in range(num_of_scores):
        S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, 0])[0].tolist() for i in range(n)]
        S_hat_ICP = [np.where(scores_simple[p, i, :] <= CCP_tau[p, y_pred[i].item()])[0].tolist() for i in range(n)]

        sub = [[] for i in range(num_class)]
        for j in range(num_class):
            S_hat_CCCP11 = [np.where(scores_simple[p, i, :] <= CCCP_tau[p, j])[0].tolist() for i in range(n)]
            sub[j].append(S_hat_CCCP11)
        #print(sub, 'sub', sub[0], 's1', sub[1], 's2', sub[3])
        #print(sub[0][0], 's1', sub[1][0], 's2', sub[3][0])

        S_hat_CCCP = []
        for i in range(n):
            s = list()
            for j in range(num_class):
                s = s + sub[j][0][i]
                #print(s, 's1', S_hat_CCCP)
                #exit(1)
            S_hat_CCCP.append(list(set(s)))
            #print(s, 's', S_hat_CCCP)
            #exit(1)
        #exit(1)
        predicted_sets[0].append(S_hat_simple)
        predicted_sets[1].append(S_hat_CCCP)
        predicted_sets[2].append(S_hat_ICP)

    return predicted_sets

def prediction1(scores_simple=None, num_of_scores=2, thresholds=None, CCCP_tau = None, CCP_tau = None, y_pred = None, num_class = None):
    # get number of points
    n = scores_simple.shape[1]
    #n = 4
    CCCP_tau = CCCP_tau.numpy()
    #print(CCCP_tau, 's1')
    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = [[], [], []]
    for p in range(num_of_scores):
        S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, 0])[0].tolist() for i in range(n)]
        S_hat_CCCP = [np.where(scores_simple[p, i, :] <= CCCP_tau[p])[0].tolist() for i in range(n)]
        S_hat_ICP = [np.where(scores_simple[p, i, :] <= CCP_tau[p, y_pred[i].item()])[0].tolist() for i in range(n)]

        predicted_sets[0].append(S_hat_simple)
        predicted_sets[1].append(S_hat_CCCP)
        predicted_sets[2].append(S_hat_ICP)

    return predicted_sets

def prediction_vanillaCP(scores_simple=None, num_of_scores=2, thresholds=None):
    # get number of points
    n = scores_simple.shape[1]
    #n = 4
    predicted_sets = []
    for p in range(num_of_scores):
        S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, 0])[0].tolist() for i in range(n)]

        predicted_sets.append(S_hat_simple)
    return predicted_sets

def prediction_CCCP(scores_simple=None, num_of_scores=2, thresholds=None, CCCP_tau = None, CCP_tau = None, y_pred = None, num_class = None):
    # get number of points
    n = scores_simple.shape[1]
    #n = 4
    #print(CCCP_tau, 's1')
    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(num_of_scores):
        sub = [[] for i in range(num_class)]
        for j in range(num_class):
            S_hat_CCCP11 = [np.where(scores_simple[p, i, :] <= thresholds[p, j])[0].tolist() for i in range(n)]
            sub[j].append(S_hat_CCCP11)
        #print(sub, 'sub', sub[0], 's1', sub[1], 's2', sub[3])
        #print(sub[0][0], 's1', sub[1][0], 's2', sub[3][0])

        S_hat_CCCP = []
        for i in range(n):
            s = list()
            for j in range(num_class):
                s = s + sub[j][0][i]
            S_hat_CCCP.append(list(set(s)))

        predicted_sets.append(S_hat_CCCP)

    return predicted_sets

def prediction_CCCP1(scores_simple=None, num_of_scores=2, thresholds=None, CCCP_tau = None, CCP_tau = None, y_pred = None, num_class = None):
    # get number of points
    n = scores_simple.shape[1]
    thresholds = thresholds.numpy()
    #print(f"thresholds_vccp = {thresholds}")
    #exit(1)
    #print(CCCP_tau, 's1')
    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(num_of_scores):

        S_hat_CCCP11 = [np.where(scores_simple[p, i, :] <= thresholds[p])[0].tolist() for i in range(n)]
        predicted_sets.append(S_hat_CCCP11)


    return predicted_sets

def prediction_ICP(y_test_preds, scores_simple=None, num_of_scores=2, thresholds=None):
    # get number of points
    #print(f"thresholds = {thresholds}")
    #print(y_test_preds, 'y_test_preds')
    y_test_preds = y_test_preds
    n = scores_simple.shape[1]
    #print(f"y_test_preds = {y_test_preds[0].item(), y_test_preds[1], thresholds[0, y_test_preds[1].item()]}")
    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(num_of_scores):
        S_hat_ICP = [np.where(scores_simple[p, i, :] <= thresholds[p, y_test_preds[i].item()])[0].tolist() for i in range(n)]
        predicted_sets.append(S_hat_ICP)

    return predicted_sets

def evaluate_predictions_ICP(S, X, y, conditional=False, coverage_on_label=False, num_of_classes=10):


    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    size = np.mean([len(S[i]) for i in range(len(y))])

    return marg_coverage, size

def calibration_class(scores_simple=None, alpha=0.1, num_of_scores=2, num_class = None, y = None):
    allQuantiles = []
    thresholds = np.zeros((num_of_scores, 1))
    #print(f"thresholds2 = {thresholds}")

    # Compute thresholds

    if y is not None:
        for i in range(num_class):
            IndicesI = torch.where(y == i)[0]
            level_adjusted = (1.0 - eff_alpha) * (1.0 + 1.0 / float(len(IndicesI)))

            scoreI = mquantiles(np.array(scores_simple[:, IndicesI]), prob=level_adjusted, axis = 1)
            allQuantiles.append(torch.from_numpy(scoreI.data))
    ClssThresholdsICP = torch.cat(allQuantiles, dim = 1).numpy()

    return ClssThresholdsICP

def HyperTuning_CCP(args, y_test_preds, scores_simple_clean_cal, num_of_scores=2, num_class = None, y_cal = None, alpha=0.1, scores_simple_clean_hyper = None, y_hyper = None):

    #Class_thresholds = calibration_ICP(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal], alpha = alpha, eff_alpha=alpha, num_of_scores=2, num_class = num_class, y = y_cal)

    eff_alphas = []
    for i in range(num_class):
        lb = 0.0
        ub = alpha/1.1
        while ub - lb >= 0.001:
            eff_alpha = (lb+ub)/2
            Class_thresholds = calibration_ICP(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal], alpha = 0.1, eff_alpha=eff_alpha, num_of_scores=2, num_class = num_class, y = y_cal)


            idx = np.where(y_hyper == i)[0]
            #print(f"len = {len(y_hyper)}, {scores_simple_clean_hyper.shape}")
            predicted_clean_sets_base = prediction_ICP(y_test_preds[idx], scores_simple=scores_simple_clean_hyper[:, idx, :], num_of_scores=num_of_scores, thresholds=Class_thresholds)
            #print(len(predicted_clean_sets_base[0]), 's1', idx, 's2', len(idx))
            
            for p in range(1): #Only for APS
                min_cvg, size = evaluate_predictions_ICP(predicted_clean_sets_base[p], None, y_hyper[idx],
                                        conditional=False, coverage_on_label=True,
                                        num_of_classes=num_class)
            #print(f"min_cvg = {min_cvg}, size = {size} class = {i} eff_alpha = {eff_alpha}")
            if (min_cvg - (1 - args.alpha))  < 0.01 and min_cvg >= (1-alpha):
                break
            if min_cvg < 1 - alpha:
                ub = eff_alpha
            else:
                lb = eff_alpha
            
        eff_alphas.append(eff_alpha)
    #print(f"alphas = {eff_alphas}")
    return eff_alphas


def HyperTuning_CCP1(args, y_test_preds, scores_simple_clean_cal, num_of_scores=2, num_class = None, y_cal = None, alpha=0.1, scores_simple_clean_hyper = None, y_hyper = None):

    #Class_thresholds = calibration_ICP(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal], alpha = alpha, eff_alpha=alpha, num_of_scores=2, num_class = num_class, y = y_cal)

    eff_alphas = []
    ALPHAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099]
    #print(f"ALPHAS = {ALPHAS}")
    for i in range(num_class):
        Coverages = []
        Sizes = []
        Alphas = []

        for j in range(len(ALPHAS)):
            eff_alpha = alpha - ALPHAS[j]
            Alphas.append(eff_alpha)
            Class_thresholds = calibration_ICP(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal], alpha = 0.1, eff_alpha=eff_alpha, num_of_scores=2, num_class = num_class, y = y_cal)
            #print(f"Class_thresholds = {Class_thresholds}")

            idx = np.where(y_hyper == i)[0]
            #print(f"len = {len(y_hyper)}, {scores_simple_clean_hyper.shape}")
            predicted_clean_sets_base = prediction_ICP(y_test_preds[idx], scores_simple=scores_simple_clean_hyper[:, idx, :], num_of_scores=num_of_scores, thresholds=Class_thresholds)
            #print(len(predicted_clean_sets_base[0]), 's1', idx, 's2', len(idx))
            
            for p in range(1): #Only for APS
                min_cvg, size = evaluate_predictions_ICP(predicted_clean_sets_base[p], None, y_hyper[idx],
                                        conditional=False, coverage_on_label=True,
                                        num_of_classes=num_class)

            Coverages.append(min_cvg)
            Sizes.append(size)
        ids = np.where(np.array(Coverages) > 1-alpha)[0]
        eff_alphas.append(Alphas[ids[0]])
        #print(f"COVERAGES = {Coverages}, {ids}")
        #print(f"Size = {Sizes}, {ids}")

    return eff_alphas

def tune_Vanilla_CP(args, scores_simple_clean_cal, num_of_scores, num_class, y_cal, alpha, scores_simple_clean_hyper, y_hyper):
    
    lb = 0.0
    ub = 2*alpha
    while ub - lb >= 0.001:
        eff_alpha = (lb+ub)/2

        threshold = Vanilla_CP_tr(scores_simple=scores_simple_clean_cal, alpha = eff_alpha, num_of_scores=2)
        prediction_set = prediction_vanillaCP(scores_simple=scores_simple_clean_cal, num_of_scores=2, thresholds=threshold)
        for p in range(1): #Only for APS
            res = evaluate_predictions(prediction_set[p], None, y_hyper,
                                    conditional=False, coverage_on_label=False,
                                    num_of_classes=num_class)
            #print(res)
            min_cvg = res['Marginal Coverage'].to_numpy()[0]

        if (min_cvg - (1 - args.alpha))  < 0.01 and min_cvg >= (1-alpha):
            break
        if min_cvg < 1 - alpha:
            ub = eff_alpha
        else:
            lb = eff_alpha    
    return eff_alpha

def HyperTuning(args, y_test_preds, scores_simple_clean_cal, num_of_scores=2, num_class = None, y_cal = None, alpha=0.1, scores_simple_clean_hyper = None, y_hyper = None):

    CCP_alphas = HyperTuning_CCP(args, y_test_preds, scores_simple_clean_cal, num_of_scores, num_class, y_cal, alpha, scores_simple_clean_hyper, y_hyper)
    #print(f"CCP_alphas = {len(CCP_alphas)}")
    
    #exit(1)
    #Class_thresholds = calibration_ICP(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal], alpha = alpha, eff_alpha=alpha, num_of_scores=2, num_class = num_class, y = y_cal)

    #print('sss2')
    lb = 0.0
    ub = 2*alpha
    while ub - lb >= 0.001:
        eff_alpha = (lb+ub)/2
        Class_thresholds = calibration_CCCP1(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal],eff_alpha=eff_alpha, num_of_scores=2, num_class = num_class, y = y_cal)

        predicted_clean_sets_base = prediction_CCCP1(scores_simple=scores_simple_clean_hyper, num_of_scores=num_of_scores, thresholds=Class_thresholds, num_class = num_class)
        #print(len(predicted_clean_sets_base[0]), 's1', idx, 's2', len(idx))
        
        for p in range(1): #Only for APS
            res, _ = evaluate_predictions(predicted_clean_sets_base[p], None, y_hyper,
                                    conditional=False, coverage_on_label=True,
                                    num_of_classes=num_class)
            #print(res, 's3', Class_thresholds)
            min_cvg = np.min(res['Coverage'].to_numpy()[1:])
            #print(f"min_cvg = {min_cvg}")
        if (min_cvg - (1 - args.alpha))  < 0.01 and min_cvg >= (1-alpha):
            break
        if min_cvg < 1 - alpha:
            ub = eff_alpha
        else:
            lb = eff_alpha


    alpha_tilde = tune_Vanilla_CP(args, scores_simple_clean_cal, num_of_scores, num_class, y_cal, alpha, scores_simple_clean_hyper, y_hyper)
    #print(f"alpha_tilde = {alpha_tilde}")
    #exit(1)
    thresholds, CCCP_tau, CCP_tau = calibration1(scores_simple=scores_simple_clean_cal[:, np.arange(len(y_cal)), y_cal], alpha = alpha_tilde, CCP_alphas=CCP_alphas, CCCP_alpha = eff_alpha, num_of_scores=2, num_class = num_class, y = y_cal)
    
    #print(f"CCP_tau = {CCP_tau}")
    #print(f"alpha = {alpha_tilde} alpha_vccp = {eff_alpha} alpha_ccp = {np.mean(CCP_alphas)}")
    return thresholds, CCCP_tau, CCP_tau
