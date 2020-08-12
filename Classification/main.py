import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import math
import operator
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
import datetime
import random

np.set_printoptions(threshold=np.nan)

max_depth = 1000

# def how_many_values(dataset):
#
#

def infogain_left_right (current, left, right):

    parent_entropy = calc_entropy(current)

    child_1 = calc_entropy(left)
    child_2 = calc_entropy(right)

    children_entropy = (len(left)/len(current) * child_1) + (len(right)/len(current) * child_2)
    return (parent_entropy - children_entropy)

def spltting_cont_value(dataset, column):

    print("splitting:",column)

    cont_col = dataset[:, column]
    #print(cont_col)
    attr_values = np.unique(cont_col) #numpy unique returns sorted elements

    #print(attr_values)
    #print(len(attr_values))

    split_list = []
    for i in range(len(attr_values)-1):
        #print('calc...')
        val = (attr_values[i]+attr_values[i+1])/2
        split_list.append(val)
    #print("split list",split_list)
    #print("len of split list:", len(split_list))

    if(len(split_list) > 1000):

        temp = np.random.choice(split_list, size=100, replace=False)
        #print("sampled split list:\n",temp)
        split_list = temp.copy()
        #print("len of split list:", len(split_list))
        split_list.sort()
        #print("sorted split list:\n",split_list)


    max_gain = 0.0

    for split_val in split_list:
        left_rows = []
        right_rows = []

        for row in dataset:
            if (row[column] < split_val):
                left_rows.append(row)
            else:
                right_rows.append(row)
        #print(left_rows)
        gain = infogain_left_right(dataset, left_rows, right_rows)
        #print(gain)
        if(gain >= max_gain):
            max_gain = gain
            best_split = split_val
        #print("split val:",split_val)

    #print('best split value is: ',best_split)

    for row in dataset:
        if (row[column] < best_split):
            row[column] = 0
        else:
            row[column] = 1

    #print(dataset)

    return dataset


def dict_of_diffval_col(dataset, col):

    dict = {}
    # print(total_rows)
    # print(total_cols)

    # store count for each diff value of class in the dataset
    for row in dataset:

        class_label = row[col]
        # print(class_label)

        if (class_label not in dict):
            dict[class_label] = 1

        else:
            dict[class_label] += 1


    return dict

def get_rows_with_selected_column_value(dataset, attribute_index, value):

    subset = []

    for row in dataset:
        if (row[attribute_index] == value):
            subset.append(row)

    return subset

def calc_entropy(dataset):

    total_rows = len(dataset)
    total_cols = len(dataset[0])

    dict = dict_of_diffval_col(dataset,total_cols-1)

    entropy = 0.0

    if(dict != {}):

        for key in dict:

            prob_key = dict[key]/total_rows
            ind_entropy = -(math.log(prob_key,2)) * prob_key
            entropy += ind_entropy

    #print(entropy)
    return entropy


def calc_info_gain(dataset,attribute_index):

    # feature_col = dataset[:,attribute_index]
    # print(feature_col)
    # attr_values = np.unique(feature_col)
    # num_of_attr_values = len(np.unique(feature_col)) #unique returns all unique values, so get its length
    # print(num_of_attr_values)

    dict = dict_of_diffval_col(dataset,attribute_index)
    parent_entropy = calc_entropy(dataset)
    #print('parent entropy ',parent_entropy)

    children_entropy = 0.0

    for key in dict: #get subsets for each attr value

        weight = dict[key]/len(dataset)
        subset_attr_rows = get_rows_with_selected_column_value(dataset, attribute_index, key)
        #print(subset_attr_rows)
        children_entropy += calc_entropy(subset_attr_rows)*weight
        #print('child ind ',calc_entropy(subset_attr_rows))

    #print('total children entropy: ',children_entropy)
    infogain = parent_entropy - children_entropy
    #print('info gain: ',infogain)
    return infogain

def find_best_attribute_to_split(dataset,list_of_attributes):

    max_gain = 0.0

    for attr in list_of_attributes:
        gain = calc_info_gain(dataset,attr)
        #print('attr ',attr, ',gain ',gain)
        if(gain >= max_gain):
            max_gain = gain
            best_attr = attr

    #print('max gain: ',max_gain, 'best_attr: ', best_attr)
    return max_gain, best_attr

class Tree:

    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        self.is_terminal_node = False
        self.final_label = None

    def set_final_label(self, label):
        self.final_label = label

    def add_child(self, child):
        self.children.append(child)

    def add_subtree(self, key, subtree):
        self.children[key] = subtree

    def set_terminal_node_true(self):
        self.is_terminal_node = True

    def set_dataset(self,dataset):
        self.dataset = dataset.copy()

    def get_terminal_node_true(self):
        return self.is_terminal_node

    def get_dataset(self):
        return self.dataset

    def get_children(self):
        return self.children

    def get_attr_index(self):
        return self.attribute

def are_all_same(dataset):

    # last_col = dataset[:, len(dataset[0])-1]
    # print(last_col)
    # attr_values = np.unique(last_col)

    dict = dict_of_diffval_col(dataset,len(dataset[0])-1)
    #print(dict)
    if(len(dict) == 1):
        return True
    return False

def classify(dataset):

    dict = dict_of_diffval_col(dataset,len(dataset[0])-1)
    #print(dict)
    ans = max(dict.items(), key=operator.itemgetter(1))[0] #get the max frequency label(key in dict) acc to value=count
    #print(ans)
    return ans

def decision_tree_learning(dataset, examples, attributes, parent_examples, level, max_level):

    if(level >= max_level):

        #print('max level reached')
        if (len(examples) == 0):
            tr = Tree(None)
            tr.set_terminal_node_true()
            tr.set_final_label(classify(parent_examples))
            #tr.set_dataset(parent_examples)
            return tr

        elif (are_all_same(examples) == True):

            tr = Tree(None)
            tr.set_terminal_node_true()
            tr.set_final_label(classify(examples))
            #tr.set_dataset(examples)
            return tr

        elif (len(attributes) == 0):
            tr = Tree(None)
            tr.set_terminal_node_true()
            tr.set_final_label(classify(examples))
            #tr.set_dataset(examples)
            return tr

        else:
            tr = Tree(None)
            tr.set_terminal_node_true()
            tr.set_final_label(classify(examples))
            #tr.set_dataset(examples)
            return tr

    elif (len(examples) == 0):
        #print('samples finished')
        tr = Tree(None)
        tr.set_terminal_node_true()
        tr.set_final_label(classify(parent_examples))
        #tr.set_dataset(parent_examples)
        return tr

    elif (are_all_same(examples) == True):
        #print('all samples same class')
        tr = Tree(None)
        tr.set_terminal_node_true()
        tr.set_final_label(classify(examples))
        #tr.set_dataset(examples)
        return tr

    elif (len(attributes) == 0):
        #print('attributes finished')
        tr = Tree(None)
        tr.set_terminal_node_true()
        tr.set_final_label(classify(examples))
        #tr.set_dataset(examples)
        return tr

    else:
        #print('current attributes: ', attributes)
        gain, attr = find_best_attribute_to_split(examples, attributes) #get best attribute
        #print('best attribute chosen: ', attr)
        #print('level: ', level)
        tree = Tree(attr)

        dict = dict_of_diffval_col(dataset,attr)

        attr_remove_flag = 0

        #print(attributes)
        for key in dict:

            subset = get_rows_with_selected_column_value(examples, attr, key)
            #print('subsets: ',subset)
            if(attr_remove_flag == 0):
                attributes.remove(attr)
                attr_remove_flag = 1

            subtree = decision_tree_learning(dataset, subset, attributes, examples, level+1, max_level)
            tree.add_subtree(key,subtree)

        return tree

def predict_class(tree, test_row):
    #print('prediction starts')

    if(tree.get_terminal_node_true() == True):
        #print('predition end!!!')
        #print(tree.dataset)
        #print('returning: ',classify(tree.dataset))
        ans = tree.final_label
        return ans

    else:
        #print('recursionnnnn')
        attr = tree.get_attr_index()
        children = tree.get_children()

        for key_val in children:
            attr_val = key_val
            if(test_row[attr] == attr_val):
                #print('attribute branch = ',attr_val," attr:",attr)
                ans = predict_class(children[attr_val], test_row)
                return  ans

def Resample(dataset, w, N):
    #sample = pd.DataFrame.sample(dataset, n=N, weights=w)
    sample = random.choices(dataset,weights=w,k=N)
    return sample

def Normalize(w):

    w_sum = np.sum(w)
    #print("sum: ", w_sum)

    # for i in range(len(w)):
    #      w[i] = w[i]/w_sum

    w = w/w_sum

    #print("sum: ", sum(w))

    return w

def Adaboost(examples, attributes, K, level):

    N = len(examples)
    w = []

    for i in range(N):
        w.append(1/N)
    #print("initial w vector: ",w)

    h = [] #save diff trees
    z = [] #save diff tree weights

    print("\nRunning adaboost...")
    #k = 0

    #while (k < K):
    for k in range(K):

        #print("\nIteration: ",k)

        #dataframe = pd.DataFrame(examples)
        data = Resample(examples,w,N)
        #data = data.values

        #print("resampled data:\n",data)
        tree = decision_tree_learning(data,data,attributes,None,0,level)
        #print("tree learned")

        error = 0.0
        for j in range(N):
            row = examples[j]
            verdict = predict_class(tree, row)

            #print("actual: ",row[len(examples[0])-1])
            #print("predicted: ", verdict)

            if(verdict != row[len(examples[0])-1]):
                error += w[j]

        #print("\nError for this tree: ",error,"\n")

        if (error > 0.5):
                #print("can't be improved by boosting")
                #print("value of k:", k)
                continue

        for j in range(N):
            row = examples[j]
            verdict = predict_class(tree, row)

            #print("actual: ", row[len(examples[0]) - 1])
            #print("predicted: ", verdict)

            if (verdict == row[len(examples[0]) - 1]):
                w[j] = w[j]*error/(1.0-error)
                #print("new weight row: ", j+1, "is: ", w[j])

        #print("final sample weights: ",w)

        w = Normalize(w)
        #print("Normalized sample weights: ",w)

        if(error == 0.0):
            tree_weight = 100
        else:
            tree_weight = math.log((1.0-error)/error,2)

        #print("error: ", error)
        #print("tree weight", tree_weight)

        z.append(tree_weight)
        h.append(tree)
        #k = k + 1

    #print("hypotehsis trees: ",h)
    #print("hypothesis weights: ",z)
    #print("value of k:",k)

    return h,z

def predict_Adaboost(uniq,row,h,z):

    #print("\npredicting by adaboost...")

    final_sum = 0.0
    for i in range(len(z)):
        #print("weight: ",z[i])

        tree = h[i]
        verdict = predict_class(tree,row)

        if(verdict == uniq[1]): #second val of uniq is +ve
            #print("positive")
            final_sum += 1.0*z[i]

        elif(verdict == uniq[0]): #first val of uniq is +ve
            #print("negative")
            final_sum += -1.0*z[i]

        #print("new weighted sum: ",final_sum)


    #print("final weighted sum: ",final_sum)

    if(final_sum > 0):
        return uniq[1]
    else:
        return uniq[0]



def process_dataset_one(data):

    print('preprocessing dataset one...')
    dataset = data.copy() #pandas frame

    dataset = dataset.replace(' ?', np.NaN)
    dataset = dataset.replace('?', np.NaN)
    dataset = dataset.replace(' ', np.NaN)
    #dataset = dataset.replace(r'\s+', np.nan, regex=True)
    dataset = dataset.dropna(subset=[dataset.columns[len(dataset.columns) - 1]], axis=0)

    #print(dataset)
    #print(dataset.iloc[488])

    # pd.to_numeric(tmp)

    dataset = dataset.values  # convert to numpy array
    dataset = np.delete(dataset, 0, axis=1) # remove first column of customer id
    #dataset = np.delete(dataset,18,axis=1)

    # convert_col = dataset[:, 18]
    # print(convert_col)
    # convert_col.astype(np.float)

    dataset = pd.DataFrame(dataset) #convert back to dataframe)

    most_freq = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]
    mean_replace = [4,17,18] #18 prblm
    binarize = [4,17,18] #18 prblm
    #label_encode = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,18]

    for i in most_freq:
        # replace string missing values with most frequent
        max = dataset[dataset.columns[i]].value_counts().idxmax()
        dataset = dataset.replace(value=max, to_replace={dataset.columns[i]: np.NaN})

    dataset = dataset.values  # convert to numpy array

    #replace nonstring missing values with mean
    for i in mean_replace:
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(dataset[:, [i]])
        dataset[:, [i]] = imputer.transform(dataset[:, [i]])

    # for i in label_encode:
    #     labelencoder = LabelEncoder()
    #     dataset[:, i] = labelencoder.fit_transform(dataset[:, i])

    for i in binarize:
        spltting_cont_value(dataset,i)

    #print("binarize done")
    return dataset


def process_dataset_two(data):

    print('preprocessing dataset two...')
    dataset = data.copy() #pandas frame

    #put missing values to nan
    dataset = dataset.replace(' ?', np.NaN)
    dataset = dataset.replace('?', np.NaN)
    dataset = dataset.replace(' ', np.NaN)
    dataset = dataset.replace('.', ' ')

    #drop rows with missing class
    dataset = dataset.dropna(subset=[dataset.columns[len(dataset.columns) - 1]], axis=0)

    most_freq = [1,3,5,6,7,8,9,13]
    mean_replace = [0,2,4,10,11,12]
    #label_encode = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    label_encode = [14]
    binarize = [0,2,4,10,11,12]

    for i in most_freq:
        # replace string missing values with most frequent
        max = dataset[dataset.columns[i]].value_counts().idxmax()
        dataset = dataset.replace(value=max, to_replace={dataset.columns[i]: np.NaN})

    dataset = dataset.values  # convert to numpy array

    #replace nonstring missing values with mean
    for i in mean_replace:
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(dataset[:, [i]])
        dataset[:, [i]] = imputer.transform(dataset[:, [i]])

    for i in label_encode:
        labelencoder = LabelEncoder()
        dataset[:, i] = labelencoder.fit_transform(dataset[:, i])

    # for i in binarize:
    #     binarizer = Binarizer().fit(dataset[:, [i]])
    #     dataset[:, [i]] = binarizer.transform(dataset[:, [i]])

    for i in binarize:
        spltting_cont_value(dataset,i)

    #print(dataset)
    return dataset


def process_dataset_three(data):

    print('preprocessing dataset three...')
    dataset = data.copy() #pandas frame

    #take a subset of whole with all positives
    #dataset = dataset.values

    #subset = dataset.iloc[dataset[len(dataset.columns)-1] == 1]
    is_1 = dataset['Class'] == 1
    all1_subset = dataset[is_1]
    #print("take all +ves\n:",all1_subset)

    is_0 = dataset['Class'] == 0
    all0_subset = dataset[is_0]

    samp = pd.DataFrame.sample(all0_subset, n=20000)

    final_data = all1_subset.append(samp,ignore_index=True)
    dataset = final_data.copy()
    dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    #print(dataset)


    #put missing values to nan
    dataset = dataset.replace(' ?', np.NaN)
    dataset = dataset.replace('?', np.NaN)
    dataset = dataset.replace(' ', np.NaN)

    #drop rows with missing class
    dataset = dataset.dropna(subset=[dataset.columns[len(dataset.columns) - 1]], axis=0)

    #most_freq = [1,3,5,6,7,8,9,13]
    mean_replace = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    #label_encode = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    binarize = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

    # for i in most_freq:
    #     # replace string missing values with most frequent
    #     max = dataset[dataset.columns[i]].value_counts().idxmax()
    #     dataset = dataset.replace(value=max, to_replace={dataset.columns[i]: np.NaN})

    dataset = dataset.values  # convert to numpy array

    #replace nonstring missing values with mean
    for i in mean_replace:
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(dataset[:, [i]])
        dataset[:, [i]] = imputer.transform(dataset[:, [i]])

    # for i in label_encode:
    #     labelencoder = LabelEncoder()
    #     dataset[:, i] = labelencoder.fit_transform(dataset[:, i])

    # for i in binarize:
    #     binarizer = Binarizer().fit(dataset[:, [i]])
    #     dataset[:, [i]] = binarizer.transform(dataset[:, [i]])

    for i in binarize:
        spltting_cont_value(dataset,i)

    #print(dataset)
    return dataset


def split(dataset):

    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    return train_set,test_set

def main():

    dataset = pd.read_csv("first_data.csv", delimiter=',')
    dataset = process_dataset_one(dataset)

    # dataset = pd.read_csv("second_data.csv", delimiter=',',skipinitialspace=True,header=None)
    # dataset = process_dataset_two(dataset)
    # train_set = dataset.copy()
    # test_set = pd.read_csv("second_data_test.csv", delimiter=',',skipinitialspace=True,header=None)
    # test_set = process_dataset_two(test_set)

    # dataset = pd.read_csv("third_data.csv", delimiter=',')
    # dataset = process_dataset_three(dataset)


    train_set, test_set = split(dataset)
    #print("initial dataset:\n",dataset)

    col = dataset[:, len(dataset[0]) - 1]
    uniq = np.unique(col)
    #print(uniq)

    attributes = []
    for i in range(len(train_set[0])-1):
        attributes.append(i)
    #print('printing attributes: ',attributes)



    tree = decision_tree_learning(train_set, train_set, attributes, None, 0, max_depth)
    print("tree building done!\n")

    print("Test Set Performance")

    total = len(test_set)
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(test_set)):

        test_row = test_set[i]
        verdict = predict_class(tree, test_row)
        actual = test_row[len(test_row) - 1]

        # print("actual: ", actual)
        # print("verdict: ", verdict)
        # print()

        if (actual == uniq[1]):

            if(verdict == uniq[0]):
                FN += 1

            elif(verdict == uniq[1]):
                TP += 1

        elif (actual == uniq[0]):

            if (verdict == uniq[0]):
                TN += 1

            elif (verdict == uniq[1]):
                FP += 1

    # print("TP: ",TP)
    # print("TN: ",TN)
    # print("FP: ",FP)
    # print("FN: ",FN)
    # print("total: ",total)

    acc = (TP + TN) / total
    print("\nAccuracy:", acc)

    if((TP + FN) != 0):
        TPR = TP/(TP+FN) #Recall
        print("True Positive Rate:", TPR)
    else:
        print("True Positive Rate: denominator = 0")

    if((FP + TN) != 0):
        TNR = TN/(FP+TN) #specificity
        print("True Negative Rate:", TNR)
    else:
        print("True Negative Rate: denominator = 0")

    if((TP + FP) != 0):
        PPV = TP/(TP+FP) #precision
        print("Positive Predictive Value:",PPV)
    else:
        print("Positive Predictive Value: denominator = 0")

    if((TP + FP) != 0):
        FDR = FP/(TP+FP)
        print("False Discovery Rate:",FDR)
    else:
        print("False Discovery Rate: denominator = 0")


    if(TPR != 0 and PPV != 0):
        F1 = 2/((1/TPR)+(1/PPV))
        print("F1 Score:",F1)
    else:
        print("F1 Score: denominator = 0")

    #print("Time : ", datetime.datetime.now().time())



    print("\n\nTrain Set Performance")

    total = len(train_set)
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(train_set)):

        test_row = train_set[i]
        verdict = predict_class(tree, test_row)
        actual = test_row[len(test_row) - 1]

        # print("actual: ", actual)
        # print("verdict: ", verdict)
        # print()

        if (actual == uniq[1]):

            if (verdict == uniq[0]):
                FN += 1

            elif (verdict == uniq[1]):
                TP += 1

        elif (actual == uniq[0]):

            if (verdict == uniq[0]):
                TN += 1

            elif (verdict == uniq[1]):
                FP += 1

    # print("TP: ", TP)
    # print("TN: ", TN)
    # print("FP: ", FP)
    # print("FN: ", FN)
    # print("total: ", total)

    acc = (TP + TN) / total
    print("\nAccuracy:", acc)

    if ((TP + FN) != 0):
        TPR = TP / (TP + FN)  # Recall
        print("True Positive Rate:", TPR)
    else:
        print("True Positive Rate: denominator = 0")

    if ((FP + TN) != 0):
        TNR = TN / (FP + TN)  # specificity
        print("True Negative Rate:", TNR)
    else:
        print("True Negative Rate: denominator = 0")

    if ((TP + FP) != 0):
        PPV = TP / (TP + FP)  # precision
        print("Positive Predictive Value:", PPV)
    else:
        print("Positive Predictive Value: denominator = 0")

    if ((TP + FP) != 0):
        FDR = FP / (TP + FP)
        print("False Discovery Rate:", FDR)
    else:
        print("False Discovery Rate: denominator = 0")

    if (TPR != 0 and PPV != 0):
        F1 = 2 / ((1 / TPR) + (1 / PPV))
        print("F1 Score:", F1)
    else:
        print("F1 Score: denominator = 0")

    #print("Time : ", datetime.datetime.now().time())






    K_values = [5,10,15,20]

    for k in K_values:

        h,z = Adaboost(train_set, attributes, k, 1)

        print("Test Set Performance")

        total = len(test_set)
        count_error = 0

        for i in range(len(test_set)):

            test_row = test_set[i]
            verdict = predict_Adaboost(uniq,test_row,h,z)

            #print("actual: ", test_row[len(test_row) - 1])
            #print("verdict: ", verdict)

            if (verdict != test_row[len(test_row) - 1]):
                count_error += 1

        acc = (total - count_error) / total
        print("Accuracy of k =",k,": ",acc)
        #print("Time : ", datetime.datetime.now().time())


        print("Train Set Performance")

        total = len(train_set)
        count_error = 0

        for i in range(len(train_set)):

            test_row = train_set[i]
            verdict = predict_Adaboost(uniq, test_row, h, z)

            # print("actual: ", test_row[len(test_row) - 1])
            # print("verdict: ", verdict)

            if (verdict != test_row[len(test_row) - 1]):
                count_error += 1

        acc = (total - count_error) / total
        print("Accuracy of k =", k, ": ", acc)
        #print("Time : ", datetime.datetime.now().time())




main()

