'''
PROGRAMMING ASSIGNMENT 2: Word Sense Disambiguation (WSD) by Team 4: AIT-580 under Dr. Liao

Introduction :
Authors : Muhammad Hassan
Date : 06/24/2021

WSD is a technique in NLP and ontology. This refers to identification of in which sense a word is
used in a sentence, when the word has several meanings.
This scorer.py programs takes the file generated from decision-list.py and compare those results with the gold standard
available i.e line-answers.txt and generates the accuracy and confusion matrix. Along with this the most
frequent sense baseline accuracy is also calculated.

Algorithm:
Step1: Extract the output file generated from decision-list.py and gold standard file
Step2: Read the sense id and keys
Step3: For a specific key check if the values matches
Step4: Calculate the accuracy
Step5: generate confusion matrix

Instructions to run decision-list.py :
1) Run the scorer.py program in the command prompt as follows:
$ python scorer.pl my-line-answers.txt line-answers.txt
2) The baseline accuracy, accuracy after adding features and confusion matrix are printed.

Sample Output (with word indices 8 on either side of the word. This gives optimal output) :
Baseline accuracy is 57.14285714285714%
Accuracy after adding learned features is 84.12698412698413%
Confusion matrix is
col_0    phone  product
row_0
phone       59        7
product     13       47


'''



import pandas as pd
import re
import sys
import seaborn as sn
import matplotlib.pyplot as plt


# command line arguments for the file sources of results obtained from decision-list and available gold standard
my_key = sys.argv[1]
gs_key = sys.argv[2]

#function to search the strings and get keys, sense
def get_senses(mylist):
    sense = {}
    keys = []
    for string in mylist:
        search = re.search('<answer instance="(.*)" senseid="(.*)"/>', string, re.IGNORECASE)
        key = search.group(1)
        keys.append(key)
        value = search.group(2)
        sense[key] = value
    return sense, keys

#open the two files taken as input from command line and strip out '\n'
with open(gs_key, 'r') as data:
    mylist1 = [line.rstrip('\n') for line in data]
answers, keys = get_senses(mylist1)

with open(my_key, 'r') as data:
    mylist2 = [line.rstrip('\n') for line in data]
preds, keys = get_senses(mylist2)

# For a specific key check if the values matches
correct = 0
total = len(keys)
for key in keys:
    if(answers[key] == preds[key]):
        correct += 1
correct

#caluclated the most frequent sense baseline
baseline_count = 0
for key in keys:
    if(answers[key] == 'phone'):
        baseline_count += 1
baseline_acc = (float(baseline_count)/float(total))*100
print("Baseline accuracy is "+str(baseline_acc)+"%")


# calucalate the accuracy after learning features
accuracy = (float(correct)/float(total))*100
print("Accuracy after adding learned features is "+str(accuracy)+"%")


#creating array for our output and append the values to list
pred_list = []
for v in preds:
    pred_list.append(preds[v])


#creating array for gold standard and append the values to its list
answers_list = []
for v in answers:
    answers_list.append(answers[v])

# creating dataframes for both the files
df1 = pd.Series( (v for v in pred_list) )
df2 = pd.Series( (v for v in answers_list) )

#generating confusion matrix
df_confusion = pd.crosstab(df1, df2, rownames=['Actual'], colnames=['Predicted'])

#textual representation of the confusion matrix
print("Confusion matrix is\n"  +str(df_confusion))

#displaying confusion matrix in graphical way using seaborn
sn.heatmap(df_confusion, annot=True)
plt.show()
