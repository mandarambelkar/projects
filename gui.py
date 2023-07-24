import pandas as pd
import numpy as np
import re
from publicsuffixlist import PublicSuffixList
import gc
import math
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as snsn
from sklearn.metrics import classification_report, confusion_matrix
RANDOM_SEED = 1

# %matplotlib inline

from PIL import ImageTk,Image
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image

app = tk.Tk()

HEIGHT = 700
WIDTH = 700

app.resizable(0,0)
canvas = Canvas(width=1300, height=700)
canvas.pack()
filename=('dga.jpg')
load = Image.open(filename)
load = load.resize((1300, 700), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
load = Image.open(filename)
img.place(x=1, y=1)

frame = tk.Frame(app,  bg='#3e3e32', bd=5)
frame.place(relx=0.3, rely=0.1, relwidth=0.5, relheight=0.25, anchor='n')


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import re
from publicsuffixlist import PublicSuffixList
import gc
import math
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
RANDOM_SEED = 1

# %matplotlib inline

df = pd.read_csv('./Data/dga_data.csv')
df

df.isnull().sum()

df = df.drop(['domain'],axis='columns')
df.head(5)

df.info()

df.describe()

df.columns = ['Class','Domain','Subclass']
df.head(5)

df['Class'].unique()

# Load Valid Top Level Domains data
import sys

topLevelDomain = []
with open('./Data/tlds-alpha-by-domain.txt', 'r') as content:
    for line in content:
        topLevelDomain.append((line.strip('\n')))
        
print(topLevelDomain)

psl = PublicSuffixList()

def ignoreVPS(domain):
    # Return the rest of domain after ignoring the Valid Public Suffixes:
    validPublicSuffix = '.' + psl.publicsuffix(domain)
    if len(validPublicSuffix) < len(domain):
         # If it has VPS
        subString = domain[0: domain.index(validPublicSuffix)]  
    elif len(validPublicSuffix) == len(domain):
        return 0
    else:
        # If not
        subString = domain
    
    return subString

def typeTo_Binary(type):
  # Convert Type to Binary variable DGA = 1, Normal = 0
  if type == 'dga':
    return 1
  else:
    return 0

def domain_length(domain):
  # Generate Domain Name Length (DNL)
  return len(domain)

def subdomains_number(domain):
  # Generate Number of Subdomains (NoS)
    subdomain = ignoreVPS(domain)
    return (subdomain.count('.') + 1)

def subdomain_length_mean(domain):
  # enerate Subdomain Length Mean (SLM) 
    subdomain = ignoreVPS(domain)
    result = (len(subdomain) - subdomain.count('.')) / (subdomain.count('.') + 1)
    return result

def has_www_prefix(domain):
  # Generate Has www Prefix (HwP)
  if domain.split('.')[0] == 'www':
    return 1
  else:
    return 0
  
def has_hvltd(domain):
  # Generate Has a Valid Top Level Domain (HVTLD)
  if domain.split('.')[len(domain.split('.')) - 1].upper() in topLevelDomain:
    return 1
  else:
    return 0
  
def contains_single_character_subdomain(domain):
  # Generate Contains Single-Character Subdomain (CSCS) 
    domain = ignoreVPS(domain)
    str_split = domain.split('.')
    minLength = len(str_split[0])
    for i in range(0, len(str_split) - 1):
        minLength = len(str_split[i]) if len(str_split[i]) < minLength else minLength
    if minLength == 1:
        return 1
    else:
        return 0

def contains_TLD_subdomain(domain):
  # Generate Contains TLD as Subdomain (CTS)
    subdomain = ignoreVPS(domain)
    str_split = subdomain.split('.')
    for i in range(0, len(str_split) - 1):
        if str_split[i].upper() in topLevelDomain:
            return 1
    return 0

def underscore_ratio(domain):
  # Generate Underscore Ratio (UR) on dataset
    subString = ignoreVPS(domain)
    result = subString.count('_') / (len(subString) - subString.count('.'))
    return result

def contains_IP_address(domain):
  # Generate Contains IP Address (CIPA) on datasetx
    splitSet = domain.split('.')
    for element in splitSet:
        if(re.match("\d+", element)) == None:
            return 0
    return 1  

def contains_digit(domain):
    """
    Contains Digits 
    """
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        if item.isdigit():
            return 1
    return 0

def vowel_ratio(domain):
    """
    calculate Vowel Ratio 
    """
    VOWELS = set('aeiou')
    v_counter = 0
    a_counter = 0
    ratio = 0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        if item.isalpha():
            a_counter+=1
            if item in VOWELS:
                v_counter+=1
    if a_counter>1:
        ratio = v_counter/a_counter
    return ratio

def digit_ratio(domain):
    """
    calculate digit ratio
    """
    d_counter = 0
    counter = 0
    ratio = 0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        if item.isalpha() or item.isdigit():
            counter+=1
            if item.isdigit():
                d_counter+=1
    if counter>1:
        ratio = d_counter/counter
    return ratio
  
def prc_rrc(domain):
    """
    calculate the Ratio of Repeated Characters in a subdomain
    """
    subdomain = ignoreVPS(domain)
    subdomain = re.sub("[.]", "", subdomain)
    char_num=0
    repeated_char_num=0
    d = collections.defaultdict(int)
    for c in list(subdomain):
        d[c] += 1
    for item in d:
        char_num +=1
        if d[item]>1:
            repeated_char_num +=1
    ratio = repeated_char_num/char_num
    return ratio

def prc_rcc(domain):
    """
    calculate the Ratio of Consecutive Consonants
    """
    VOWELS = set('aeiou')
    counter = 0
    cons_counter=0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        i = 0
        if item.isalpha() and item not in VOWELS:
            counter+=1
        else:
            if counter>1:
                cons_counter+=counter
            counter=0
        i+=1
    if i==len(subdomain) and counter>1:
        cons_counter+=counter
    ratio = cons_counter/len(subdomain)
    return ratio

def prc_rcd(domain):
    """
    calculate the ratio of consecutive digits
    """
    counter = 0
    digit_counter=0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        i = 0
        if item.isdigit():
            counter+=1
        else:
            if counter>1:
                digit_counter+=counter
            counter=0
        i+=1
    if i==len(subdomain) and counter>1:
        digit_counter+=counter
    ratio = digit_counter/len(subdomain)
    return ratio

def prc_entropy(domain):
    """
    calculate the entropy of subdomain
    :param domain_str: subdomain
    :return: the value of entropy
    """
    subdomain = ignoreVPS(domain)
    # get probability of chars in string
    prob = [float(subdomain.count(c)) / len(subdomain) for c in dict.fromkeys(list(subdomain))]

    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

# Ready to generate features
def extract_features():
    df['DNL'] = df['Domain'].apply(lambda x: domain_length(x))
    df['NoS'] = df['Domain'].apply(lambda x: subdomains_number(x))
    df['SLM'] = df['Domain'].apply(lambda x: subdomain_length_mean(x))
    df['HwP'] = df['Domain'].apply(lambda x: has_www_prefix(x))
    df['HVTLD'] = df['Domain'].apply(lambda x: has_hvltd(x))
    df['CSCS'] = df['Domain'].apply(lambda x: contains_single_character_subdomain(x))
    df['CTS'] = df['Domain'].apply(lambda x: contains_TLD_subdomain(x))
    df['UR'] = df['Domain'].apply(lambda x: underscore_ratio(x))
    df['CIPA'] = df['Domain'].apply(lambda x: contains_IP_address(x))
    df['contains_digit']= df['Domain'].apply(lambda x:contains_digit(x))
    df['vowel_ratio']= df['Domain'].apply(lambda x:vowel_ratio(x))
    df['digit_ratio']= df['Domain'].apply(lambda x:digit_ratio(x))
    df['RRC']= df['Domain'].apply(lambda x:prc_rrc(x))
    df['RCC']= df['Domain'].apply(lambda x:prc_rcc(x))
    df['RCD']= df['Domain'].apply(lambda x:prc_rcd(x))
    df['Entropy']= df['Domain'].apply(lambda x:prc_entropy(x))

extract_features()

# Change Type virable from DGA and Normal to 1 and 0
df['Class'] = df['Class'].apply(lambda x: typeTo_Binary(x))

df.head(5)

df.dtypes

df.describe()

df.to_csv('./Data/domainwith_features.csv',index=False)

df = pd.read_csv('./Data/domainwith_features.csv')

corrmat = df.corr()


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Class')['Class'].index
cm = np.corrcoef(df[cols].values.T)
f, ax = plt.subplots(figsize=(16, 16))

drop_column = {'Subclass', 'Domain', 'HwP', 'HVTLD', 'UR', 'CSCS', 'CTS','CIPA'}

# Drop the unnecessary columns
df = df.drop(drop_column, axis = 1)
df.head()

df.describe()

df.isnull().sum()

# Get independent variables and dependent variables
attributes = df.drop('Class', axis=1)
observed = df['Class']
attributes.shape, observed.shape

# Split the dataset into training dataset and test dataset
train_X, test_X, train_y, test_y = train_test_split(attributes, observed, test_size = 0.1, random_state = RANDOM_SEED)
train_X.shape, test_X.shape, train_y.shape, test_y.shape

#Random Forest
rf = RandomForestClassifier(random_state= RANDOM_SEED)
rf.fit(train_X, train_y)


# prediction 
def domain_pred():
    dname = textbox.get("1.0",END)
    df1 = pd.DataFrame({'Domain': dname},index=['0'])
    print(df1)
    #df1 = pd.DataFrame({'Domain':dname})
    df1['DNL'] = df1['Domain'].apply(lambda x: domain_length(x))
    df1['NoS'] = df1['Domain'].apply(lambda x: subdomains_number(x))
    df1['SLM'] = df1['Domain'].apply(lambda x: subdomain_length_mean(x))
    df1['HwP'] = df1['Domain'].apply(lambda x: has_www_prefix(x))
    df1['HVTLD'] = df1['Domain'].apply(lambda x: has_hvltd(x))
    df1['CSCS'] = df1['Domain'].apply(lambda x: contains_single_character_subdomain(x))
    df1['CTS'] = df1['Domain'].apply(lambda x: contains_TLD_subdomain(x))
    df1['UR'] = df1['Domain'].apply(lambda x: underscore_ratio(x))
    df1['CIPA'] = df1['Domain'].apply(lambda x: contains_IP_address(x))
    df1['contains_digit']= df1['Domain'].apply(lambda x:contains_digit(x))
    df1['vowel_ratio']= df1['Domain'].apply(lambda x:vowel_ratio(x))
    df1['digit_ratio']= df1['Domain'].apply(lambda x:digit_ratio(x))
    df1['RRC']= df1['Domain'].apply(lambda x:prc_rrc(x))
    df1['RCC']= df1['Domain'].apply(lambda x:prc_rcc(x))
    df1['RCD']= df1['Domain'].apply(lambda x:prc_rcd(x))
    df1['Entropy']= df1['Domain'].apply(lambda x:prc_entropy(x))
    drop_column = {'Domain', 'HwP', 'HVTLD', 'UR', 'CSCS', 'CTS','CIPA'}
    # Drop the unnecessary columns
    df1 = df1.drop(drop_column, axis = 1)
    df1.head()
    df1.dropna(inplace=True)
    test = df1
    test_pred = rf.predict(test)
    print(test_pred[0])
    if test_pred[0]==1:
        textbox1.insert("end-1c","DGA")
    else:
        textbox1.insert("end-1c","Legit")

#DGA = 1 and Legit = 0
def clearall():
    textbox.delete("1.0","end")
    textbox1.delete("1.0","end")


lab1 = Label(frame, text="Enter Domain to predict: ", bg="#FF9A29", fg="Black")
lab1.grid(row=2, column=1, padx=5)

textbox = tk.Text(frame, font=20,width="30",height=2)
textbox.grid(row=2, column=2)

submit = tk.Button(frame,font=40, text='Predict',height=1,width="13",command=domain_pred)
submit.grid(row=3, column=2,padx=20,pady=20)

textbox1 = tk.Text(frame, font=20,width="30",height=2)
textbox1.grid(row=4, column=2)

clear = tk.Button(frame,font=40, text='Clear',height=1,width="13",command=clearall)
clear.grid(row=3, column=3,padx=20,pady=20)


app.mainloop()
