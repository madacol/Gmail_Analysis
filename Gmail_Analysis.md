
# Gmail analysis
### Exploring more than 13 years of Gmail messages

Forked from: https://github.com/jsdiazpo/Gmail_Analysis/blob/master/Gmail_Analysis.ipynb

The first step is [requesting the data](https://takeout.google.com/settings/takeout). There is data available for several Google services, only the Gmail data is used here. Depending on the amount of data the request can take several hours. Once we notified that the file is ready to be downloaded, the data will come in a special format called `mailbox`. After importing some useful modules we can explore and clean the data.


## Dependencies
    pip install pandas matplotlib seaborn NLTK


```python
import mailbox
import pandas as pd
import csv
import unicodedata
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(rc={'figure.facecolor':'white'})
#import dateparser

from time import time
import operator
```

## Data preprocessing
We begin the analysis by saving the data file `gmail_data.mbox` in a directory called `data` (for privacy reasons, this file is absent from the Github repository). The file can be loaded using the `mailbox` module


```python
dir_path = 'data/'
filename = 'gmail_data.mbox'
file_path = dir_path + filename
mbox = mailbox.mbox(file_path)
sample = len(mbox)
print('samples:', sample)
```

samples: 24249

The file contains 24249 samples. Even though these are mostly email messages, many other entry types are counted, such as drafts and chats. These can be removed by filtering by Gmail label. The file contains a huge amount of labels

```python
headersFreqDict={}
for mail in mbox:
  for header in set(mail.keys()):
    if header in headersFreqDict:
      headersFreqDict[header] += 1
    else:
      headersFreqDict[header] = 1

sortedHeadersFreqArray= sorted(headersFreqDict.items(), key=operator.itemgetter(1), reverse=True)

for header, freq in sortedHeadersFreqArray[:20]:
  print("{:5.3f}".format(round(100*freq/sample, 3)),'% |', header)
```

100.0 X-GM-THRID
99.996 From
99.992 X-Gmail-Labels
98.891 Content-Type
95.039 MIME-Version
64.007 Received
63.97 Date
63.937 Subject
63.862 To
61.314 Return-Path
37.956 Delivered-To
35.977 Content-Transfer-Encoding
35.049 Received-SPF
35.049 Authentication-Results
34.232 X-Received
32.022 Message-ID
31.977 Message-Id
30.249 User-Agent
27.708 DKIM-Signature
27.696 X-Google-Original-From


These are the top 20 more frequent headers



We find that there are several section of little interest. In order to avoid loading unnecessary information, we can extract the fields of interest and put them into a `pandas` dataframe for further processing. We are interested in the following fields: `subject`, `from`, `to`, `date`, and `Gmail-label`.


```python
t0 = time()
subject = []
from_ = []
to = []
date = []
label = []
for i, message in enumerate(mbox):
    try:
        if i%2000 == 0:
            print(i, end=' ')
        subject.append(message['subject'])
        from_.append(message['from'])
        to.append(message['to'])
        date.append(message['date'])
        label.append(message['X-Gmail-Labels'])
    except:
        print(i, end=' ')
        print('subject', subject[i])
        print('from', from_[i])
        print('to', to[i])
        print('date', date[i])
        print('label', label[i])
print('\ntime: {:.1f} min'.format((time()-t0)/60))
```


```python
df = pd.DataFrame()
df['subject'] = subject
df['from'] = from_
df['to'] = to
df['date'] = date
df['label'] = label
```


```python
df[['subject', 'date', 'label']].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>consulta DS</td>
      <td>Tue, 8 Aug 2017 11:20:05 +0200</td>
      <td>Important,Sent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Re: consulta DS</td>
      <td>Tue, 08 Aug 2017 10:35:39 +0100</td>
      <td>Important,Inbox</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Re: consulta DS</td>
      <td>Tue, 08 Aug 2017 10:43:30 +0100</td>
      <td>Important,Inbox</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Re: consulta DS</td>
      <td>Tue, 8 Aug 2017 13:16:37 +0200</td>
      <td>Sent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Sun, 19 Nov 2017 14:28:18 +0100</td>
      <td>Important,Inbox</td>
    </tr>
  </tbody>
</table>
</div>



My Gmail data contains mostly messages in English; however, there is plenty of Spanish and German, which introduce special characters that can lead to encoding issues. For this reason, it is better to encode special characters such as `ñ` and letters with accents and umlauts


```python
def remove_accents(text):
    text = str(text)
    nfkd_norm = unicodedata.normalize('NFKD', text)
    text = nfkd_norm.encode('ASCII', 'ignore').decode('utf-8')
    return text
```


```python
df['subject'] = df['subject'].map(remove_accents)
```

After cleaning the `subject` field, we can get a general overview of the integrity of different fields


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 69988 entries, 0 to 69987
    Data columns (total 5 columns):
    subject    66828 non-null object
    from       69988 non-null object
    to         55762 non-null object
    date       56735 non-null object
    label      69604 non-null object
    dtypes: object(5)
    memory usage: 2.7+ MB


We find that `date`, one of the most relevant fields, contains many null entries (mostly from chat entries)


```python
df[df['date'].isnull()][['subject', 'to', 'date']].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>to</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Chat</td>
    </tr>
    <tr>
      <th>26</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Chat</td>
    </tr>
    <tr>
      <th>90</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Chat</td>
    </tr>
    <tr>
      <th>91</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Chat</td>
    </tr>
    <tr>
      <th>92</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Chat</td>
    </tr>
  </tbody>
</table>
</div>



The total number of null entries is


```python
len(df[df['date'].isnull()])
```




    56735



All these entries can be removed


```python
# delete null rows
df = df[df['date'].notnull()]
```


```python
len(df)
```




    56735



We can now focus on the `date` field. The next goal is to transform the class type: dates are given as strings


```python
df[['date']].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tue, 8 Aug 2017 11:20:05 +0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tue, 08 Aug 2017 10:35:39 +0100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tue, 08 Aug 2017 10:43:30 +0100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, 8 Aug 2017 13:16:37 +0200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun, 19 Nov 2017 14:28:18 +0100</td>
    </tr>
  </tbody>
</table>
</div>



These string dates can be converted into timestamps using the converted available for dataframes


```python
df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, errors='coerce', utc=True))
```

Some dates have unappropriate shape for conversion (these are drafts of spam messages), which can be simply removed


```python
df = df[df['date'].notnull()]
```

Given that the date is now a timestamp, messages can be easily sorted by date, after which the dataframe index must be reset


```python
df = df.sort_values(['date'], ascending=False)
df = df.reset_index(drop=True)
```

The most recent messages are the following


```python
df[['subject', 'date', 'label']].head(8)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Your Google data archive is ready</td>
      <td>2018-01-27 18:30:32+00:00</td>
      <td>Important,Inbox</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>2018-01-27 17:48:17+00:00</td>
      <td>Important,Jan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2018-01-27 17:17:36+00:00</td>
      <td>Drafts</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>2018-01-27 10:21:42+00:00</td>
      <td>Unread,old_stuff/_KIT,Spam</td>
    </tr>
    <tr>
      <th>4</th>
      <td>easy loan to obtain</td>
      <td>2018-01-26 20:36:15+00:00</td>
      <td>Unread,Spam</td>
    </tr>
    <tr>
      <th>5</th>
      <td>=?utf-8?Q?The=20Support=20Your=20Student=20Bod...</td>
      <td>2018-01-26 20:24:02+00:00</td>
      <td>Unread,Spam</td>
    </tr>
    <tr>
      <th>6</th>
      <td>None</td>
      <td>2018-01-26 18:59:24+00:00</td>
      <td>Important,Jan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>aantonop: "Bitcoin Q&amp;A: Layered scaling and pr...</td>
      <td>2018-01-26 17:47:20+00:00</td>
      <td>Trash</td>
    </tr>
  </tbody>
</table>
</div>



where the most recent message is the notification from Gmail to download the data used here. The oldest messages are


```python
df[['subject', 'date', 'label']].tail(9)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51986</th>
      <td>Re: Asamblea</td>
      <td>2004-10-05 14:21:47+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51987</th>
      <td>Asamblea</td>
      <td>2004-10-05 12:55:58+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51988</th>
      <td>None</td>
      <td>2004-10-01 22:06:10+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51989</th>
      <td>NaN</td>
      <td>2004-09-29 18:24:55+00:00</td>
      <td>Sent</td>
    </tr>
    <tr>
      <th>51990</th>
      <td>Web del teste para el cambio (fwd)</td>
      <td>2004-09-27 19:43:40+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51991</th>
      <td>=?ISO-8859-1?B?SW5mb3JtYWNp824=?=\n\tSegunda E...</td>
      <td>2004-09-27 16:42:58+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51992</th>
      <td>de dayton</td>
      <td>2004-09-14 23:29:03+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51993</th>
      <td>Escuela del CERN</td>
      <td>2004-09-14 14:58:38+00:00</td>
      <td>Inbox</td>
    </tr>
    <tr>
      <th>51994</th>
      <td>NaN</td>
      <td>2004-09-10 17:03:41+00:00</td>
      <td>Inbox</td>
    </tr>
  </tbody>
</table>
</div>



Finally, there are many messages in the `Drafts` folder that should also be removed


```python
df = df[df['label'] != 'Drafts']
```

The same applies for `Spam` messages. Unfortunately, this label does not appear alone so it must be searched in the `label` column


```python
cnt = 0
idx_to_remove = []
for i, lab in enumerate(df['label']):
    if 'Spam' in str(lab):
        idx_to_remove.append(i)
        
df = df.drop(df.index[idx_to_remove])
df = df.reset_index(drop=True)
```

At this point, and given the time used for cleaning the data file, it is a good idea to export it as a `csv` file for future use without the need of redoing the preprocessing above.


```python
df.to_csv('data/gmail_data_preprocessed.csv', 
          encoding='utf-8', index=False)
```

## Data exploration

We can now begin exploring the data set.


```python
df = pd.read_csv('data/gmail_data_preprocessed.csv')
len(df)
```




    51423



Since the data was loaded from a `csv` file, the dates are back as `str` so they must be converted into `timestamp` again


```python
df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
```

### 1. Incoming vs. outgoing messages
For simplicity, all messages written by me or sent to me can be labeled by the string `me` instead of my email address. This will make the identification of incoming and outgoing emails easier. For this the following helper function returns `me` my email address is found and leaves the text unchanged, otherwise:


```python
def rename_me(txt):
    txt = str(txt).lower()
    if('jsdiaz' in txt or
       'jorge.diaz' in txt):
        txt_out = 'me'
    else:
        txt_out = txt
    return txt_out
```


```python
df['from'] = df['from'].apply(rename_me)
df['to'] = df['to'].apply(rename_me)
```


```python
df[['subject', 'to', 'date', 'label']].head(4)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>to</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Your Google data archive is ready</td>
      <td>me</td>
      <td>2018-01-27 18:30:32</td>
      <td>Important,Inbox</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>me</td>
      <td>2018-01-27 17:48:17</td>
      <td>Important,Jan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>me</td>
      <td>2018-01-26 18:59:24</td>
      <td>Important,Jan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aantonop: "Bitcoin Q&amp;A: Layered scaling and pr...</td>
      <td>me</td>
      <td>2018-01-26 17:47:20</td>
      <td>Trash</td>
    </tr>
  </tbody>
</table>
</div>



Since we want to explore the statistical distribution of messages, a useful information is a count of messages, for which a unit `count` column can be created


```python
df['count'] = [1 for _ in range(len(df))]
```

In order to keep the original data intact, we cam make a copy and set the timestamp as a index, so that messages can be grouped and resampled by time periods


```python
data = df.copy()
data.set_index('date', drop=True, inplace=True)
```

Now we can identify incoming vs. outgoing emails 


```python
data_in = data[data['to'] == 'me']
data_out = data[data['from'] == 'me']
```


```python
monthly_in = data_in['count'].resample('M').sum()
monthly_out= data_out['count'].resample('M').sum()
monthly_in.plot(color='g', label='incoming emails')
monthly_out.plot(color='r', label='outgoing emails')
plt.ylabel('Monthly email count')
plt.legend(loc='lower right', frameon=True).get_frame().set_color('white');
```


![png](Gmail_Analysis_files/Gmail_Analysis_54_0.png)


It can be seen that most of the time the number of received emails is greater than the number of emails sent. This trend appears only to be reversed in late 2013


```python
monthly_in.plot(color='g', label='incoming emails')
monthly_out.plot(color='r', label='outgoing emails')
plt.ylabel('Monthly email count')
plt.axis(['2013-01-27', '2014-01-27', 0, 350])
plt.legend(loc='best', frameon=True).get_frame().set_color('white');
```


![png](Gmail_Analysis_files/Gmail_Analysis_56_0.png)


More precisely during the month of Sep 2013 I sent significantly more emails than I received. Interestingly, this coincides with the potsdoc application period for the next year. Hence, the spike shows the many emails that I wrote regarding postdoc applications for 2014.

### 2. Busy days

We can now try to identify email activity vs. day of the week. We use the `timestamp` method `weekday()`, which returns an index $\in [0,\cdots, 6]$ corresponding to the days of the from Monday to Sunday.


```python
dow = []
for i in range(len(df)):
    dow.append(df['date'][i].weekday())
```

Create new `series` with the day of the week of the message


```python
df['dow'] = dow
df[['subject', 'date', 'dow', 'label']].head(4)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>date</th>
      <th>dow</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Your Google data archive is ready</td>
      <td>2018-01-27 18:30:32</td>
      <td>5</td>
      <td>Important,Inbox</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>2018-01-27 17:48:17</td>
      <td>5</td>
      <td>Important,Jan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>2018-01-26 18:59:24</td>
      <td>4</td>
      <td>Important,Jan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aantonop: "Bitcoin Q&amp;A: Layered scaling and pr...</td>
      <td>2018-01-26 17:47:20</td>
      <td>4</td>
      <td>Trash</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_in = df[df['to'] == 'me']
df_out = df[df['from'] == 'me']
```

We can now get the distribution of messages per day of the week. For this a dictionary can easily capture the frequency of messages on each day


```python
dow_in, dow_out = {}, {}
for i in range(7):
    dow_in[i] = 0
    dow_out[i] = 0
for i in df_in['dow']:
    dow_in[i] += 1
for i in df_out['dow']:
    dow_out[i] += 1
```


```python
x, y_in, y_out, y_all = [], [], [], []
for key in dow_in.keys():
    x.append(key)
    y_in.append(dow_in[key])
    y_out.append(dow_out[key])
    y_all.append(dow_in[key] + dow_out[key])

plt.plot(x, y_in, 'o-', color='g', label='incoming emails')
plt.plot(x, y_out, 'o-', color='r', label='outgoing emails')
plt.plot(x, y_all, 'o-', color='k', label='all emails')
plt.axis([-0.5, 6.5, 0, 10000])
plt.xlabel('day of the week')
plt.ylabel('number of messages')
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.xticks(x, days)
plt.legend(frameon=True).get_frame().set_color('white');
```


![png](Gmail_Analysis_files/Gmail_Analysis_65_0.png)


This plot shows clearly that the activity is quite uniform during the week days and it decreases during the weekens, as sort of expected.

### 3. Frequent contacts
Another immediate question that can be answered with this data is regarding the most frequent contacts I have received messages from and those to whom I have written the most. Just as done earlier with my email address that was replaced by the string `me`, the privacy of my contacts (I prefer not to post their email address) can be protected by creating a function that can replace their contact information with a nickname. Moreover, some of my contacts might have used different email addresses. For example, if I received emails from Richard Feynman (no, I didn't), his contact information can appear in a variety of forms:
- richard.feynman@qedmail.com
- "Feynman, Richard Phillips" richard.feynman@qedmail.com
- "Richard Feynman" richard.feynman@qedmail.com
- feynman_rules@qedmail.com

Even though they look different, they refer to the same contact. They can should all be replaced by the same nickname. Thsi can be achieved with the following function:


```python
def nickname(txt, list_to_rename, new_name):
    txt = str(txt)
    for name in list_to_rename:
        if name in txt.lower():
            txt_out = new_name
            break
        else:
            txt_out = txt
    return txt_out
```

where `txt` is the text where the names on the list `list_to_rename` are been searched, if found they will be replaced by the nickname `new_name`. In the example above, the nickname for Feynman would be obtained in the form:
`df['from'] = df['from'].apply(lambda x: nickname(x, ['Feynman'], 'Richard F.'))`
As an explicit example, emails from amazon.com and amazon.de can be grouped together by using


```python
amazon = ['amazon.de', 'amazon.com']
df['from'] = df['from'].apply(lambda x: nickname(x, amazon, 'Amazon.com'))
```

This is done for a few frequent contacts. The explicit lists are omitted for privacy. I have also included a list of my collaborators so they can all be represented by a unique nickname.


```python
df['to'] = df['to'].apply(lambda x: nickname(x, jsd, 'me'))
df['to'] = df['to'].apply(lambda x: nickname(x, mjc, 'Cote'))
df['to'] = df['to'].apply(lambda x: nickname(x, vak, 'Alan K.'))
df['to'] = df['to'].apply(lambda x: nickname(x, coll, 'collaborators'))
df['to'] = df['to'].apply(lambda x: nickname(x, juan, 'Juan M.'))
df['to'] = df['to'].apply(lambda x: nickname(x, crm, 'Cristian M.'))
df['to'] = df['to'].apply(lambda x: nickname(x, rob, 'Roberto L.'))
```

The most frequent contacts that I have written to are


```python
df['to'].value_counts()[1:7]
```




    Cote             5334
    Alan K.          1688
    collaborators    1130
    Juan M.           579
    Cristian M.       512
    Roberto L.        422
    Name: to, dtype: int64



This shows `nan`


```python
df = df[df['to'] != 'nan']
df = df.reset_index(drop=True)
```


```python
df['to'].value_counts()[1:7]
```




    Cote             5334
    Alan K.          1688
    collaborators    1130
    Juan M.           579
    Cristian M.       512
    Roberto L.        422
    Name: to, dtype: int64



This shows that after my PhD advisor (Alan K.), I have mostly written to my collaborators.
Similarly, for received messages we have


```python
df['from'] = df['from'].apply(lambda x: nickname(x, jsd, 'me'))
df['from'] = df['from'].apply(lambda x: nickname(x, mjc, 'Cote'))
df['from'] = df['from'].apply(lambda x: nickname(x, vak, 'Alan K.'))
df['from'] = df['from'].apply(lambda x: nickname(x, coll, 'collaborators'))

df['from'] = df['from'].apply(lambda x: nickname(x, juan, 'Juan M.'))
df['from'] = df['from'].apply(lambda x: nickname(x, crm, 'Cristian M.'))
df['from'] = df['from'].apply(lambda x: nickname(x, rob, 'Roberto L.'))
```

where the most frequent contacts that have written to me are


```python
df['from'].value_counts()[1:8]
```




    Cote             5999
    Alan K.          1733
    collaborators    1360
    Amazon.com       1202
    Cristian M.       668
    Juan M.           508
    Roberto L.        461
    Name: from, dtype: int64



This shows that after my PhD advisor (Alan K.), I have mostly received emails from my collaborators, followed closely from Amazon. All these results can be visualized


```python
df['from'].value_counts()[1:8].plot(kind='barh', color='g', alpha=0.6).invert_yaxis()
plt.title('emails from');
```


![png](Gmail_Analysis_files/Gmail_Analysis_83_0.png)



```python
df['to'].value_counts()[1:7].plot(kind='barh', color='r', alpha=0.6).invert_yaxis()
plt.title('emails to');
```


![png](Gmail_Analysis_files/Gmail_Analysis_84_0.png)


### 4. Frequent contacts: deep dive
The exploration of the contacts that I most frequently interact with shows some clear appearance of my Ph.D. advisor and my research collaborators. In this section I would like to dive deeper into these two contacts to see if more interesting features show up.


```python
name = 'collaborators' #'Alan K.'
mess_from = df[['date']][df['from'] == name]
mess_to = df[['date']][df['to'] == name]
```


```python
mess_from.set_index('date', drop=False, inplace=True)
mess_to.set_index('date', drop=False, inplace=True)
```


```python
counts_to = [1 for _ in range(len(mess_to))]
counts_from = [1 for _ in range(len(mess_from))]
mess_to['counts'] = counts_to
mess_from['counts'] = counts_from
```


```python
del mess_from['date']
del mess_to['date']
```


```python
b = mess_from.resample('W').apply({'score':'count'})
c = mess_to.resample('W').apply({'score':'count'})
b.plot(color='g')
plt.title('messages from {}'.format(name))
plt.ylabel('number')
plt.xlabel('year');

c.plot(color='r')
plt.title('messages to {}'.format(name))
plt.ylabel('number')
plt.xlabel('year');
```

    C:\Users\jsdiaz\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: using a dict with renaming is deprecated and will be removed in a future version
      """Entry point for launching an IPython kernel.
    C:\Users\jsdiaz\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: using a dict with renaming is deprecated and will be removed in a future version
      



![png](Gmail_Analysis_files/Gmail_Analysis_90_1.png)



![png](Gmail_Analysis_files/Gmail_Analysis_90_2.png)



```python
b['2012-01-01 00:00:00+00:00':'2017-01-01 00:00:00+00:00'].plot(color='g')
plt.title('messages from {}'.format(name))
plt.ylabel('number')
plt.xlabel('year');
```


![png](Gmail_Analysis_files/Gmail_Analysis_91_0.png)



```python
from collections import Counter
```


```python
all_wrds = []
for wrds in list(df['subject'][df['to'] == 'me']):
    all_wrds.extend(str(wrds).lower().split())
```


```python
all_wrds[:6]
```




    ['your', 'google', 'data', 'archive', 'is', 'ready']




```python
from nltk.corpus import stopwords
stopwords_en = set(stopwords.words('english'))
stopwords_es = set(stopwords.words('spanish'))
all_stopwords = stopwords_en | stopwords_es | set(my_stopwords)
```


```python
my_stopwords = ['re:', 'nan', 'none', '-', 'fwd:', 'fw:', '&', 'hola', 'order', 'amazon.com']
```


```python
len(all_wrds)
```




    59475




```python
all_wrds = [wrd for wrd in all_wrds if wrd not in all_stopwords]
```


```python
len(all_wrds)
```




    58393




```python
a = Counter(all_wrds)
```


```python
a.most_common(20)
```




    [('question', 296),
     ('paper', 277),
     ('new', 275),
     ('physics', 263),
     ('lorentz', 242),
     ('consulta', 241),
     ('meeting', 218),
     ('saludos', 211),
     ('library', 199),
     ('update', 172),
     ('application', 167),
     ('reminder', 167),
     ('report', 165),
     ('ready', 162),
     ('amazon.de', 161),
     ('jorge', 160),
     ('diaz', 160),
     ('confirmation', 157),
     ('violation', 155),
     ('pregunta', 151)]


