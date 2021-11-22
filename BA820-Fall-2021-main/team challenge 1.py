import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

from sklearn.feature_extraction.text import CountVectorizer
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import os

PROJECT = "ba820-kl"

SQL = """
    select * from `questrom.SMSspam.train`
"""

sms = pd.read_gbq(SQL, PROJECT)
sms.message = sms.message.str.lower()

cv = CountVectorizer()
model = cv.fit(sms.message)
sms_cv = model.transform(sms.message)
y = sms.label
X = pd.DataFrame(sms_cv.toarray())
LR = LogisticRegression().fit(X, y)


SQL_test = """
    select * from `questrom.SMSspam.test`
"""

sms_test = pd.read_gbq(SQL_test, PROJECT)
sms_test.message = sms_test.message.str.lower()
test_cv = model.transform(sms_test.message)
result = LR.predict(test_cv)
df = pd.DataFrame({'label':result,'id':sms_test.id})
df.to_csv('D:/Desktop/result.csv',index=False)


