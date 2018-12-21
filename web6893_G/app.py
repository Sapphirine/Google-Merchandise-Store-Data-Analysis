from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort, make_response, url_for
import os
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response
import flask_login
import logging
from flask_login import LoginManager
import flask_logger
import numpy as np

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import numpy
# import torch.nn
# import torch.nn.functional
# from torch.utils.data import Dataset, DataLoader
# import torch.optim
import numpy.random
import numpy as np
import math
import pandas as pd
import copy
import time
from sklearn.preprocessing import LabelEncoder
# import torch.nn
# import torch.nn.functional
# import torch.optim
import numpy.random
import math
import pandas
from sklearn.preprocessing import LabelEncoder


tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('welcome.html')

@app.route('/map', methods=['GET', 'POST'])
def map():
    return render_template('Worldmap.html')


@app.route('/post',methods=['GET'])
def post():
    return render_template('input.html')

@app.route('/do_post',methods=['GET', 'POST'])
def do_post():
    l1 = request.form['hits']
    l2 = request.form['sessionQualityDim']
    l3 = request.form['pageviews']
    l4 = request.form['bounces']
    l5 = request.form['newVisits']
    l6 = request.form['deviceCategory']
    l7 = request.form['customDimensions']
    l8 = request.form['channelGrouping']
    # l9=5.492112545288729
    # a1 = -8.296643e-02
    # a2 = 4.003274e-02
    # a3 = 2.092876e-01
    # a4 = 3.389896e-01
    # a5 = -1.771933e-01
    # a6 = -6.642596e-02
    # a7 = 2.526242e-02
    # a8 = 1.725804e-02
    device={'desktop':0,'tablet':2,'mobile':1}
    custom={'North America':4,'APAC':1,'Central America':2,'EMEA':3,'South America':5}
    channel={'Referral':6,'Organic Search':4,'Social':7,'Paid Search':5,'Affiliates':1,'Direct':2,'Display':3,'Other':0}
    if l6:
        l6=device[l6]
    if l7:
        l7=custom[l7]
    if l8:
        l8=channel[l8]
    
    # target=float(l9)+float(l1)*float(a1)+float(l2)*float(a2)+float(l3)*float(a3)+float(l4)*float(a4)+float(l5)*float(a5)+float(l6)*float(a6)+float(l7)*float(a7)+float(l8)*float(a8)
    iput=[float(l1),float(l2),float(l3),float(l4),float(l5),float(l6),float(l7),float(l8)]
    # out = np.exp(target)
    # output=(out-1)/(10**6)
    bst = lgb.Booster(model_file='model.txt')   #load the generated model
    lst=bst.predict(data=[iput,iput])
    output=lst[0]

    cont = dict(data = output)
    return render_template('return.html',**cont)




if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True, host='0.0.0.0', port=8112)