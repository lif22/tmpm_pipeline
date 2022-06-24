import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from stages.TM.textmining import CasesList

def getEvaluationList(p1_id, p2_id, casesList):
    evaluation_list_p1 = pd.DataFrame(columns=["datestr", "duration_delta", "msg_delta", "headcount_delta"])
    evaluation_list_p2 = pd.DataFrame(columns=["datestr", "duration_delta", "msg_delta", "headcount_delta"])

    # Only select subset of successful cases
    subcaseList = CasesList()
    subcaseList.extend([c for c in casesList if c.checkCaseSuccess()])

    # median case duration
    median_case_duration = subcaseList.getMedianCaseDuration(timest_format="%Y-%m-%d %H:%M:%S")
    # median message count between cases
    median_msg_count = subcaseList.getMedianCaseMessageCount()
    # median # of people involved
    median_head_count = subcaseList.getMedianCaseHeadcount()

    for case in subcaseList:
        if p1_id not in case.actors and p2_id not in case.actors:
            # only check cases in the responsiblity of investigated HR practitioners
            continue
        start, end, caseduration = case.getCaseDuration(timest_format="%Y-%m-%d %H:%M:%S")
        if p2_id in case.actors:
            print(case.messages[0].from_, start, end)
        messagecount = case.getMessageCount()
        headcount = case.getHeadCount()
        if p1_id in case.actors:
            evaluation_list_p1.loc[len(evaluation_list_p1)] = [end.date().strftime("%Y-%m-%d"), (median_case_duration-caseduration)/(60*60), (median_msg_count-messagecount), (median_head_count-headcount)]
        elif p2_id in case.actors:
            evaluation_list_p2.loc[len(evaluation_list_p2)] = [end.date().strftime("%Y-%m-%d"), (median_case_duration-caseduration)/(60*60), (median_msg_count-messagecount), (median_head_count-headcount)]

    # reformat datestrings
    evaluation_list_p1.index = pd.to_datetime(evaluation_list_p1["datestr"], format="%Y-%m-%d")
    evaluation_list_p1.drop(columns=["datestr"], inplace=True)
    evaluation_list_p2.index = pd.to_datetime(evaluation_list_p2["datestr"], format="%Y-%m-%d")
    evaluation_list_p2.drop(columns=["datestr"], inplace=True)
    # sort by date indices
    evaluation_list_p1.sort_index(inplace=True)
    evaluation_list_p2.sort_index(inplace=True)
    evaluation_list_p1["score"] = False
    evaluation_list_p2["score"] = False
    for idx, val in evaluation_list_p1.iterrows():
        evaluation_list_p1.loc[idx,"score"] = 0.025*val["duration_delta"]+0.65*val["msg_delta"]+0.2*val["headcount_delta"]
    for idx, val in evaluation_list_p2.iterrows():
        evaluation_list_p2.loc[idx,"score"] = 0.025*val["duration_delta"]+0.65*val["msg_delta"]+0.2*val["headcount_delta"]

    # Normalize score column to 0 <= x <= 1
    normmax = max(evaluation_list_p1["score"].max(), evaluation_list_p2["score"].max())
    normmin = min(evaluation_list_p1["score"].min(), evaluation_list_p2["score"].min())
    evaluation_list_p1["score"] = (evaluation_list_p1["score"]-normmin)/(normmax-normmin)
    evaluation_list_p2["score"] = (evaluation_list_p2["score"]-normmin)/(normmax-normmin)

    return evaluation_list_p1, evaluation_list_p2

def preparePlot():
    pass

def objective(x, a, b, c):
    return a * x + b * x**2 + c

def objective_cube(x,a,b,c,d):
    return a*x + b*x**2 + c*x**3+d

def fitParameters(evaluation_list_p1, evaluation_list_p2):
    # Reformat x-Axis from dt object to int (days)
    start = evaluation_list_p1.index[0]
    indices = evaluation_list_p1.index
    new_i = []
    for idx, val in evaluation_list_p1.iterrows():
        i = (idx-start).days
        new_i.append(i)
    evaluation_list_p1.index = new_i

    start = evaluation_list_p2.index[0]
    indices = evaluation_list_p2.index
    new_i = []
    for idx, val in evaluation_list_p2.iterrows():
        i = (idx-start).days
        new_i.append(i)
    evaluation_list_p2.index = new_i

    # curve fit
    popt, _ = curve_fit(objective, evaluation_list_p1.index, evaluation_list_p1["score"])
    popt2,_ = curve_fit(objective_cube, evaluation_list_p1.index, evaluation_list_p1["score"])
    # summarize the parameter values
    a1, b1, c1  = popt
    a1c, b1c, c1c, d1c = popt2
    popt,_ = curve_fit(objective, evaluation_list_p2.index, evaluation_list_p2["score"])
    popt2,_ = curve_fit(objective_cube, evaluation_list_p2.index, evaluation_list_p2["score"])

    a2,b2,c2 = popt
    a2c,b2c,c2c,d2c = popt2
    return a1, a2, b1, b2, c1, c2, a1c, a2c, b1c, b2c, c1c, c2c, d1c, d2c

def fitAndPlot(evaluation_list_p1, evaluation_list_p2, a1, a2, b1, b2, c1, c2, a1c, a2c, b1c, b2c, c1c, c2c, d1c, d2c):
    figure(figsize=(8,6), dpi=80)
    # Employee 1
    x1 = evaluation_list_p1.index
    y1 = evaluation_list_p1["score"]
    plt.plot(x1,y1, '-', color='b')
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x1), max(x1)+100, 1)
    # calculate the output for the range
    y_line = objective(x_line, a1, b1, c1)
    y_line2 = objective_cube(x_line, a1c, b1c, c1c, d1c)
    # create a line plot for the mapping function
    #plt.plot(x_line, y_line, ':', color='b')
    plt.plot(x_line, y_line2, ':', color='b')

    # Employee 2
    x2 = evaluation_list_p2.index
    y2 = evaluation_list_p2["score"]
    plt.plot(x2,y2, '-', color='r')
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x2), max(x2)+200, 1)
    # calculate the output for the range
    y_line = objective(x_line, a2, b2, c2)
    y_line2 = objective_cube(x_line, a2c, b2c, c2c, d2c)

    # create a line plot for the mapping function
    #plt.plot(x_line, y_line, ':', color='r')
    plt.plot(x_line, y_line2, ':', color='r')
    plt.xlabel("Days")
    plt.ylabel("$m_i$")
    plt.legend(['Jessica Parker (hist.)', 'Jessica Parker (extrap.)', 'Johanna Nielsen (hist.)', 'Johanna Nielsen (extrap.)'])