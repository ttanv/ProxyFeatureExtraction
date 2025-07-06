"""
Script modified to extract connection-level features [Hayes et al. USENIX 2016]
min limit pkt length: proxy 1000 normal 60

Note: dictionary_() will extract features; -1: INCOMING, 1: OUTGOING Tor Cell
Cell file format: "topk#time direction(-/+)cellsize"
"""

import math
import numpy as np
from itertools import chain



"""
Feeder functions
"""
def neighborhood(iterable):
    iterator = iter(iterable)
    prev = (0)
    item = next(iterator)  # throws StopIteration if empty.
    for nex in iterator:
        yield (prev,item,nex)
        prev = item
        item = nex
    yield (prev,item,None)

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0
  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out

"""
Non-feeder functions
"""
def get_pkt_list(trace_data):
    dta = []
    for record in trace_data:
        ts = float(record[2])
        src_ip = record[3]
        if src_ip == '10.0.2.16' or src_ip == '10.0.2.15':
            direction = 1
        else:
            direction = -1
        size = int(record[7])
        dta.append((ts, direction, size))
    return dta

def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] == -1:
            In.append(p)
        if p[1] == 1:
            Out.append(p)
    return In, Out

"""
Time features
"""
def inter_pkt_time(list_data):
    if not list_data:
        return []

    times = [x[0] for x in list_data if x]  # Safeguard against empty items

    if len(times) < 2:
        # print(f"Insufficient data for inter-packet times: {times}")
        return []

    return [next_elem - elem for elem, next_elem in zip(times, times[1:])]  # No circular wrap-around

def interarrival_times(list_data):
    In, Out = In_Out(list_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(list_data)
    return IN, OUT, TOTAL

def interarrival_maxminmeansd_stats(list_data):
    interstats = []
    In, Out, Total = interarrival_times(list_data)
    if In and Out:
        avg_in = sum(In)/float(len(In))
        avg_out = sum(Out)/float(len(Out))
        avg_total = sum(Total)/float(len(Total))
        interstats.append((max(In), max(Out), max(Total), avg_in, avg_out, avg_total, np.std(In), np.std(Out), np.std(Total), np.percentile(In, 75), np.percentile(Out, 75), np.percentile(Total, 75)))
    elif Out and not In:
        avg_out = sum(Out)/float(len(Out))
        avg_total = sum(Total)/float(len(Total))
        interstats.append((0, max(Out), max(Total), 0, avg_out, avg_total, 0, np.std(Out), np.std(Total), 0, np.percentile(Out, 75), np.percentile(Total, 75)))
    elif In and not Out:
        avg_in = sum(In)/float(len(In))
        avg_total = sum(Total)/float(len(Total))
        interstats.append((max(In), 0, max(Total), avg_in, 0, avg_total, np.std(In), 0, np.std(Total), np.percentile(In, 75), 0, np.percentile(Total, 75)))
    else:
        interstats.extend([0]*12) #where did 15 come from??
    return interstats

#n calculates percentiles of timestamps
def time_percentile_stats(list_data):
    #Total = get_pkt_list(trace_data)
    In, Out = In_Out(list_data)
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in list_data]
    if Total1:
        min_time = min(Total1)  # Earliest timestamp
        In1 = [t - min_time for t in In1]
        Out1 = [t - min_time for t in Out1]
        Total1 = [t - min_time for t in Total1]
    STATS = []
    if In1:
        STATS.append(np.percentile(In1, 25)) # return 25th percentile
        STATS.append(np.percentile(In1, 50))
        STATS.append(np.percentile(In1, 75))
        STATS.append(np.percentile(In1, 100))
    if not In1:
        STATS.extend(([0]*4))
    if Out1:
        STATS.append(np.percentile(Out1, 25)) # return 25th percentile
        STATS.append(np.percentile(Out1, 50))
        STATS.append(np.percentile(Out1, 75))
        STATS.append(np.percentile(Out1, 100))
    if not Out1:
        STATS.extend(([0]*4))
    if Total1:
        STATS.append(np.percentile(Total1, 25)) # return 25th percentile
        STATS.append(np.percentile(Total1, 50))
        STATS.append(np.percentile(Total1, 75))
        STATS.append(np.percentile(Total1, 100))
    if not Total1:
        STATS.extend(([0]*4))
    return STATS

def number_pkt_stats(list_data):
    #Total = get_pkt_list(trace_data)
    In, Out = In_Out(list_data)
    return len(In), len(Out), len(list_data)

def first_and_last_30_pkts_stats(list_data):
    #Total = get_pkt_list(trace_data)
    first30 = list_data[:30]
    last30 = list_data[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[1] == -1:
            first30in.append(p)
        if p[1] == 1:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[1] == -1:
            last30in.append(p)
        if p[1] == 1:
            last30out.append(p)
    stats= []
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats

#concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(list_data):
    #Total = get_pkt_list(trace_data)
    chunks= [list_data[x:x+20] for x in range(0, len(list_data), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] == 1:
                c+=1
        concentrations.append(c)
    return np.std(concentrations), sum(concentrations)/float(len(concentrations)), np.percentile(concentrations, 50), min(concentrations), max(concentrations), concentrations

#Average number packets sent and received per second
def number_per_sec(list_data):
    #Total = get_pkt_list(trace_data)
    last_time = list_data[-1][0]
    last_second = math.ceil(last_time)

    temp = []
    sec = [x for x in range(1, int(last_second)+1)]
    idx = 0
    sec_number = sec[idx]
    c = 0
    for p in list_data:
        if p[0] <= sec_number:
            c += 1
        else:
            temp.append(c)
            idx += 1
            sec_number = sec[idx]
            c = 1
    temp.append(c)
    avg_number_per_sec = sum(temp)/float(len(temp))
    return avg_number_per_sec, np.std(temp), np.percentile(temp, 50), min(temp), max(temp), temp

#Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
def avg_pkt_ordering_stats(list_data):
    #Total = get_pkt_list(trace_data)
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in list_data:
        if p[1] == -1:
            temp1.append(c1)
        c1+=1
        if p[1] == 1:
            temp2.append(c2)
        c2+=1
    avg_in = sum(temp1) / float(len(temp1)) if temp1 else 0
    avg_out = sum(temp2) / float(len(temp2)) if temp2 else 0

    return avg_in, avg_out, np.std(temp1), np.std(temp2)

def perc_inc_out(list_data):
    #Total = get_pkt_list(trace_data)
    In, Out = In_Out(list_data)
    percentage_in = len(In)/float(len(list_data))
    percentage_out = len(Out)/float(len(list_data))
    return percentage_in, percentage_out


"""
Size features
"""
def total_size(list_data):
    return sum([x[2] for x in list_data])

def in_out_size(list_data):
    In, Out = In_Out(list_data)
    size_in = sum([x[2] for x in In])
    size_out = sum([x[2] for x in Out])
    return size_in, size_out

def average_total_pkt_size(list_data):
    return np.mean([x[2] for x in list_data])

def average_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    average_size_in = np.mean([x[2] for x in In])
    average_size_out = np.mean([x[2] for x in Out])
    return average_size_in, average_size_out

def variance_total_pkt_size(list_data):
    return np.var([x[2] for x in list_data])

def variance_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    var_size_in = np.var([x[2] for x in In])
    var_size_out = np.var([x[2] for x in Out])
    return var_size_in, var_size_out

def std_total_pkt_size(list_data):
    return np.std([x[2] for x in list_data])

def std_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    std_size_in = np.std([x[2] for x in In])
    std_size_out = np.std([x[2] for x in Out])
    return std_size_in, std_size_out

def max_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    max_size_in = max([x[2] for x in In])
    max_size_out = max([x[2] for x in Out])
    return max_size_in, max_size_out

def unique_pkt_lengths(list_data):
    pass

"""
Feature function
"""
Features_names=["max_in", "max_out","max_total","avg_in","avg_out","avg_total","std_in","std_out","std_total","75th_percentile_in","75th_percentile_out","75th_percentile_total",
                "nb_pkts_in","nb_pkts_out","nb_pkts_total",
                "nb_pkts_in_f30","nb_pkts_out_f30","nb_pkts_in_l30","nb_pkts_out_l30","std_pkt_conc_out20", "avg_pkt_conc_out20","avg_per_sec","std_per_sec","avg_order_in",
                "avg_order_out","std_order_in","std_order_out","medconc","med_per_sec","min_per_sec","max_per_sec","maxconc","perc_in","perc_out","sum_altconc",
                "sum_alt_per_sec","sum_number_pkts","sum_intertimestats"]

def get_ft_labels(ALL_FEATURES):
    prev = 0
    next_l = 0
    # TIME Features
    ALL_FEATURES.extend(intertimestats)
    prev = len(ALL_FEATURES)
    # print("Inter packet time stats: ", 0, prev-1) #0-11


    prev = next_l
    ALL_FEATURES.extend(number_pkts)
    next_l = len(ALL_FEATURES)
    # print("Number of pkts: ", prev, next_l-1) 

    prev = next_l
    ALL_FEATURES.extend(thirtypkts)
    next = len(ALL_FEATURES)
    # print("Thirty packets stats: ", prev, next_l-1) 

    prev = next_l
    ALL_FEATURES.append(stdconc)
    next_l = len(ALL_FEATURES)
    # print("Std pkt conc: ", prev, next_l-1) 

    prev = next_l
    ALL_FEATURES.append(avgconc) #32
    next_l = len(ALL_FEATURES)
    # print("Avg pkt conc: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(avg_per_sec)
    next_l = len(ALL_FEATURES)
    # print("Avg per sec: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(std_per_sec)
    next_l = len(ALL_FEATURES)
    # print("Std per sec: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(avg_order_in)
    next_l = len(ALL_FEATURES)
    # print("avg order in: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(avg_order_out)
    next_l = len(ALL_FEATURES)
    # print("avg order out: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(std_order_in)
    next_l = len(ALL_FEATURES)
    # print("Std order in: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(std_order_out)
    next_l = len(ALL_FEATURES)
    # print("std order out: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(medconc)
    next_l = len(ALL_FEATURES)
    # print("medconc: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(med_per_sec)
    next_l = len(ALL_FEATURES)
    # print("med per sec: ", prev, next_l)

    prev = next_l
    ALL_FEATURES.append(min_per_sec)
    next_l = len(ALL_FEATURES)
    # print("min per sec: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(max_per_sec)
    next_l = len(ALL_FEATURES)
    # print("max per sec: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(maxconc)
    next_l = len(ALL_FEATURES)
    # print("max conc: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(perc_in)
    next_l = len(ALL_FEATURES)
    # print("% in: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(perc_out)
    next_l = len(ALL_FEATURES)
    # print("% out : ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.extend(alt_per_sec)
    next_l = len(ALL_FEATURES)
    # print("alt per sec: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(sum(altconc))
    next_l = len(ALL_FEATURES)
    # print("sum alt conc: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(sum(alt_per_sec))
    next_l = len(ALL_FEATURES)
    # print("sum alt per conc: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(sum(intertimestats))
    next_l = len(ALL_FEATURES)
    # print("sum inter time stats: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(sum(timestats))
    next_l = len(ALL_FEATURES)
    # print("sum time stats: ", prev, next_l-1)

    prev = next_l
    ALL_FEATURES.append(sum(number_pkts))
    next_l = len(ALL_FEATURES)
    # print("sum number of pkts: ", prev, next_l-1)

   
    return

# Function to count features and index mapping
def count_fts(lst_fts):
    fnames = ["intertimestats","timestats","number_pkts","thirtypkts","stdconc","avgconc","avg_per_sec","std_per_sec","avg_order_in","avg_order_out","std_order_in",
             "std_order_out","medconc","med_per_sec","min_per_sec","max_per_sec","maxconc","perc_in","perc_out","altconc","alt_per_sec","sum_altconc","sum_alt_per_sec","sum_intertimestats",
             "sum_timestats","sum_number_pkts"]
    ind = 0
    fts = []
    for x in range(0, len(lst_fts)):
        if isinstance(lst_fts[x][0], list):
            if ind == 0:
               start = ind
               ind += len(lst_fts[x][0])-1
            else:
               start = ind+1
               ind += len(lst_fts[x][0])
            #print(fnames[x], ":", start, "-", ind, ": ",len(lst_fts[x][0]))
            fts += lst_fts[x][0]
        else:
            ind += 1
            #print(fnames[x], ":", ind, ": 1")
            fts += [lst_fts[x][0]]
    #print("Feat list len: ", len(fts))


    return

#If size information available add them in to function below
def TOTAL_FEATURES(trace_data, max_size=150):

    list_data = get_pkt_list(trace_data)
    ALL_FEATURES = []

    intertimestats = [x for x in interarrival_maxminmeansd_stats(list_data)[0]]
    timestats = time_percentile_stats(list_data)
    number_pkts = list(number_pkt_stats(list_data))
    thirtypkts = first_and_last_30_pkts_stats(list_data)
    stdconc, avgconc, medconc, minconc, maxconc, conc = pkt_concentration_stats(list_data)
    avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec = number_per_sec(list_data)
    avg_order_in, avg_order_out, std_order_in, std_order_out = avg_pkt_ordering_stats(list_data)
    perc_in, perc_out = perc_inc_out(list_data)

    altconc = []
    alt_per_sec = []
    altconc = [sum(x) for x in chunkIt(conc, 20)]
    alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
    while len(altconc) < 20:
        altconc.append(0)
    if len(altconc) > 20:
        altconc = altconc[:20]
    while len(alt_per_sec) < 20:
        alt_per_sec.append(0)  
    if len(alt_per_sec) > 20:
        alt_per_sec = alt_per_sec[:20]

    ALL_FEATURES.extend(intertimestats)
    ALL_FEATURES.extend(number_pkts)
    ALL_FEATURES.extend(thirtypkts)
    ALL_FEATURES.append(stdconc)
    ALL_FEATURES.append(avgconc)
    ALL_FEATURES.append(avg_per_sec)
    ALL_FEATURES.append(std_per_sec)
    ALL_FEATURES.append(avg_order_in)
    ALL_FEATURES.append(avg_order_out)
    ALL_FEATURES.append(std_order_in)
    ALL_FEATURES.append(std_order_out)
    ALL_FEATURES.append(medconc)
    ALL_FEATURES.append(med_per_sec)
    ALL_FEATURES.append(min_per_sec)
    ALL_FEATURES.append(max_per_sec)
    ALL_FEATURES.append(maxconc)
    ALL_FEATURES.append(perc_in)
    ALL_FEATURES.append(perc_out)
    ALL_FEATURES.append(sum(altconc))
    ALL_FEATURES.append(sum(alt_per_sec))
    ALL_FEATURES.append(sum(number_pkts))
    ALL_FEATURES.append(sum(intertimestats))
    ALL_FEATURES.append(sum(timestats))
    ALL_FEATURES.extend(altconc) #20 fixed length
    ALL_FEATURES.extend(alt_per_sec) #20 fixed length
    ALL_FEATURES.extend(conc) # 60 fixed length


    #print("Extracted features: ", len(ALL_FEATURES))
    while len(ALL_FEATURES)<max_size:
        ALL_FEATURES.append(0)
    features = ALL_FEATURES[:max_size]
    #print("Length of features after truncation:", len(features))
    return features

def get_features(pkts, conn_name, limit):
    if len(pkts) >= limit:
        features = TOTAL_FEATURES(pkts)
        return features
    else:
        # print("The connection", conn_name, "only have", str(len(pkts)), "packets. Ignored.")
        return False

def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def checkequal(lst):
    return lst[1:] == lst[:-1]
