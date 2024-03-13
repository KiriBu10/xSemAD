
import os
import pm4py
from copy import deepcopy
import random
from pm4py.objects.log.obj import Event, Trace, EventLog

def _get_event_classes(log):
    classes = set()
    for trace in log:
        for event in trace:
            classes.add(event["concept:name"])
    return classes

def _remove_event(trace: Trace):
    del_index = random.randint(0, len(trace) - 1)
    trace2 = Trace()
    for i in range(0, len(trace)):
        if i != del_index:
            trace2.append(trace[i])
    return trace2

def _insert_event(trace: Trace, tasks):
    ins_index = random.randint(0, len(trace))
    task = random.choice(list(tasks))
    e = Event()
    e["concept:name"] = task
    trace.insert(ins_index, e)
    return trace

def _swap_events(trace: Trace):
    if len(trace) == 1:
        return trace
    indices = list(range(len(trace)))
    index1 = random.choice(indices)
    indices.remove(index1)
    index2 = random.choice(indices)
    trace2 = Trace()
    for i in range(len(trace)):
        if i == index1:
            trace2.append(trace[index2])
        elif i == index2:
            trace2.append(trace[index1])
        else:
            trace2.append(trace[i])
    return trace2

def insert_noise(log, noisy_trace_prob, noisy_event_prob, log_size):
    if len(log) < log_size:
        # add additional traces until desired log size reached
        log_cpy = EventLog()
        for i in range(0, log_size):
            log_cpy.append(deepcopy(log[i % len(log)]))
        log = log_cpy
    classes = _get_event_classes(log)
    log_new = EventLog()
    for trace in log:
        if len(trace) > 0:
            trace_cpy = deepcopy(trace)
            # check if trace makes random selection
            if random.random() <= noisy_trace_prob:
                insert_more_noise = True
                while insert_more_noise:
                    # randomly select which kind of noise to insert
                    noise_type = random.randint(0, 2)
                    if noise_type == 0:
                        _remove_event(trace_cpy)
                    if noise_type == 1:
                        _insert_event(trace_cpy, classes)
                    if noise_type == 2:
                        _swap_events(trace_cpy)
                    # flip coin to see if more noise will be inserted
                    insert_more_noise = (random.random() <= noisy_event_prob)
            log_new.append(trace_cpy)
    return log_new



def save_noisy_logs(input_dir, target_dir, noisy_trace_prob, noisy_event_prob, log_size):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    log_files = [f for f in os.listdir(input_dir) if f.endswith(".xes")]
    for log_file in log_files:
        df = pm4py.read_xes(os.path.join(input_dir, log_file))
        log = pm4py.convert_to_event_log(df)
        noisy_log = insert_noise(log, noisy_trace_prob, noisy_event_prob, log_size)
        noisy_xes_file = os.path.join(target_dir, log_file)
        pm4py.write_xes(noisy_log, noisy_xes_file)
        print(f"Saved as model {noisy_xes_file}")
    print('done.')




xes_dir_filtered='input/sap_sam_2022/filtered/test/logs/'
xes_dir_noisy='input/sap_sam_2022/filtered/test/noisy_logs/'
NOISY_TRACE_PROB= .7
NOISY_EVENT_PROB = .7
LOG_SIZE=1000
save_noisy_logs(input_dir=xes_dir_filtered, target_dir=xes_dir_noisy,
                noisy_trace_prob=NOISY_TRACE_PROB,
                noisy_event_prob=NOISY_EVENT_PROB,
                log_size=LOG_SIZE)