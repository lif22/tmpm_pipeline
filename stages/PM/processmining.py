import pm4py
import pandas as pd

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import evaluator as precision_evaluator
from pm4py.algo.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.algo.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.heuristics_net.defaults import DEPENDENCY_THRESH
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.vis import save_vis_heuristics_net
    

def importCSVToLog(filepath):
    event_log = pd.read_csv(filepath, sep=';')
    event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log, timest_format="%Y-%m-%d %H:%M:%S", timest_columns="Date")
    event_log = pm4py.format_dataframe(event_log, case_id='Case', activity_key='Action', timestamp_key='Date', timest_format="%Y-%m-%d %H:%M:%S")
    log = log_converter.apply(event_log)
    
    num_events = len(event_log)
    num_cases = len(event_log["Case"].unique())
    print(f"Imported {num_events} events with {num_cases} cases.")
    return log

def heuristicsMiner(log):
    net, im, fm = heuristics_miner.apply(
        log,
        parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.05,
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.MIN_ACT_COUNT: 4,
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.AND_MEASURE_THRESH: 0.95,
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.MIN_DFG_OCCURRENCES: 4,
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.DFG_PRE_CLEANING_NOISE_THRESH: 0.75
        }
    )
    return net, im, fm

def previewAndSave(log, net, im, fm):
    gviz = pn_visualizer.apply(net, im, fm, variant=pn_visualizer.Variants.FREQUENCY, log=log, parameters={"format": "svg"})
    gviz_vis = pn_visualizer.apply(net, im, fm, variant=pn_visualizer.Variants.FREQUENCY, log=log, parameters={"format": "png"})
    pn_visualizer.view(gviz_vis)
    pn_visualizer.save(gviz, "heu1.svg")

def computeMetrics(log, net, im, fm):
    fitness = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    prec = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    gen = generalization_evaluator.apply(log, net, im, fm)
    simp = simplicity_evaluator.apply(net)
    return fitness, prec, gen, simp

def previewAndSaveHeuristicsNet(log):
    parameters = {
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.MIN_DFG_OCCURRENCES: 3,
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.MIN_ACT_COUNT: 3,
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH.DFG_PRE_CLEANING_NOISE_THRESH: 0.1
    }
    heu_net = heuristics_miner.apply_heu(log, parameters=parameters)
    gviz = hn_visualizer.apply(heu_net, parameters={"format": "png"})
    hn_visualizer.view(gviz)
    save_vis_heuristics_net(heu_net, "heu1_net.png")