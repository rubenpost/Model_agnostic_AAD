# %%
import sys
import time
sys.path.insert(1,"/workspaces/thesis/compliance_analysis")
import logging
import tempfile
import image
import math
import pm4py as pm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from pm4py.visualization.common import save
from aad_experiment.common.utils import configure_logger
from aad_experiment.aad.aad_globals import (
    AAD_IFOREST, IFOR_SCORE_TYPE_NEG_PATH_LEN, HST_LOG_SCORE_TYPE, AAD_HSTREES, RSF_SCORE_TYPE,
    AAD_RSFOREST, INIT_UNIF, AAD_CONSTRAINT_TAU_INSTANCE, QUERY_DETERMINISIC, ENSEMBLE_SCORE_LINEAR,
    get_aad_command_args, AadOpts
)
from aad_experiment.aad.aad_support import get_aad_model
from aad_experiment.aad.forest_aad_detector import is_forest_detector
from aad_experiment.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, \
    BayesianRulesetsDescriber, get_region_memberships
from aad_experiment.aad.query_model import Query
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
# %%
logger = logging.getLogger(__name__)

def get_debug_args(budget=10, detector_type=AAD_IFOREST):
    # return the AAD parameters what will be parsed later
    return ["--resultsdir=./temp", "--randseed=42",
            "--reruns=1",
            "--detector_type=%d" % detector_type,
            "--forest_score_type=%d" %
            (IFOR_SCORE_TYPE_NEG_PATH_LEN if detector_type == AAD_IFOREST
            else HST_LOG_SCORE_TYPE if detector_type == AAD_HSTREES
            else RSF_SCORE_TYPE if detector_type == AAD_RSFOREST else 0),
            "--init=%d" % INIT_UNIF,  # initial weights
            "--withprior", "--unifprior",  # use an (adaptive) uniform prior
            # ensure that scores of labeled anomalies are higher than tau-ranked instance,
            # while scores of nominals are lower
            "--constrainttype=%d" % AAD_CONSTRAINT_TAU_INSTANCE,
            "--querytype=%d" % QUERY_DETERMINISIC,  # query strategy
            "--num_query_batch=1",  # number of queries per iteration
            "--budget=%d" % budget,  # total number of queries
            "--tau=0.03",
            # normalize is NOT required in general.
            # Especially, NEVER normalize if detector_type is anything other than AAD_IFOREST
            # "--norm_unit",
            "--forest_n_trees=100", "--forest_n_samples=256",
            "--forest_max_depth=%d" % (100 if detector_type == AAD_IFOREST else 7),
            # leaf-only is preferable, else computationally and memory expensive
            "--forest_add_leaf_nodes_only",
            "--ensemble_score=%d" % ENSEMBLE_SCORE_LINEAR,
            # "--bayesian_rules",
            "--resultsdir=./temp",
            "--log_file=./temp/demo_aad.log",
            "--debug"]

def detect_anomalies(x, df):
    start = time.time()
    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args  
    args = get_aad_command_args(debug=True, debug_args=get_debug_args())
    configure_logger(args)
    opts = AadOpts(args)
    logger.debug(opts.str_opts())
    np.random.seed(opts.randseed)

    # rng = np.random.RandomState(opts.randseed)

    # prepare the AAD model
    model = get_aad_model(x, opts, 0)
    model.fit(x)
    model.init_weights(init_type=opts.init)

    # get the transformed data which will be used for actual score computations
    x_transformed = model.transform_to_ensemble_features(x, dense=False, norm_unit=opts.norm_unit)

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1

    qstate = Query.get_initial_query_state(opts.qtype, opts=opts, budget=opts.budget)
    queried = []  # labeled instances
    ha = []  # labeled anomaly instances
    hn = []  # labeled nominal instances
    while len(queried) < opts.budget:
        ordered_idxs, anom_score = model.order_by_score(x_transformed)
        original_scores = ordered_idxs
        qx = qstate.get_next_query(ordered_indexes=ordered_idxs,
                                queried_items=queried)
        queried.extend(qx)
        for xi in qx:
            show_anomaly(xi, df)
            while True:
                y_labeled[xi] = input("Is the trace an anomaly? 1 for yes, 0 for no:")
                if y_labeled[xi] not in (1, 0):
                    print("Not an appropriate choice. Please enter 1 for yes, 0 for no.")
                else:
                    break
            if y_labeled[xi] == 1:
                ha.append(xi)
            else:
                hn.append(xi)

        # incorporate feedback and adjust ensemble weights
        model.update_weights(x_transformed, y_labeled, ha=ha, hn=hn, opts=opts, tau_score=opts.tau)

        # most query strategies (including QUERY_DETERMINISIC) do not have anything
        # in update_query_state(), but it might be good to call this just in case...
        qstate.update_query_state()

    # the number of anomalies discovered within the budget while incorporating feedback
    found = np.sum(y_labeled[queried])
    end = time.time()
    print("You have identified %s anomalies. In total, AAD took" % found, end - start, "seconds.")

    return ordered_idxs, model, x_transformed, queried, y_labeled, original_scores

def show_anomaly(queried, df, index=None):

    # Grap queried case from population
    average_cancel = df.data.groupby(['case:concept:name'])['average_cancellation'].mean()
    average_cancel_total = average_cancel.value_counts()[0]    
    average_cancel = average_cancel.mean()
    case_number = len(df.data.groupby(['case:concept:name']))
    df.num_cols.drop(['average_cancellation', 'average_resource'], axis=1, inplace=True)

    average_resource = df.data.groupby(['case:concept:name'])['average_resource'].mean()
    resource_number = average_resource
    average_resource = round(average_resource.mean())

    gp = df.data.groupby('case:concept:name')
    queried_case = gp.get_group(df.data['case:concept:name'].unique()[queried])

    # Create event log from queried case
    log = pm.convert_to_event_log(queried_case)

    # Remove boolean columns from dataframe as visualization would not provide benefit
    for column in df.num_cols.columns:
        if df.num_cols[column].nunique() <= 2:
            df.num_cols.drop(column, axis=1, inplace=True)
    
    # Set seaborn style, subplot size, and initiate position number
    sns.set(style="white", color_codes=True, font_scale = 1)
    sns.despine(left=True)
    fig, ax = plt.subplots(len(df.num_cols.columns)+3, figsize=(10, (len(df.num_cols.columns)+3)*10))
    position = 0

    # Visualize process trace
    dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
    gviz = dfg_visualization.apply(dfg, log, variant=dfg_visualization.Variants.PERFORMANCE)
    file_name = tempfile.NamedTemporaryFile(suffix='.png')
    file_name.close()
    save.save(gviz, file_name.name)
    img = mpimg.imread(file_name.name)
    ax[position].axis('off')
    ax[position].imshow(img)
    ax[position].set_title(f"Process model of case {queried}")
    position += 1

    ax[position].imshow(np.asarray(queried_case[['bounded_existence_O_ACCEPTED', 'four_eye_principle_O_CREATED_O_ACCEPTED']][0:1]).transpose(), cmap='Reds')
    ax[position].set_yticks(np.arange(2))
    ax[position].set_xticks(np.arange(1))
    ax[position].set_yticklabels(['Bounded existence (O_ACCEPTED)', 'Seperation of duties (O_CREATED and O_ACCEPTED)'])
    ax[position].set_xticklabels(['Does the case comply?'])
    ax[position].set_title("Tested compliance rules", y=1.02)

    annotation = queried_case[['bounded_existence_O_ACCEPTED', 'four_eye_principle_O_CREATED_O_ACCEPTED']][0:1]

    if annotation.iloc[0,0] == 0:
        annotation.iloc[0,0] = 'No violation noted'
    else:
        annotation.iloc[0,0] = 'Violation noted'

    if annotation.iloc[0,1] == 0:
        annotation.iloc[0,1] = 'No violation noted'
    else:
        annotation.iloc[0,1] = 'Violation noted'

    if annotation.iloc[0,0] == 'No violation noted':
        text = ax[position].text(0, 0, annotation.iloc[0,0], ha="center", va="center", color="b")
    else:
        text = ax[position].text(0, 1, annotation.iloc[0,1], ha="center", va="center", color="w")

    
    if annotation.iloc[0,1] == 'No violation noted':
        text = ax[position].text(0, 1, annotation.iloc[0,1], ha="center", va="center", color="b")
    else:
        text = ax[position].text(0, 1, annotation.iloc[0,1], ha="center", va="center", color="w")

    position += 1

    # Create input for ancedent/consequence table
    queried_case.reset_index(drop=True, inplace=True)
    empty_list = pd.DataFrame(columns=['activity', 'consequence'])
    for activity in queried_case['concept:name'].unique():
        occurences = np.where(queried_case['concept:name'] == activity)[0]
        for consequence_index in occurences:
            if consequence_index != len(queried_case['concept:name'])-1:
                consequence = queried_case['concept:name'].loc[consequence_index+1]
                empty_list = empty_list.append({'activity': activity, 'consequence': consequence}, ignore_index=True)
            else:
                pass
    activity_list = empty_list
    empty_list = empty_list.groupby(['activity', 'consequence']).size()
    empty_list = pd.DataFrame(empty_list).reset_index()
    empty_list = empty_list.pivot_table(index='activity', columns='consequence', values=0, fill_value=0).reset_index()
    empty_list.drop(['activity'], axis=1, inplace=True)

    # Create custom table
    # activities = sorted(list(activity_list['activity'].unique()))
    activities_y = sorted(list(activity_list['activity'].unique()))
    activities_x = sorted(list(activity_list['consequence'].unique()))
    ax[position].imshow(np.asarray(empty_list), cmap='Reds')
    ax[position].set_xticks(np.arange(len(activities_x)))
    ax[position].set_yticks(np.arange(len(activities_y)))
    ax[position].set_xticklabels(activities_x)
    ax[position].set_yticklabels(activities_y)
    
    cancel_queried = queried_case['concept:name'].value_counts()
    cancel_queried = cancel_queried['O_CANCELLED']
    percentage_cancel = cancel_queried / case_number
    title = "Antecedent (Y-xis) and consequence activities (X-axis). \n Out of all {} cases, {} have cancellations. This case is cancelled {} times, which happends in {}% of the cases.".format(case_number, average_cancel_total, cancel_queried, str(round(percentage_cancel, 3)))
    ax[position].set_title(title, y=1.02)
    plt.setp(ax[position].get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
       
    # Loop over data dimensions and create text annotations.
    for i in range(len(empty_list)):
        for j in range(len(empty_list.columns)):
            if np.asarray(empty_list)[i,j] == 0:
                pass
            else:
                text = ax[position].text(j, i, np.asarray(empty_list)[i,j],
                        ha="center", va="center", color="w")
    position += 1

    # Create input for research/activity table
    table_input = pd.DataFrame(queried_case[['concept:name','org:resource']].value_counts().reset_index())
    table_input = table_input.pivot_table(index='concept:name', columns='org:resource', values=0, fill_value=0).reset_index()
    table_input.drop(['concept:name'], axis=1, inplace=True)
    table_input = np.asarray(table_input)

    # Create custom table
    activities, resources = sorted(list(queried_case['concept:name'].unique())), list(queried_case['org:resource'].unique())
    ax[position].imshow(table_input, cmap='Reds')
    ax[position].set_xticks(np.arange(len(resources)))
    ax[position].set_yticks(np.arange(len(activities)))
    ax[position].set_xticklabels(resources)
    ax[position].set_yticklabels(activities)
    resource_queried = queried_case['org:resource'].nunique()
    resource_number = resource_number.value_counts()
    resource_number = resource_number[resource_queried]
    resource_percenteage = resource_number / case_number

    title = "Activities performed by resources. \n On average, a case is performed by {} resources. This case was performed by {} resources. \n {}% of the cases are performed by {} resources.".format(average_resource, resource_queried, str(round(resource_percenteage, 3)), resource_queried)
    ax[position].set_title(title, y=1.02)
    plt.setp(ax[position].get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(activities)):
        for j in range(len(resources)):
            if table_input[i, j] == 0:
                pass
            else:
                text = ax[position].text(j, i, table_input[i, j],
                            ha="center", va="center", color="w")
    position += 1

    plt.tight_layout()

    # Create dataframe used to make base plots
    data = pd.concat([df.num_cols, df.case_id_col], axis=1)
    plot_data = data.dropna(subset=df.num_cols.columns).groupby(['case:concept:name']).max().reset_index()

    # Loop over every variable we want to visualize
    for variable in df.num_cols.drop(['activity_count'], axis=1).columns:

        # Plot the variable
        sns.histplot(plot_data[variable], color="mistyrose", ax=ax[position], bins=15)

        # Get variable value
        variable_value = queried_case[variable].max()

        # Get bar information and highlight bar in which the variable value is found
        patches = ax[position].patches
        bin_width = patches[0].get_width()
        bin_number = round(variable_value / bin_width)
        bin_number -= 1
        patches[bin_number].set_facecolor('r')
        ax[position].axvline(x=bin_width*bin_number, ymax=0, color='r')

        # Adjust arrow in plot annotation to fit variable
        if (plot_data[variable].value_counts().max()/patches[bin_number].get_height() <= 1) or (bin_number == 0):
            text = "\u2190 " + variable_value.astype(str)
        else:
            text = "\u2193" + variable_value.astype(str)

        # Create placement of annotation in plot to fit variable
        heights = [h.get_height() for h in patches]
        if (plot_data[variable].value_counts().max()/patches[bin_number].get_height() <= 1) or (bin_number == 0): 
            bin_number += 1
            variable_value += bin_width
        if variable == 'OfferedAmount':
            text_y = (patches[bin_number].get_height()/max(heights))+0.02
            text_x = (bin_width*bin_number)+(bin_width*1.5)              
        else:
            text_y = (patches[bin_number].get_height()/max(heights))+0.02
            text_x = (bin_width*bin_number)+bin_width/2
        plt.text(text_x, text_y, text, transform=ax[position].get_xaxis_transform())

        # Adjust labels
        handle1 = mpatches.Patch(color='r', label='Sample')
        handle2 = mpatches.Patch(color='mistyrose', label='Population')
        ax[position].legend(handles=[handle1, handle2])

        # Create the figure

        ax[position].set_ylabel('Count')
        ax[position].xaxis.labelpad = 20
        ax[position].yaxis.labelpad = 20

        position += 1
        if index != None:
            plt.savefig('/workspaces/thesis/vis/{}_2012/{}_{}.jpg'.format(index, index, queried), bbox_inches='tight')



    return plt.show()