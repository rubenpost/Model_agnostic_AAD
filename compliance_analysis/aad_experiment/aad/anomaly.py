# %%
import sys
import time
sys.path.insert(1,"/workspaces/thesis/compliance_analysis")
import logging
import tempfile
import image
import pm4py as pm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
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

    rng = np.random.RandomState(opts.randseed)

    # prepare the AAD model
    model = get_aad_model(x, opts, rng)
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
    print("You have identified %s anomalies. In total, AAD took", end - start, "seconds." % found)



#     generate compact descriptions for the detected anomalies
#     ridxs_counts, region_extents = None, None
#     if len(ha) > 0:
#         ridxs_counts, region_extents = describe_instances(x, np.array(ha), model=model,
#                                                         opts=opts, interpretable=True)
#         logger.debug("selected region indexes and corresponding instance counts (among %d):\n%s" %
#                     (len(ha), str(list(ridxs_counts))))
#         logger.debug("region_extents: these are of the form [{feature_index: (feature range), ...}, ...]\n%s" %
#                     (str(region_extents)))
#     return ordered_idxs, model, x_transformed, queried #, ridxs_counts, region_extents

# def describe_instances(x, instance_indexes, model, opts, interpretable=False):
#     """ Generates compact descriptions for the input instances

#     :param x: np.ndarray
#         The instance matrix with ALL instances
#     :param instance_indexes: np.array(dtype=int)
#         Indexes for the instances which need to be described
#     :param model: Aad
#         Trained Aad model
#     :param opts: AadOpts
#     :return: tuple, list(map)
#         tuple: (region indexes, #instances among instance_indexes that fall in the region)
#         list(map): list of region extents where each region extent is a
#             map {feature index: feature range}
#     """
#     if not is_forest_detector(opts.detector_type):
#         raise ValueError("Descriptions only supported by forest-based detectors")

#     # setup dummy y
#     y = np.zeros(x.shape[0], dtype=np.int32)
#     y[instance_indexes] = 1

#     if interpretable:
#         if opts.bayesian_rules:
#             # use BayesianRulesetsDescriber to get compact and [human] interpretable rules
#             describer = BayesianRulesetsDescriber(x, y=y, model=model, opts=opts)
#         else:
#             # use CompactDescriber to get compact and [human] interpretable rules
#             describer = CompactDescriber(x, y=y, model=model, opts=opts)
#     else:
#         # use MinimumVolumeCoverDescriber to get simply compact (minimum volume) rules
#         describer = MinimumVolumeCoverDescriber(x, y=y, model=model, opts=opts)

#     selected_region_idxs, desc_regions, rules = describer.describe(instance_indexes)

#     _, memberships = get_region_memberships(x, model, instance_indexes, selected_region_idxs)
#     instances_in_each_region = np.sum(memberships, axis=0)
#     if len(instance_indexes) < np.sum(instances_in_each_region):
#         logger.debug("\nNote: len instance_indexes (%d) < sum of instances_in_each_region (%d)\n"
#                     "because some regions overlap and cover the same instance(s)." %
#                     (len(instance_indexes), int(np.sum(instances_in_each_region))))

#     if rules is not None:
#         rule_details = []
#         for rule in rules:
#             rule_details.append("%s: %d/%d instances" % (str(rule),
#                                                         len(rule.where_satisfied(x[instance_indexes])),
#                                                         len(instance_indexes)))
#         logger.debug("Rules:\n  %s" % "\n  ".join(rule_details))

#     return zip(selected_region_idxs, instances_in_each_region), desc_regions

def show_anomaly(queried, df):
    # Grap queried case from population
    gp = df.groupby('case:concept:name')
    queried_case = gp.get_group(df['case:concept:name'].iloc[queried])
    log = pm.convert_to_event_log(queried_case)
    
    sns.set(style="white", palette="muted", color_codes=True, font_scale = 1)
    fig, ax = plt.subplots(len(df.select_dtypes(['float','int64']).columns)+2, figsize=(10, (len(df.select_dtypes(['float','int64']).columns)+2)*10))
    sns.despine(left=True)
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
    position += 1
    # Plot a filled kernel density estimate

    sample = pd.DataFrame(queried_case[['concept:name','org:resource']].value_counts().reset_index())
    sample = pd.DataFrame(pd.concat([sample['concept:name'], pd.get_dummies(sample['org:resource'])], axis=1).value_counts()).reset_index()
    user_columns = sample.iloc[:,1:-1]
    table_input = sample['concept:name'].drop_duplicates().reset_index(drop=True)
    for user in user_columns.columns:
        user_data = sample.groupby(['concept:name'], as_index=False).agg({user:'sum'})
        table_input = pd.concat([table_input, user_data.iloc[:,1:]], axis=1)
    table_input = np.asarray(table_input.iloc[:,1:])
    activities, resources = list(queried_case['concept:name'].unique()), list(queried_case['org:resource'].unique())
    ax[position].imshow(table_input, cmap='Blues')
    # We want to show all ticks...
    ax[position].set_xticks(np.arange(len(resources)))
    ax[position].set_yticks(np.arange(len(activities)))
    # ... and label them with the respective list entries
    ax[position].set_xticklabels(resources)
    ax[position].set_yticklabels(activities)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax[8].get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(activities)):
        for j in range(len(resources)):
            text = ax[position].text(j, i, table_input[i, j],
                        ha="center", va="center", color="w")
    ax[position].set_title("Activities performed per resource", y=1.02)
    position += 1

    for variable in df.select_dtypes(['float','int64']).columns:
        sns.distplot(df[variable], hist = False, color="orange", kde_kws={"shade": True}, ax=ax[position])
        ax[position].axvline(queried_case[variable].max(), color='royalblue')
        ax[position].legend(labels=['Sample','Population'])
        trans = ax[position].get_xaxis_transform()
        text = "  \u27f5  "  + queried_case[variable].max().astype(str)
        plt.text(queried_case[variable].max(), 0.75, text, transform=trans)
        position += 1
    
    plt.savefig('testplot.pdf')

    return plt.show()
# %%
