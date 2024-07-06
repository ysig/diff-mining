MAIN_PATH="results/$1"
FIGURE_MAIN_PATH="${MAIN_PATH}/figures"
CACHE_MAIN_PATH="${MAIN_PATH}/cache"
TYPICALITY_MAIN_PATH="${MAIN_PATH}/typicality"
SUBMISSION_MAIN_PATH="${MAIN_PATH}/submission"
HTML_MAIN_PATH="${MAIN_PATH}/html"

WHICH_FEATURE="dift-161"
CLUSTER_SCRIPT_PATH="diffmining/typicality/cluster.py"
COMPUTE_SCRIPT_PATH="diffmining/typicality/compute.py"
SEND_EMAIL="diffmining/typicality/email-client.py"
MODEL_PATH_SD="runwayml/stable-diffusion-v1-5"
HTML_SCRIPT="diffmining/typicality/make-html.py"
t_min="0.1"
t_max="0.7"

for mode in "pt"; do #"ft"
    TYPICALITY="${TYPICALITY_MAIN_PATH}/${mode}/${t_min}-${t_max}"
    CACHE_PATH="${CACHE_MAIN_PATH}/${mode}/${t_min}-${t_max}"
    SUBMISSION_PATH="${SUBMISSION_MAIN_PATH}/${mode}/${t_min}-${t_max}"
    FIGURE_PATH="${FIGURE_MAIN_PATH}/${mode}/${t_min}-${t_max}"
    GOOD="t"

    case $1 in
        ftt)
            echo "Running 'ftt'"
            MODEL_PATH="/home/isig/diff-geo-mining/gold/models/faces/ft/checkpoint-12610-export"
            DATA_PATH="dataset/ftt/train"
            k=50
            ;;
        cars)
            MODEL_PATH="/home/isig/diff-geo-mining/gold/models/cars/ft/export"
            DATA_PATH="dataset/cars/train"
            k=50
            ;;
        geo)
            echo "Running 'geo'"
            MODEL_PATH="/home/isig/diff-geo-mining/gold/models/geo/ft/checkpoint-64542-export"
            DATA_PATH="dataset/parallel-2"
            k=64
            ;;
        places)
            echo "Running 'places'"
            MODEL_PATH="/home/isig/diff-geo-mining/sd-places/checkpoint-112716/"
            DATA_PATH="/home/isig/diff-mining/places-val"
            k=64
            ;;
        *)
            echo "Invalid argument: $1. Please select 'ftt', 'cars', or 'geo'."
            GOOD="f"
            ;;
    esac

    if [ "${mode}" = "pt" ]; then
        MODEL_PATH="${MODEL_PATH_SD}"
    fi

    if [ "${GOOD}" = "t" ]; then
        python ${COMPUTE_SCRIPT_PATH} --typicality_path ${TYPICALITY} -i ${DATA_PATH} -m ${MODEL_PATH} --submission_path ${SUBMISSION_PATH} --make_submission --sub_split 1 --which $1 --t_min ${t_min} --t_max ${t_max}
        MAIN_COMMAND="python ${CLUSTER_SCRIPT_PATH} --which $1 --typicality_path ${TYPICALITY} --cache_path ${CACHE_PATH} --model_path ${MODEL_PATH} --k ${k} -d ${DATA_PATH} "
        ${MAIN_COMMAND} --cluster --feature_which ${WHICH_FEATURE} --topk #--recache

        ${MAIN_COMMAND} --figure_path ${FIGURE_PATH} --feature_which ${WHICH_FEATURE} --figures_only --max_row 20 --top_k_figure 32 --min_row 6 --topk
        ${MAIN_COMMAND} --figure_path ${FIGURE_PATH} --feature_which ${WHICH_FEATURE} --figures_only --max_row 7 --top_k_figure 6 --min_row 6 --topk
    fi
done
python ${HTML_SCRIPT} ${FIGURE_MAIN_PATH} ${HTML_MAIN_PATH}
