if [ "$1" = "translate" ]; then
    MAIN="parallel-dataset/pnp.py --model_path /gpfsstore/rech/hkt/uoi78rt/geo/ --cache /gpfsscratch/rech/hkt/uoi78rt/geo-cache --batch_size 10 --save_dir /gpfsscratch/rech/hkt/uoi78rt/parallel-new/ --base_path /gpfsscratch/rech/hkt/uoi78rt/base"
    CONTEXT="-t 20 --ngpu 1 --ncpu 8 --module-load pytorch-gpu/py3/2.0.1 --gb 80 --ram m --email y.siglidis@gmail.com --array 0-10"

    python ${MAIN} --make_submission
    spython ${MAIN} --k_start 0 --k_end 250 --idx_start \${SLURM_ARRAY_TASK_ID} : ${CONTEXT} --tag parallel-1
    spython ${MAIN} --k_start 250 --k_end 500 --idx_start \${SLURM_ARRAY_TASK_ID} : ${CONTEXT} --tag parallel-2
    spython ${MAIN} --k_start 500 --k_end 750 --idx_start \${SLURM_ARRAY_TASK_ID} : ${CONTEXT} --tag parallel-3
    spython ${MAIN} --k_start 750 --k_end 1000 --idx_start \${SLURM_ARRAY_TASK_ID} : ${CONTEXT} --tag parallel-4
else
    MAIN_PATH="${SCRATCH}/results-parallel"
    FIGURE_MAIN_PATH="${MAIN_PATH}/figures"
    CACHE_MAIN_PATH="${MAIN_PATH}/cache"
    TYPICALITY_MAIN_PATH="${MAIN_PATH}/typicality"
    SUBMISSION_MAIN_PATH="${MAIN_PATH}/submission"
    HTML_MAIN_PATH="${MAIN_PATH}/html"

    WHICH_FEATURE="dift-161"
    CLUSTER_SCRIPT_PATH="parallel-dataset/cluster.py"
    COMPUTE_SCRIPT_PATH="parallel-dataset/compute.py"
    HTML_SCRIPT="parallel/make-html.py"

    TYPICALITY="${TYPICALITY_MAIN_PATH}-0.1-0.7"
    CACHE_PATH="${CACHE_MAIN_PATH}"
    CACHE_PATH_NC="${CACHE_MAIN_PATH}"
    SUBMISSION_PATH="${SUBMISSION_MAIN_PATH}"
    FIGURE_PATH="${FIGURE_MAIN_PATH}"

    MODEL_PATH="/gpfsstore/rech/hkt/uoi78rt/geo/"
    DATA_PATH="/gpfsscratch/rech/hkt/uoi78rt/parallel-new/"

    if [ "$1" = "compute" ]; then
        spython ${COMPUTE_SCRIPT_PATH} --typicality_path ${TYPICALITY} -i ${DATA_PATH} -s ${SUBMISSION_PATH} --split_id \${SLURM_ARRAY_TASK_ID} --model_path ${MODEL_PATH} --t_min 0.1 --t_max 0.7 : -t 20 --ngpu 1 --ncpu 8 --module-load pytorch-gpu/py3/2.0.1 --tag parallel --gb 80 --ram m --email y.siglidis@gmail.com --array 0-20
    else;
        spython ${CLUSTER_SCRIPT_PATH} --typicality_path ${TYPICALITY} --cache_path ${CACHE_PATH} --model_path ${MODEL_PATH} --k 64 -d ${DATA_PATH} --cluster --feature_which dift-161 --num_clusters 32 --figure_path ${FIGURE_PATH} --feature_which ${WHICH_FEATURE} --max_row 20 --top_k_figure 32 --min_row 0 : -t 10 --ngpu 1 --ncpu 8 --module-load pytorch-gpu/py3/1.11.0 --tag parallel-cluster --gb 80 --ram m --email y.siglidis@gmail.com --debug
done

