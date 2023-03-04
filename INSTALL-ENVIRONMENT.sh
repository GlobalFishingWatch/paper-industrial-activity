#!/usr/bin/env bash

set -euo pipefail


# check pre-requisates programs
check_installed(){
    failmsg=${2:-"please install [${1}] with: brew install $1"}
    which $1 > /dev/null|| (echo ${failmsg}; exit 1)
}

check_installed yq
check_installed jq
check_installed conda "please install [conda] or [miniconda] as described on the website"

# check file dependencies
check_file_exists(){
    if [ ! -f ${1} ]; then
        echo "file [${BASE_ENV_FILE}] must exist";
        exit 1;
    fi
}

BASE_ENV_FILE=${1:-radenv.yaml}
check_file_exists ${BASE_ENV_FILE}
check_file_exists setup.py
check_file_exists setup.cfg

# functions supporting datetime operations on conda environments and files
get_conda_env(){
    conda env list --json | jq -r --arg env_name "/$1" '.envs[] | select(endswith($env_name))'
}

get_conda_date(){
    local env_prefix=$(get_conda_env ${1})
    if [ -z "${env_prefix}" ]; then
        echo "0"; # return zero seconds for datetime if the environment does not exist
    else
        local env_hist="${env_prefix}/conda-meta/history";
        local env_date=$(date -j -f "==> %Y-%m-%d %H:%M:%S <==" "$(head -n1 $env_hist)" "+%s");
        echo ${env_date};
    fi
}


get_file_date(){
    date -r ${1} "+%s"
}

env_older_than_file(){
    local env_file=${1}
    local env_name=${2:-$(cat ${env_file} | yq '.name')}

    local ENV_FILE_DATE=$(get_file_date ${env_file})
    local ENV_DATE=$(get_conda_date ${env_name})

    if [ $ENV_FILE_DATE -ge $ENV_DATE ]; then
        echo "environment [${env_name}] older than describing file [${env_file}]"
        return 0
    else
        echo "environment [${env_name}] is ready"
        return 1
    fi
}

# create base environment if out of date
BASE_ENV=$(cat ${BASE_ENV_FILE} | yq '.name')
if env_older_than_file ${BASE_ENV_FILE} ${BASE_ENV}; then
    echo "rebuilding this environment may be unnescisary if you have an up to date [${BASE_ENV}] from another repository with the same dependencies"
    read -r -p "would your like to rebuild base environment from [${BASE_ENV_FILE}]? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            conda env remove -n ${BASE_ENV}
            conda env create -f ${BASE_ENV_FILE}
            INVALIDATE_CHILDREN=1
            ;;
        *)
            ;;
    esac
fi


build_derived_local_environment(){
    local base_env=$1
    local work_env=$2
    conda env remove -n ${work_env}
    conda create --name ${work_env} --clone ${base_env}    # clone `radenv` so that it doesn't get corrupted by current project
    $(get_conda_env ${work_env})/bin/python -m pip install --editable .[all]  # install local `setup.cfg` in an editable way in new environment
    $(get_conda_env ${work_env})/bin/python -m pre_commit install  # install pre-commit

}


WORK_ENV=$(python setup.py --name)
if [ -n "${INVALIDATE_CHILDREN:-}" ]; then
    build_derived_local_environment $BASE_ENV $WORK_ENV
else
    if env_older_than_file setup.cfg ${WORK_ENV}; then
        read -r -p "would your like to rebuild working environment named [${WORK_ENV}] from [setup.cfg]? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                build_derived_local_environment $BASE_ENV $WORK_ENV
                ;;
            *)
                ;;
        esac
    fi
fi

echo "to activate: conda activate $WORK_ENV"
