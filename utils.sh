#!/bin/bash

# Helper function to exit script on failed command
exit_on_fail() {
    local err_msg=$1
    local success_msg=$2
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $err_msg ]]; then
            echo -e "\n [FATAL] exit_on_fail: $err_msg"
        fi
        echo -e "[FATAL] exit_on_fail: failed!"
        exit 1
    else
        if [[ ! -z $success_msg ]]; then
            echo -e "[INFO] $success_msg"
        fi
    fi
}

check_env_variables () {
    echo "---- Checking for environment variables ----"
    for env_var in "$@"; do
        local echo_in='echo $env_var' 
        local echo_out="echo \$$(eval $echo_in)"
        local env_val=`eval $echo_out`
        if [[ -z "$env_val" ]]; then
            echo "[FATAL] check_env_variables: environment variable $env_var is not defined, abort"
            exit
        else
            echo "[INFO] check_env_variables: found $env_var ($env_val)"
        fi
    done
    echo -e "---- Done! ----"\n
}

copy_src () {
    local src_dir=$1
    local dst_dir=$2
    local name=$3
    local c_ext=$4
    local force=$5

    # Check whether source file already exists in local folder
    if [[ $force -ne 1 && -f "$dst_dir/$name.c" ]]; then
        echo "[INFO] copy_src: Source exists"
        return 0
    fi

    # Check whether source files exist in the remote folder
    if [[ ! -f "$src_dir/$name.$c_ext" || ! -f "$src_dir/$name.h"  ]]; then
        echo "[ERROR] copy_src: Failed to find benchmark"
        return 1
    fi

    # Copy code files (C/CPP + H) from source to destination folder
    mkdir -p "$dst_dir"
    cp "$src_dir/$name.$c_ext" "$dst_dir/$name.c"
    cp "$src_dir/$name.h" "$dst_dir/$name.h"
    return 0
}