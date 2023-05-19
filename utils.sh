#!/bin/bash

# Helper function to exit script on failed command
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo "[FATAL] exit_on_fail: $1"
            exit 1
        fi
        echo -e "[FATAL] exit_on_fail: failed!"
        exit 1
    else
        if [[ ! -z $2 ]]; then
            echo "[INFO] $2"
        fi
    fi
}

# Helper function to print large section text
echo_section() {
    echo "# ===----------------------------------------------------------------------=== #"
    echo "# $1"
    echo "# ===----------------------------------------------------------------------=== #"
}

# Helper function to print status after command (returns the same value that was
# in $? when the function was called)
echo_status() {
    local ret=$?
    if [[ $ret -ne 0 ]]; then
        echo -e "[ERROR] $1"
    else
        if [[ ! -z $2 ]]; then
            echo -e "[INFO] $2"
        fi
    fi
    return $ret
}

# Helper function to print status depending on a return value passed as argument
# (returns the same value that was in $1)
echo_status_arg() {
    local ret=$1
    if [[ $ret -ne 0 ]]; then
        echo -e "[ERROR] $2"
    else
        if [[ ! -z $2 ]]; then
            echo -e "[INFO] $3"
        fi
    fi
    return $ret
}

check_env_variables () {
    echo_section "Checking environment variables"
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
    echo ""
}

copy_src () {
    local src_dir=$1
    local dst_dir=$2
    local name=$3
    local c_ext=$4
    local force=$5

    # Check whether source file already exists in local folder
    if [[ $force -eq 0 && -f "$dst_dir/$name.c" ]]; then
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
    echo "[INFO] copy_src: Source copied $src_dir -> $dst_dir"
    return 0
}