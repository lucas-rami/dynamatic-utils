#!/bin/bash

# ===- utils.sh - Utilities for bash scripting -----------------*- Bash -*-=== #
# 
# This script is meant to be sourced by other scripts and performs nothing on
# its own. It contains bash functions that are useful for pretty-printing, error
# handling, and performing common tasks when using Dynamatic.
# 
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Pretty-printing
# ===----------------------------------------------------------------------=== #

# Prints some information to stdout.
#   $1: the text to print
echo_info() {
    echo "[INFO] $1"
}

# Prints a warning message to stdout.
#   $1: the text to print
echo_warning() {
    echo "[WARN] $1"
}

# Prints a (potentially non-fatal) error message to stdout.
#   $1: the text to print
echo_error() {
    echo "[ERROR] $1"
}

# Prints a fatal error message to stdout.
#   $1: the text to print
echo_fatal() {
    echo "[FATAL] $1"
}

# Print a large section text to stdout.
#   $1: the section's title
echo_section() {
    echo "# ===----------------------------------------------------------------------=== #"
    echo "# $1"
    echo "# ===----------------------------------------------------------------------=== #"
}

# Prints a small subsection text to stdout.
#   $1: the subsection's title
echo_subsection() {
    echo "# ===--- $1 ---==="
}

# ===----------------------------------------------------------------------=== #
# Error-handling
# ===----------------------------------------------------------------------=== #

# Prints an error message or (optional) information message depending on an
# integer value (error message when the value is non-zero, information message
# otherwise).
#   $1: integer value
#   $2: error message
#   $3: [optional] information message
echo_status_arg() {
    local ret=$1
    if [[ $ret -ne 0 ]]; then
        echo_error "$2"
    else
        if [[ ! -z $3 ]]; then
            echo_info "$3"
        fi
    fi
    return $ret
}

# Prints an error message or (optional) an information message depending on
# the return value of the last command that was called before this function
# (error message when the value is non-zero, information message otherwise).
#   $1: error message
#   $2: [optional] information message
echo_status() {
    echo_status_arg $? "$1" "$2"
    return $?
}

# Exits the script with a fatal error message if the last command that was
# called before this function failed, otherwise optionally prints an information
# message.
#   $1: fatal error message
#   $2: [optional] information message
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo_fatal "$1"
            exit 1
        fi
        echo_fatal "Failed!"
        exit 1
    else
        if [[ ! -z $2 ]]; then
            echo_info "$2"
        fi
    fi
}

# ===----------------------------------------------------------------------=== #
# Dynamatic tasks
# ===----------------------------------------------------------------------=== #

# Checks whether environment variables currently exist. Exits the script if at
# least one environment variable does not exist.
#   $@: names of environment variables to check for 
check_env_variables () {
    for env_var in "$@"; do
        local echo_in='echo $env_var' 
        local echo_out="echo \$$(eval $echo_in)"
        local env_val=`eval $echo_out`
        if [[ -z "$env_val" ]]; then
            echo_fatal "Environment variable $env_var is not defined, abort"
            exit 1
        fi
    done
}

# Copies a benchmark's source code (source file + header files) to a new
# location, potentially changing the extension of the source file to .c.
#   $1: source directory
#   $2: destination directory
#   $3: name of the benchmark
#   $4: extension of the source file
#   $5: boolean to decide whether to perform the copy even if the file already
#       exists at the destination
copy_src () {
    local src_dir=$1
    local dst_dir=$2
    local name=$3
    local c_ext=$4
    local force=$5

    # Check whether source file already exists in local folder
    if [[ $force -eq 0 && -f "$dst_dir/$name.c" ]]; then
        echo_info "Source exists"
        return 0
    fi

    # Check whether source files exist in the remote folder
    if [[ ! -f "$src_dir/$name.$c_ext" || ! -f "$src_dir/$name.h"  ]]; then
        echo_error "Failed to find benchmark"
        return 1
    fi

    # Copy code files (C/CPP + H) from source to destination folder
    mkdir -p "$dst_dir"
    cp "$src_dir/$name.$c_ext" "$dst_dir/$name.c"
    cp "$src_dir/$name.h" "$dst_dir/$name.h"
    echo_info "Source copied $src_dir -> $dst_dir"
    return 0
}