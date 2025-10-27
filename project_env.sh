#!/bin/bash

CURRENT_DIR=$(pwd)

export PROJECT_ROOT=$CURRENT_DIR

croot() {
    cd $PROJECT_ROOT
}

calc_eer_ASV5() {
    score=$1
    cd eval_package/ASV5
    python evaluate_asvspoof5.py $CURRENT_DIR/$1 track1
    cd -
}
