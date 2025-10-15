#!/bin/bash

CURRENT_DIR=$(pwd)

export PROJECT_ROOT=$CURRENT_DIR

croot() {
    cd $PROJECT_ROOT
}

calc_eer_19LA() {
    score=$1
    cd eval-package/19LA
    python evaluate.py $CURRENT_DIR/$1 z
    cd -
}

calc_eer_21LA() {
    score=$1
    cd eval-package/21
    python main.py --cm-score-file $CURRENT_DIR/$1 --track LA --subset eval
    cd -
}

calc_eer_21DF() {
    score=$1
    cd eval-package/21
    python main.py --cm-score-file $CURRENT_DIR/$1 --track DF --subset eval
    cd -
}

calc_eer_ITW() {
    score=$1
    cd eval-package/ITW
    python evaluate_in_the_wild.py $CURRENT_DIR/$1 ./keys/ eval
    cd -
}
