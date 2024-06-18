#!/usr/bin/env sh

TRAINCMD="python hmtest/main.py ml train --cuda --seed 0"
TRAINARGS="data/meta-images-split.csv data/png runs"


${TRAINCMD} --fusion output --lfabnorm 0 no_abnorm_fusion_output ${TRAINARGS}
${TRAINCMD} --fusion output --lfabnorm 1 abnorm_fusion_output ${TRAINARGS}
${TRAINCMD} --fusion mean-feats --lfabnorm 0 no_abnorm_fusion_mean ${TRAINARGS}
${TRAINCMD} --fusion max-feats --lfabnorm 0 no_abnorm_fusion_max ${TRAINARGS}
${TRAINCMD} --fusion mean-feats --lfabnorm 1 abnorm_fusion_mean ${TRAINARGS}
${TRAINCMD} --fusion max-feats --lfabnorm 1 abnorm_fusion_max ${TRAINARGS}
