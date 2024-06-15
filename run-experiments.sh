#!/usr/bin/env sh

MAINCMD="python hmtest/main.py ml train --cuda --seed 0"
BASECMD="${MAINCMD} data/meta-images-split.csv data/png runs"


${BASECMD} --fusion output --lfabnorm 0 no_abnorm_fusion_output
${BASECMD} --fusion output --lfabnorm 1 abnorm_fusion_output
${BASECMD} --fusion mean-feats --lfabnorm 0 no_abnorm_fusion_mean
${BASECMD} --fusion max-feats --lfabnorm 0 no_abnorm_fusion_max
${BASECMD} --fusion mean-feats --lfabnorm 1 abnorm_fusion_mean
${BASECMD} --fusion max-feats --lfabnorm 1 abnorm_fusion_max
