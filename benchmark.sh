#!/bin/bash

if [[ -z $1 ]];
then
    WORKINGDIR="build/tests"
else
    WORKINGDIR=$1
fi

${WORKINGDIR}/bGemm --benchmark_out=bGemm.json --benchmark_out_format=json --benchmark_time_unit=s
${WORKINGDIR}/bGemmSweep --benchmark_out=bGemmSweep.json --benchmark_out_format=json --benchmark_time_unit=s

