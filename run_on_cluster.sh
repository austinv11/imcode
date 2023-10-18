#!/usr/bin/env bash

docker build -t austinv11/imcode:latest .
docker push austinv11/imcode:latest

ssh -A varela@juno "module load singularity/3.7.1"\
" && singularity pull docker:://austinv11/imcode:latest"\
" && echo 'bash -c singularity run --bind /work/tansey/sanayeia/IMC_Data/:/data imcode.sig' >> run_imcode.sh"\
" && chmod +x run_imcode.sh"\
" && bsub -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=16] -e imcode.err -o imcode.out ./run_imcode.sh'"
