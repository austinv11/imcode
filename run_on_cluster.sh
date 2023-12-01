#!/usr/bin/env bash

docker build -t austinv11/imcode:latest .
docker push austinv11/imcode:latest

# Copy nextflow and python files to juno
rsync -avzP --progress nextflow/* varelaa@juno-xfer01.mskcc.org:/work/tansey/varelaa/imcode/
rsync -avzP --progress --exclude imc_transformer/__pycache__ imc_transformer varelaa@juno-xfer01:/work/tansey/varelaa/imcode/

ssh -A varelaa@juno "cd /work/tansey/varelaa/imcode"\
" && wget https://raw.githubusercontent.com/tansey-lab/juno-nextflow-template/main/juno_nextflow.sh -O juno_nextflow.sh"\
" && chmod +x juno_nextflow.sh"\
" && rm -rf /work/tansey/varelaa/.singularity/cache"\
" && rm -f /work/tansey/varelaa/tmp-singularity-cachedir/austinv11-imcode-latest.img"\
" && screen ./juno_nextflow.sh run -resume -N varelaa@mskcc.org --data_path /work/tansey/sanayeia/IMC_Data/stacks/ --script_dir imc_transformer/ train.nf"
