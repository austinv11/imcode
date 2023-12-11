nextflow.enable.dsl=2

log.info """\
Training...
"""

process prepareData {
    memory '128 GB'
    time '24h'
    cpus 2
    executor = 'lsf'
    container = 'docker://austinv11/imcode:latest'

    input:
        path data_path
        path script_dir
    output:
        path('compiled_data', type: 'dir')
        path script_dir

    """
    python-conda imc_transformer/data.py ${data_path} compiled_data --image_size=256 --patch_overlap=0.005
    """
}

process trainAndEvaluateModel {
    // GPU settings
//     memory '32 GB'
//     time '6 h'
//     cpus 4
//     queue 'gpuqueue'
//     clusterOptions '-gpu "num=1:gmem=8"'
//     containerOptions = '--nv'  // GPU Access for singularity


    //CPU Settings
    memory '64 GB'
    time '120 h'
    cpus 10

    executor = 'lsf'
    container = 'docker://austinv11/imcode:latest'

    input:
        path compiled_data
        path script_dir
    output:
//         path('mae_pretrain/last.ckpt', type: 'file')
        path('reg_finetune/last.ckpt', type: 'file')
        path('lightning_logs', type: 'dir')
        path('model_outputs', type: 'dir')

    """
    python-conda imc_transformer/model.py ${compiled_data} model_outputs --epochs=25 --random_erasing=0 --random_flip_augmentation=0 --random_rotate_augmentation=0 --random_shear_augmentation=0 --random_translate_augmentation=0 --random_scale_augmentation=0 --spike_in_channels=5
    """
}

workflow {
    def raw_data_file = Channel.fromPath(params.data_path, type: 'dir', followLinks: true, checkIfExists: true)
    def script_dir = Channel.fromPath(params.script_dir, type: 'dir', followLinks: true, checkIfExists: true)
    prepareData(raw_data_file, script_dir) | trainAndEvaluateModel
}
