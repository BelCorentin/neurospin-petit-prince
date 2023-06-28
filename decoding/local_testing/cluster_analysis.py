import submitit
import sys
import argparse
from dataset import analysis_subject, get_subjects, get_path
import mne
mne.set_log_level(verbose='error')

criterions = ('bow', 'embeddings', 'only1', 'only2', 'only3', 'only4', 'only5')
modalities = ('visual', 'auditory')
starts = ('onset', 'offset')


def get_parser(args):
    parser = argparse.ArgumentParser(description='Launch the LPP analysis')

    parser.add_argument('--nb_cpu', type=int, default=32, help="cpus")
    parser.add_argument('--nb_nodes', type=int, default=1, help="nb nodes")
    parser.add_argument('--criterion', type=str,
                        default="embeddings", help="embeddings / wlength")
    parser.add_argument('--modality', type=str,
                        default="visual", help="auditory/visual")
    parser.add_argument('--start', type=str,
                        default="onset", help="onset/offset")
    parser.add_argument('--level', type=str,
                        default="word", help="word/sentence/constituent")
    parser.add_argument('--total', type=bool,
                        default=True, help="True/False")
    args_class = parser.parse_args(args)
    return args_class


def main(args):
    args_class = get_parser(args)

    # Create the submitit job
    executor = submitit.AutoExecutor(folder="LPP_logs")
    NUM_TASKS_PER_NODE = args_class.nb_cpu
    NUM_NODES = args_class.nb_nodes
    criterion = args_class.criterion
    modality = args_class.modality
    start = args_class.start
    level = args_class.level
    total = args_class.total

    subjects = get_subjects(get_path(modality))

    executor.update_parameters(slurm_partition="cpu_p1",  # gpu_p2
                               nodes=NUM_NODES,
                               cpus_per_task=NUM_TASKS_PER_NODE,
                               timeout_min=90,
                               account="qtr@cpu")  # fqt@gpu

    class Task:
        def __call__(self, subject):
            if total:
                # Iterate on modalities, starts, decoding criterions
                for modality in modalities:
                    for start in starts:
                        for criterion in criterions:
                            analysis_subject(subject, modality, start, level, criterion)
            else:
                print('Decoding criterion chosen: ', criterion)
                print('Decoding modality chosen: ', modality)
                print(f'Doing it for subject: {subject}')
                

                analysis_subject(subject, modality, start, level, criterion)

    task = Task()
    jobs = []
    with executor.batch():
        for subject in subjects[2:58]:
            job = executor.submit(task, subject)
            jobs.append(job)
    submitit.helpers.monitor_jobs(jobs)
    outputs = [job.result() for job in jobs]
    print("Finished executing the analysis")
    print(outputs)


if __name__ == "__main__":
    main(sys.argv[1:])
