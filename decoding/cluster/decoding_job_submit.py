import submitit
from dataset import read_raw

# Define the function that will be run in parallel:
# For each subject, we can decode from 6 sub-epoched sets

# TODO finalize
def analysis(sub):
    # load MEG data
    raw = read_raw(sub)
    raw = rec.raw()
    raw.load_data()
    raw.filter(.5, 20., n_jobs=-1);

    # get metadata
    meta = add_syntax(rec)

    # epoch
    def mne_events(meta):
        events = np.ones((len(meta), 3), dtype=int)
        events[:, 0] = meta.start*raw.info['sfreq']
        return dict(events=events, metadata=meta.reset_index())

    epochs = mne.Epochs(raw, **mne_events(meta), decim=20, tmin=-.2, tmax=1.5, preload=True)
    epochs = epochs['kind=="word"']
    
    scores = dict()
    scores['n_closing'] = decod(epochs, 'n_closing')
    scores['n_closing_notlast'] = decod(epochs['content_word and not is_last_word'], 'n_closing')
    scores['n_closing_noun_notlast'] = decod(epochs['pos=="NC" and not is_last_word'], 'n_closing')
    return scores


## Create the submitit job
# Using Jean Zay parameters
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(slurm_partition="cpu_p1", #gpu_p2
                           cpus_per_task=32, #10
                           timeout_min=30, #15 min for each checkpoint 700
                           account="qtr@cpu") #


recs = list(Pallier2023Recording.iter())[:150]
jobs = executor.map_array(analysis, recs)

results = []
for job in jobs:
    if job.state == 'COMPLETED':
        results.append(job.results()[0])


results = pd.DataFrame(results)


times = np.linspace(-.2, 1.5, 86)
for key in results.keys():
    plt.fill_between(times, np.mean(results[key].values, 0), alpha=.5, label=key)
plt.axhline(0, color='k', ls=':')
plt.legend()


class Task:
    def __call__(self,model):

        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")

        model_dir = os.path.join(args_class.modelpath, model)
        if args_class.save is None:
            save_dir = os.path.join("outputs", model)
        else:
            save_dir = args_class.save

        if "LOCAL_RANK" in os.environ.keys():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            # We don't need to set the specific environment but assign the right gpu
            # to this process by setting the local_rank

        ## Load the set of checkpoint names
        chkList = os.listdir(model_dir)
        lmodels = list(filter(lambda e: e.__contains__("checkpoint"), chkList))
        lmodels = np.array(lmodels)[np.argsort(np.array([int(l.split("checkpoint-")[1]) for l in lmodels]))]
        chkList = np.array(["initialmodel"] + list(lmodels))
        nb_total_checkpoint = len(chkList)
        chunk_size = np.ceil(nb_total_checkpoint / (args_class.nb_nodes * args_class.nb_gpu))

        postAnalyzer = PostAnalyzer(dataset_names, model_dir, save_dir,
                                    nb_total_checkpoint, chunk_size)
        # The checkpoints are a function of the total rank in the set of subprocess.
        checkpoints_id = np.arange(int(dist_env.rank)*chunk_size,
                                   int(np.min([(int(dist_env.rank)+1)*chunk_size,nb_total_checkpoint])))
        print("checkpoint managed by this process:")
        print(checkpoints_id)

        checkpoints_names = np.array(chkList)[np.array(checkpoints_id,dtype=int)]

        postAnalyzer.multiprocess_init(np.array(checkpoints_id,dtype=int), checkpoints_names)

        print("extraction of quantize codevectors")
        postAnalyzer.save_quantize()
        return True

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)

task = Task()
jobs = []
with executor.batch():
    for model in models:
        job = executor.submit(task, model)
        jobs.append(job)
submitit.helpers.monitor_jobs(jobs)
outputs = [job.result() for job in jobs]