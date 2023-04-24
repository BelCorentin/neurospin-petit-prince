import submitit

## Create the submitit job
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(slurm_partition="gpu_p2", #gpu_p2
                           cpus_per_task=32, #10
                           timeout_min=15, #15 min for each checkpoint 700
                           account="qtr@cpu") #


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