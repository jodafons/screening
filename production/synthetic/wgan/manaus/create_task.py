import json, os


# create jobs
def create_jobs(output_path = os.getcwd()+'/jobs'):
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
    for test in range(10):
        for sort in range(9):
            d = {'sort':sort,'test':test}
            o = output_path + '/job.test_%d.sort_%d.json'%(test,sort)
            with open(o, 'w') as f:
                json.dump(d, f)


def create_task( task_name, production_card, dry_run=False):
  
  local_path  = os.getcwd()
  proj_path   = os.environ["PROJECT_DIR"]
  image_path  = proj_path + '/images/wgan_base.sif'
  config_path = local_path + '/configs/'+ production_card
  repo_path   = os.environ['REPO_DIR']
  job_path    = os.getcwd()+'/jobs'


  exec_cmd  = f"cd {repo_path} && source envs.sh && source activate.sh && cd %JOB_WORKAREA\n"
  exec_cmd += f"run_train.py --j %IN -c {config_path} --wandb_taskname {task_name}\n"
  envs      = { 'DATA_DIR':os.environ['DATA_DIR'] }
  binds     = {"/mnt/brics_data":"/mnt/brics_data", "/home":"/home"}
  command = f"""maestro task create \
    -t {task_name} \
    -i {job_path} \
    --exec "{exec_cmd}" \
    --envs "{str(envs)}" \
    --partition gpu-large \
    --image {image_path} \
    --binds "{str(binds)}" \
    """
  print(command)
  if not dry_run:
    os.system(command)


#
# create production
#

dry_run = False

create_jobs()
#create_task( 'user.joao.pinto.task.Manaus.manaus.wgan_v2_tb'       , 'manaus_tb_card.json'      , dry_run=dry_run )
#create_task( 'user.joao.pinto.task.Manaus.manaus.wgan_v2_notb'     , 'manaus_notb_card.json'    , dry_run=dry_run )

#create_task( 'user.joao.pinto.task.Manaus.c_manaus.wgan_v2_tb'     , 'c_manaus_tb_card.json'    , dry_run=dry_run )
create_task( 'user.joao.pinto.task.Manaus.c_manaus.wgan_v2_notb'   , 'c_manaus_notb_card.json'  , dry_run=dry_run )

  

