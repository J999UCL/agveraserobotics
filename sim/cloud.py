"""Bounded, single-GPU episode batch. All resources belong to j999ucl."""
import os
os.environ['MODAL_PROFILE']='j999ucl'
import json
from pathlib import Path
import sys
import time
import modal
import threading

HERE=Path(__file__).resolve().parent
REV='215abfb217dbac7d5f1273282331b9b1866c0479'
app=modal.App('agverase-mujoco-agent')
volume=modal.Volume.from_name('agverase-mujoco-episodes',create_if_missing=True)
image=(modal.Image.from_registry('ubuntu:22.04',add_python='3.11')
 .apt_install('git','libegl1','libgl1','libglib2.0-0','ffmpeg','ca-certificates')
 .pip_install('uv')
 .run_commands('apt-get update && apt-get install -y --no-install-recommends build-essential linux-libc-dev clang cmake pkg-config libegl1-mesa-dev libgl1-mesa-dev')
 .run_commands('git clone https://github.com/Physical-Intelligence/openpi.git /opt/openpi',f'cd /opt/openpi && git checkout {REV} && git submodule update --init --recursive')
 .run_commands('cd /opt/openpi && CC=gcc CXX=g++ GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev')
 .run_commands('uv venv --python 3.8 /opt/libero-env',
 'CC=gcc CXX=g++ uv pip install --python /opt/libero-env/bin/python -r /opt/openpi/examples/libero/requirements.txt -r /opt/openpi/third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match',
 'CC=gcc CXX=g++ uv pip install --python /opt/libero-env/bin/python -e /opt/openpi/packages/openpi-client -e /opt/openpi/third_party/libero')
 .run_commands("printf 'n\\n' | PYTHONPATH=/opt/openpi/third_party/libero /opt/libero-env/bin/python -c 'import libero.libero'")
 .pip_install('mujoco==4.0.0','numpy==2.4.6','scipy==1.17.1','pillow==13.0.0','msgpack','websockets','dm-tree')
 .env({'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','XLA_PYTHON_CLIENT_PREALLOCATE':'false','OPENPI_DATA_HOME':'/data/weights','NVIDIA_DRIVER_CAPABILITIES':'compute,utility,graphics','OMP_NUM_THREADS':'2'})
 .add_local_dir(str(HERE),'/opt/scene',ignore=['episodes','__pycache__']))

@app.function(image=image,cpu=(1,2),memory=(4096,16384),timeout=1200,retries=0,volumes={'/data':volume})
def prepare():
 import subprocess
 subprocess.run(['/opt/openpi/.venv/bin/python','-c',"from openpi.shared.download import maybe_download; print(maybe_download('gs://openpi-assets/checkpoints/pi05_libero/'))"],cwd='/opt/openpi',check=True,timeout=1000)
 volume.commit()

@app.function(image=image,gpu='A10',cpu=(2,4),memory=(32768,65536),timeout=6900,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':volume})
def run_batch(run_id,attempt=1,max_seconds=1800,episodes=10):
 import subprocess,signal
 sys.path.insert(0,'/opt/scene');sys.path.insert(0,'/opt/openpi/packages/openpi-client/src')
 root=Path('/data/runs')/run_id
 if attempt>1:root=root/f'attempt-{attempt:02}'
 root.mkdir(parents=True,exist_ok=True)
 import hashlib,shutil
 snapshot=root/'source-snapshot';snapshot.mkdir(exist_ok=True)
 for name in ['libero_client.py','libero_task.py','cloud.py','prepare_libero_assets.py']:
  shutil.copy2(Path('/opt/scene')/name,snapshot/name)
 checkpoint=Path('/data/weights/openpi-assets/checkpoints/pi05_libero')
 identity={str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.rglob('*') if p.is_file() and (p.name=='manifest.ocdbt' or p.suffix=='.json')}
 (root/'checkpoint-identity.json').write_text(json.dumps({'revision':REV,'metadata_hashes':identity},indent=2))
 deadline=time.monotonic()+max_seconds-30
 env=dict(os.environ,PYTHONPATH='/opt/openpi/src:/opt/openpi/packages/openpi-client/src')
 server=None
 policy_log=(root/'policy.log').open('w')
 runtime_env=dict(os.environ,PYTHONPATH='/opt/openpi/third_party/libero:/opt/openpi/packages/openpi-client/src:/opt/scene')
 check=subprocess.run(['/opt/libero-env/bin/python','/opt/scene/libero_client.py','camera-check',str(root)],env=runtime_env,capture_output=True,text=True,timeout=180)
 (root/'camera-check.log').write_text(check.stdout+check.stderr)
 volume.commit()
 if check.returncode:
  (root/'status.json').write_text(json.dumps({'status':'error','error':'Native camera check failed before policy startup'}));policy_log.close();volume.commit()
  raise RuntimeError('Native camera check failed before policy startup')
 server=subprocess.Popen(['/opt/openpi/.venv/bin/python','/opt/openpi/scripts/serve_policy.py','--env','LIBERO','--port','8000','--record'],cwd=str(root),env=env,stdout=policy_log,stderr=subprocess.STDOUT)
 try:
  import socket
  for _ in range(180):
   if server.poll() is not None:raise RuntimeError('Policy server exited; inspect policy.log')
   try:
    with socket.create_connection(('127.0.0.1',8000),timeout=1):break
   except OSError:time.sleep(5)
  else:raise TimeoutError('Policy server startup exceeded 15 minutes')
  # The provided example is executed without editing its source.
  runtime_env=dict(os.environ,PYTHONPATH='/opt/openpi/third_party/libero:/opt/openpi/packages/openpi-client/src:/opt/scene')
  # Reuse only a completed unchanged reference run with identical checkpoint identity.
  reference=None
  for previous in sorted(root.parent.glob('attempt-*/baseline.log')):
   identity_file=previous.parent/'checkpoint-identity.json'
   if previous.parent==root or not identity_file.exists():continue
   text=previous.read_text()
   if text.count('INFO:root:Success: True')==10 and 'INFO:root:Total success rate: 1.0' in text and identity_file.read_text()==(root/'checkpoint-identity.json').read_text():reference=previous
  if reference:
   (root/'baseline-reference.json').write_text(json.dumps({'path':str(reference),'sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'successes':10,'trials':10,'revision':REV},indent=2))
  else:
   with (root/'baseline.log').open('w') as log:
    baseline=subprocess.run(['/opt/libero-env/bin/python','/opt/openpi/examples/libero/main.py','--args.host','127.0.0.1','--args.task-suite-name','libero_goal','--args.num-trials-per-task','1','--args.video-out-path',str(root/'baseline-videos')],env=runtime_env,stdout=log,stderr=subprocess.STDOUT,text=True,input="n\n",timeout=min(600,max_seconds//2))
   logs=(root/'baseline.log').read_text()
   if baseline.returncode or 'Success: True' not in logs:raise RuntimeError('Unmodified LIBERO reference failed')
  for command in ['run']:
   with (root/(command+'.log')).open('w') as log:
    result=subprocess.run(['/opt/libero-env/bin/python','-u','/opt/scene/libero_client.py',command,str(root),'--episodes',str(episodes)],env=runtime_env,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=max(30,int(deadline-time.monotonic())))
   print((root/(command+'.log')).read_text()[-4000:],flush=True)
   if result.returncode:raise RuntimeError('LIBERO '+command+' failed; inspect saved log')
  (root/'status.json').write_text(json.dumps({'status':'completed','revision':REV}))
 except Exception as exc:
  (root/'status.json').write_text(json.dumps({'status':'error','error':repr(exc),'revision':REV}));raise
 finally:
  server.terminate()
  try:server.wait(timeout=10)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  policy_log.close();volume.commit()
 return str(root)


@app.function(image=image,cpu=(1,2),memory=(4096,16384),timeout=600,retries=0,volumes={'/data':volume})
def validate_api():
 import subprocess
 output=Path('/data/api-validation');output.mkdir(exist_ok=True)
 result=subprocess.run(['/opt/libero-env/bin/python','/opt/scene/libero_client.py','validate',str(output)],env=dict(os.environ,PYTHONPATH='/opt/openpi/third_party/libero:/opt/openpi/packages/openpi-client/src:/opt/scene'),capture_output=True,text=True,timeout=550)
 (output/'validation.log').write_text(result.stdout+result.stderr);volume.commit();print(result.stdout+result.stderr,flush=True)
 if result.returncode:raise RuntimeError('CPU API validation failed')

def main():
 import argparse,subprocess,uuid
 p=argparse.ArgumentParser();p.add_argument('command',choices=['run','resume','status','fetch','stop','validate']);p.add_argument('run_id',nargs='?');p.add_argument('--gpu',choices=['L4','A10','T4'],default='A10');p.add_argument('--episodes',type=int,choices=range(1,11),default=10);p.add_argument('--seconds',type=int,default=3600);a=p.parse_args()
 runtime=HERE.parent/'.runtime/mujoco-agent';runtime.mkdir(parents=True,exist_ok=True)
 ledger=runtime/'budget.json';old=json.loads(ledger.read_text()) if ledger.exists() else {'reserved':0,'runs':[]}
 if a.command=='validate':
  entry=old['runs'][-1];count=entry.get('api_validation_attempts',0)
  if count>=3:raise RuntimeError('CPU validation allowance exhausted')
  entry['api_validation_attempts']=count+1;ledger.write_text(json.dumps(old,indent=2))
  with modal.enable_output(),app.run():validate_api.remote()
  return
 if a.command=='status':print(json.dumps(old,indent=2));return
 if a.command=='stop':
  import signal
  for entry in old['runs']:
   if entry.get('driver_pid'):
    process=subprocess.run(['ps','-p',str(entry['driver_pid']),'-o','command='],capture_output=True,text=True)
    if str(HERE/'cloud.py') in process.stdout:os.kill(entry['driver_pid'],signal.SIGINT)
   if entry.get('app_id'):subprocess.run([sys.executable,'-m','modal','app','stop',entry['app_id'],'--yes'],check=True)
  return
 if a.command=='fetch':
  rid=a.run_id or old['runs'][-1]['id'];destination=HERE/'episodes'/rid;destination.mkdir(parents=True,exist_ok=True)
  subprocess.run([sys.executable,'-m','modal','volume','get','--force','agverase-mujoco-episodes',f'runs/{rid}/',str(destination.parent)],check=True);return
 # Hold the process lock throughout cloud execution to prevent parallel GPU batches.
 import fcntl
 runner_lock=(runtime/'runner.lock').open('a+')
 try:fcntl.flock(runner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('A cloud batch is already running')
 old=json.loads(ledger.read_text()) if ledger.exists() else {'reserved':0,'runs':[]}
 # Reserve the complete budget before any cloud work; no automatic second batch.
 if a.command=='resume':
  entry=old['runs'][-1]
  if entry.get('gpu_seconds_reserved',0)>=6600 or entry.get('prep_attempts',1)>=14:raise RuntimeError('No additional GPU attempts or preparation retries permitted')
  rid=entry['id'];entry['prep_attempts']=entry.get('prep_attempts',1)+1;entry['status']='resuming'
 else:
  if old['reserved']+5>5:raise RuntimeError('The $5 batch budget is already reserved. Inspect results; no automatic budget reset.')
  rid=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
  old['reserved']+=5;entry={'id':rid,'reserved_usd':5,'status':'starting','prep_attempts':1};old['runs'].append(entry)
 entry['driver_pid']=os.getpid();ledger.write_text(json.dumps(old,indent=2))
 def expire():os.kill(os.getpid(),__import__('signal').SIGINT)
 whole_run_watchdog=threading.Timer(8400,expire);whole_run_watchdog.daemon=True;whole_run_watchdog.start()
 try:
  with modal.enable_output(),app.run():
   entry['app_id']=app.app_id;ledger.write_text(json.dumps(old,indent=2))
   watchdog=threading.Timer(7800,lambda:subprocess.run([sys.executable,'-m','modal','app','stop',app.app_id,'--yes']))
   watchdog.daemon=True;watchdog.start()
   try:
    prepare.remote()
    duration=min(max(600,min(a.seconds,3600)),7200-entry.get('gpu_seconds_reserved',0)-300)
    if duration<600:raise RuntimeError('Insufficient cumulative GPU allowance')
    entry['gpu_started']=True;entry['gpu_seconds_reserved']=entry.get('gpu_seconds_reserved',0)+duration+300;entry['inference_attempts']=entry.get('inference_attempts',0)+1
    ledger.write_text(json.dumps(old,indent=2))
    entry['gpu']=a.gpu;ledger.write_text(json.dumps(old,indent=2))
    worker=run_batch.with_options(timeout=duration,gpu=a.gpu)
    call=worker.spawn(rid,entry['inference_attempts'],duration,a.episodes);waiting_since=time.monotonic()
    entry['call_id']=call.object_id;ledger.write_text(json.dumps(old,indent=2))
    while worker.get_current_stats().num_running_inputs<1:
     try:
      call.get(timeout=0);break
     except TimeoutError:pass
     if time.monotonic()-waiting_since>600:
      call.cancel(terminate_containers=True);raise TimeoutError('GPU scheduling exceeded ten minutes')
     time.sleep(2)
    result_path=call.get()
   finally:watchdog.cancel()
  entry['status']='completed'
 except BaseException:
  entry['status']='failed';raise
 finally:
  whole_run_watchdog.cancel();entry['driver_pid']=None
  ledger.write_text(json.dumps(old,indent=2))
  print('Run:',rid,'; fetch recordings with ./sim/cloud.sh fetch',rid)
 if entry['status']=='completed':
  destination=HERE/'episodes'/rid;destination.mkdir(parents=True,exist_ok=True)
  # Fetch the just-completed attempt; explicit `fetch` retains full-history retrieval.
  remote=f'runs/{rid}/';download_parent=destination.parent
  if entry['inference_attempts']>1:remote+=Path(result_path).name+'/';download_parent=destination
  subprocess.run([sys.executable,'-m','modal','volume','get','--force','agverase-mujoco-episodes',remote,str(download_parent)],check=True)
  current=destination/Path(result_path).name if entry['inference_attempts']>1 else destination
  episodes=sorted((current/'recordings').glob('ep_*'))
  if episodes:subprocess.Popen([str(HERE/'replay.sh'),str(episodes[-1])])
  else:print('No vendor recordings were produced; inspect the saved logs.')

if __name__=='__main__':main()
